"""모듈 1 예측: NS 요인의 5일/20일 변화 예측 (walk-forward).

검증 설계 (기획서 2.1 / 계획 Block 2-2)
- shuffle 금지. expanding window walk-forward.
- 최근 2년(HOLDOUT_YEARS)은 최종 테스트 구간으로 봉인하고 여기서만 성능을 보고한다.
- 모든 특징은 t시점까지의 정보만 사용. 타깃은 t+h. (look-ahead 차단)
- 기준선 Random Walk = '변화량 0 예측'. 요인이 random walk면 이걸 못 이기는 게 정상이다.

최종 예측(final) = 기울기 요인만 ENet과 AR의 평균, 수준·곡률은 0(RW)
- 근거: src/experiments/ 의 비교 실험. 2019~봉인 직전(dev)에서 고르고 봉인 구간에서 확인했다.
  수준·곡률 예측을 쓰면 3~10년물 RMSE가 RW보다 나빠졌고, 기울기 요인만 남기면
  두 구간 모두 1·3·10년물이 RW 이상이었다. 딥러닝(LSTM/CNN), 표본 확장(2001~, 미국 합동),
  미국 전일 금리 특징은 이 설정을 넘지 못했다.
"""

from __future__ import annotations

import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):      # Windows cp949 콘솔에서 한글/기호 깨짐 방지
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LinearRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.nelson_siegel import CURVE, ns_loadings  # noqa: E402
from src.models.walkforward import MIN_TRAIN, REFIT_EVERY, split_live, walk_forward  # noqa: E402

PROC = ROOT / "data" / "processed"
FACTORS = ["beta1_level", "beta2_slope", "beta3_curv"]
HORIZONS = [5, 20]
HOLDOUT_YEARS = 2
SEED = 42


def make_features(panel: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """t시점까지 알 수 있는 정보만으로 특징 생성."""
    X = pd.DataFrame(index=factors.index)
    for f in FACTORS:
        s = factors[f]
        X[f] = s
        X[f"{f}_d5"] = s.diff(5)          # 최근 모멘텀
        X[f"{f}_d20"] = s.diff(20)
        X[f"{f}_z60"] = (s - s.rolling(60).mean()) / s.rolling(60).std()
    X["cd_msb_bp"] = panel["cd_msb_bp"]
    X["cp_cd_bp"] = panel["cp_cd_bp"]
    X["ktb1y_base_gap_bp"] = panel["ktb1y_base_gap_bp"]
    X["vol20"] = panel["ktb3y_vol20_bp"]
    return X


def enet_model():
    return ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=SEED,
                        max_iter=5000, n_jobs=-1)


def forecast_factor(X: pd.DataFrame, f: str, h: int, split: pd.Timestamp) -> pd.DataFrame:
    """요인 f의 h일 변화 예측. AR = 그 요인의 레벨/모멘텀 4개만 쓴 선형모형."""
    y = X[f].shift(-h) - X[f]                       # t+h 변화량
    en = walk_forward(X, y, split, h, enet_model)
    ar = walk_forward(X[[f, f"{f}_d5", f"{f}_d20", f"{f}_z60"]], y, split, h, LinearRegression)
    ar_p = ar["pred"].reindex(en.index)
    final = (en["pred"] + ar_p) / 2 if f == "beta2_slope" else en["pred"] * 0.0
    return pd.DataFrame({"actual": en["actual"], "rw": 0.0,
                         "ar": ar_p, "enet": en["pred"], "final": final})


def significance(pred: np.ndarray, act: np.ndarray, h: int) -> dict:
    """겹치는 h일 예측의 유의성.
    - DM: 손실차(RW 제곱오차 - 모델 제곱오차) 평균을 Newey-West(lag=h-1) 분산으로 검정(양측).
    - 방향: h일 간격으로 뽑은 겹치지 않는 표본의 이항검정(단측). 시작 오프셋 h개의 중앙값.
    """
    from scipy import stats
    d = act**2 - (act - pred) ** 2
    n, dc = len(d), d - d.mean()
    v = dc @ dc / n + sum(2 * (1 - l / h) * (dc[l:] @ dc[:-l]) / n for l in range(1, h))
    dm_p = 2 * (1 - stats.norm.cdf(abs(d.mean() / np.sqrt(v / n))))
    hits, ps = [], []
    for o in range(h):
        p, a = pred[o::h], act[o::h]
        m = p != 0
        k = int((np.sign(p[m]) == np.sign(a[m])).sum())
        hits.append(k / max(m.sum(), 1) * 100)
        ps.append(stats.binomtest(k, m.sum(), 0.5, alternative="greater").pvalue if m.sum() else np.nan)
    return {"dm_p": dm_p, "n_indep": n // h, "hit_indep_pct": float(np.median(hits)),
            "binom_p": float(np.nanmedian(ps))}


def score(res: pd.DataFrame) -> pd.DataFrame:
    """RMSE/MAE/방향정확도. RW 대비 개선율을 함께 낸다."""
    rows = []
    rw_rmse = np.sqrt(((res["actual"] - res["rw"]) ** 2).mean())
    for m in ["rw", "ar", "enet", "final"]:
        e = res["actual"] - res[m]
        rmse = np.sqrt((e**2).mean())
        # 방향정확도: 예측이 0이면(RW) 방향 판단 없음
        if m == "rw":
            hit = np.nan
        else:
            mask = res[m] != 0
            if not mask.any():                       # final의 수준·곡률 = RW
                rows.append({"model": m, "rmse": rw_rmse, "mae": res["actual"].abs().mean(),
                             "vs_rw_%": 0.0, "hit_%": np.nan})
                continue
            hit = (np.sign(res.loc[mask, m]) == np.sign(res.loc[mask, "actual"])).mean() * 100
        rows.append({
            "model": m,
            "rmse": rmse,
            "mae": e.abs().mean(),
            "vs_rw_%": (rw_rmse - rmse) / rw_rmse * 100,
            "hit_%": hit,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    np.random.seed(SEED)
    panel = pd.read_parquet(PROC / "panel.parquet")
    factors = pd.read_parquet(PROC / "ns_factors.parquet")
    X = make_features(panel, factors)

    split = factors.index.max() - pd.DateOffset(years=HOLDOUT_YEARS)
    print(f"최종 테스트 구간(봉인): {split:%Y-%m-%d} ~ {factors.index.max():%Y-%m-%d}")
    print(f"학습: expanding window, 최소 {MIN_TRAIN}일, {REFIT_EVERY}일마다 재학습\n")

    all_res = {}
    for h in HORIZONS:
        print(f"===== {h}영업일 예측 =====")
        for f in FACTORS:
            res, live = split_live(forecast_factor(X, f, h, split))
            live.to_parquet(PROC / f"live_{f}_h{h}.parquet")
            if res.empty:
                print(f"  {f}: 표본 부족")
                continue
            all_res[(h, f)] = res
            tab = score(res)
            print(f"\n  [{f}]  n={len(res)}")
            print(tab.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
        print()

    # 요인 예측 -> 만기별 금리 변화 복원 (기획서 3.3-6)
    lam = float(pd.read_json(PROC / "ns_config.json", typ="series")["lambda"])
    print("\n===== 만기별 금리 변화 예측 복원 =====")
    for h in HORIZONS:
        keys = [(h, f) for f in FACTORS if (h, f) in all_res]
        if len(keys) < 3:
            continue
        common = all_res[keys[0]].index
        for k in keys[1:]:
            common = common.intersection(all_res[k].index)

        taus = np.array([t for c, t in CURVE])
        L = ns_loadings(taus, lam)
        names = [c for c, _ in CURVE]

        B_pred = np.column_stack([all_res[(h, f)].loc[common, "final"] for f in FACTORS])
        B_act = np.column_stack([all_res[(h, f)].loc[common, "actual"] for f in FACTORS])
        dy_pred, dy_act = (B_pred @ L.T) * 100, (B_act @ L.T) * 100   # bp

        print(f"\n  {h}영업일 (n={len(common)})  단위 bp")
        print(f"  {'만기':<12}{'RW RMSE':>10}{'모델 RMSE':>11}{'개선%':>8}{'방향%':>8}{'DM p':>8}{'독립방향%':>10}{'이항p':>8}")
        rows = []
        for j, nm in enumerate(names):
            rw = np.sqrt((dy_act[:, j] ** 2).mean())
            md = np.sqrt(((dy_act[:, j] - dy_pred[:, j]) ** 2).mean())
            hit = (np.sign(dy_pred[:, j]) == np.sign(dy_act[:, j])).mean() * 100
            imp = (rw - md) / rw * 100
            sg = significance(dy_pred[:, j], dy_act[:, j], h)
            print(f"  {nm:<12}{rw:>10.2f}{md:>11.2f}{imp:>8.2f}{hit:>8.1f}{sg['dm_p']:>8.3f}"
                  f"{sg['hit_indep_pct']:>10.1f}{sg['binom_p']:>8.3f}")
            rows.append({"tenor": nm, "h": h, "rw_rmse_bp": rw, "model_rmse_bp": md,
                         "improve_pct": imp, "hit_pct": hit, **sg})
        pd.DataFrame(rows).to_csv(PROC / f"curve_rmse_h{h}.csv", index=False)

    for (h, f), r in all_res.items():
        r.to_parquet(PROC / f"fc_{f}_h{h}.parquet")
    print(f"\n저장 완료: {PROC}")
