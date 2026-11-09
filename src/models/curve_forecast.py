"""모듈 1 예측: NS 요인의 5일/20일 변화 예측 (walk-forward).

검증 설계 (기획서 2.1 / 계획 Block 2-2)
- shuffle 금지. expanding window walk-forward.
- 최근 2년(HOLDOUT_YEARS)은 최종 테스트 구간으로 봉인하고 여기서만 성능을 보고한다.
- 모든 특징은 t시점까지의 정보만 사용. 타깃은 t+h. (look-ahead 차단)
- 기준선 Random Walk = '변화량 0 예측'. 요인이 random walk면 이걸 못 이기는 게 정상이다.
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
    return pd.DataFrame({"actual": en["actual"], "rw": 0.0,
                         "ar": ar["pred"].reindex(en.index), "enet": en["pred"]})


def score(res: pd.DataFrame) -> pd.DataFrame:
    """RMSE/MAE/방향정확도. RW 대비 개선율을 함께 낸다."""
    rows = []
    rw_rmse = np.sqrt(((res["actual"] - res["rw"]) ** 2).mean())
    for m in ["rw", "ar", "enet"]:
        e = res["actual"] - res[m]
        rmse = np.sqrt((e**2).mean())
        # 방향정확도: 예측이 0이면(RW) 방향 판단 없음
        if m == "rw":
            hit = np.nan
        else:
            mask = res[m] != 0
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

        B_pred = np.column_stack([all_res[(h, f)].loc[common, "enet"] for f in FACTORS])
        B_act = np.column_stack([all_res[(h, f)].loc[common, "actual"] for f in FACTORS])
        dy_pred, dy_act = (B_pred @ L.T) * 100, (B_act @ L.T) * 100   # bp

        print(f"\n  {h}영업일 (n={len(common)})  단위 bp")
        print(f"  {'만기':<12}{'RW RMSE':>10}{'모델 RMSE':>11}{'개선%':>8}{'방향%':>8}")
        rows = []
        for j, nm in enumerate(names):
            rw = np.sqrt((dy_act[:, j] ** 2).mean())
            md = np.sqrt(((dy_act[:, j] - dy_pred[:, j]) ** 2).mean())
            hit = (np.sign(dy_pred[:, j]) == np.sign(dy_act[:, j])).mean() * 100
            imp = (rw - md) / rw * 100
            print(f"  {nm:<12}{rw:>10.2f}{md:>11.2f}{imp:>8.2f}{hit:>8.1f}")
            rows.append({"tenor": nm, "h": h, "rw_rmse_bp": rw, "model_rmse_bp": md,
                         "improve_pct": imp, "hit_pct": hit})
        pd.DataFrame(rows).to_csv(PROC / f"curve_rmse_h{h}.csv", index=False)

    for (h, f), r in all_res.items():
        r.to_parquet(PROC / f"fc_{f}_h{h}.parquet")
    print(f"\n저장 완료: {PROC}")
