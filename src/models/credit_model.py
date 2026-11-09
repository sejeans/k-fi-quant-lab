"""모듈 3: 회사채 스프레드 방향 예측 + 급확대 조기경보.

범위 한계 (반드시 명시)
- ECOS 공개 데이터에는 등급×만기 매트릭스가 없다. AA-/BBB- 3년 2개 버킷만 가능하다.
  기획서 5.2의 등급별 확장은 라이선스 데이터(민평 3사, 개별 채권)가 필요하다.

모델 (기획서 5.4)
1. ECM  : 스프레드가 장기평균에서 벌어진 정도가 되돌아오는지 (오차수정항)
2. Elastic Net : 20일 변화 방향
3. 급확대 이진분류 : 20일 변화 > 롤링 90분위  (class_weight 사용, SMOTE 금지)
   -> 시계열 구조를 훼손하지 않기 위해 합성 오버샘플링은 쓰지 않는다(기획서 5.4).
4. 평가 : 불균형이므로 정확도가 아니라 PR-AUC / Recall / Precision / Brier
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
import statsmodels.api as sm
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.walkforward import split_live, walk_forward  # noqa: E402

PROC = ROOT / "data" / "processed"

H = 20
HOLDOUT_YEARS = 2
SEED = 42
BUCKETS = {"cs_aa3y_bp": "AA- 3년", "cs_bbb3y_bp": "BBB- 3년"}


def make_features(panel: pd.DataFrame, target: str) -> pd.DataFrame:
    s = panel[target]
    X = pd.DataFrame(index=panel.index)
    X["level"] = s
    X["d5"] = s.diff(5)
    X["d20"] = s.diff(20)
    # 평균회귀(오차수정) 항: 장기평균 대비 괴리
    X["ecm_z250"] = (s - s.rolling(250).mean()) / s.rolling(250).std()
    X["ecm_z60"] = (s - s.rolling(60).mean()) / s.rolling(60).std()
    X["vol60"] = s.diff().rolling(60).std()

    X["ktb3y"] = panel["ktb3y_pct"]
    X["ktb3y_d20"] = panel["ktb3y_pct"].diff(20) * 100
    X["slope_3s10s_bp"] = panel["slope_3s10s_bp"]
    X["rate_vol20"] = panel["ktb3y_vol20_bp"]
    X["cd_msb_bp"] = panel["cd_msb_bp"]       # 유동성 경색 지표
    X["cp_cd_bp"] = panel["cp_cd_bp"]
    X["base_gap_1y"] = panel["ktb1y_base_gap_bp"]
    # 다른 등급 스프레드(교차 정보)
    other = "cs_bbb3y_bp" if target == "cs_aa3y_bp" else "cs_aa3y_bp"
    X["other_z250"] = (panel[other] - panel[other].rolling(250).mean()) / panel[other].rolling(250).std()
    return X


def run_ecm(panel: pd.DataFrame, target: str) -> None:
    """오차수정모형: d(spread) ~ 괴리 + 금리변화. 되돌림 계수 부호·유의성 확인."""
    s = panel[target]
    lt = s.rolling(250).mean()
    df = pd.DataFrame({
        "dy": s.diff(H).shift(-H),          # 향후 H일 변화
        "ecm": (s - lt),                    # 장기평균 대비 괴리
        "drate": panel["ktb3y_pct"].diff(20) * 100,
        "dvol": panel["ktb3y_vol20_bp"],
    }).dropna()
    X = sm.add_constant(df[["ecm", "drate", "dvol"]])
    m = sm.OLS(df["dy"], X).fit(cov_type="HAC", cov_kwds={"maxlags": H})
    print(f"  [ECM] n={int(m.nobs)}  R2={m.rsquared:.3f}")
    print(f"    {'변수':<8}{'계수':>10}{'t값':>9}{'p값':>9}")
    for nm in ["ecm", "drate", "dvol"]:
        print(f"    {nm:<8}{m.params[nm]:>10.4f}{m.tvalues[nm]:>9.2f}{m.pvalues[nm]:>9.4f}")
    b = m.params["ecm"]
    print(f"    -> 오차수정계수 {b:+.4f} "
          f"({'평균회귀 확인' if b < 0 and m.pvalues['ecm'] < 0.05 else '유의한 평균회귀 아님'})")


def enet_model():
    return ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=SEED,
                        max_iter=5000, n_jobs=-1)


def logit_model():
    """급확대 이진분류. class_weight로 불균형 처리(SMOTE 금지)."""
    return LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced", random_state=SEED)


if __name__ == "__main__":
    np.random.seed(SEED)
    panel = pd.read_parquet(PROC / "panel.parquet")
    split = panel.index.max() - pd.DateOffset(years=HOLDOUT_YEARS)
    print(f"최종 테스트 구간(봉인): {split:%Y-%m-%d} ~ {panel.index.max():%Y-%m-%d}\n")

    summary = []
    for tgt, label in BUCKETS.items():
        print("=" * 60)
        print(f"{label} 스프레드  (현재 {panel[tgt].iloc[-1]:.1f}bp)")
        print("=" * 60)
        run_ecm(panel, tgt)

        X = make_features(panel, tgt)

        # (1) 방향 예측
        y = panel[tgt].shift(-H) - panel[tgt]
        res, _ = split_live(walk_forward(X, y, split, H, enet_model)
                            .rename(columns={"pred": "enet"}))
        if not res.empty:
            rw = np.sqrt((res["actual"] ** 2).mean())
            md = np.sqrt(((res["actual"] - res["enet"]) ** 2).mean())
            hit = (np.sign(res["enet"]) == np.sign(res["actual"])).mean() * 100
            corr = res["enet"].corr(res["actual"])
            print(f"\n  [방향 예측] n={len(res)}")
            print(f"    RW RMSE {rw:.2f}bp / 모델 {md:.2f}bp  (개선 {(rw-md)/rw*100:+.2f}%)")
            print(f"    corr {corr:+.3f}, 방향정확도 {hit:.1f}%")
            res.to_parquet(PROC / f"credit_reg_{tgt}.parquet")

        # (2) 급확대 조기경보
        fwd = panel[tgt].shift(-H) - panel[tgt]
        thresh = fwd.rolling(500, min_periods=250).quantile(0.90)
        y_bin = (fwd > thresh).astype(float)
        y_bin[fwd.isna() | thresh.isna()] = np.nan
        clf, _ = split_live(walk_forward(X, y_bin, split, H, logit_model, proba=True)
                            .rename(columns={"pred": "prob"}))
        clf = clf.dropna()
        if not clf.empty and clf["actual"].sum() > 0:
            ap = average_precision_score(clf["actual"], clf["prob"])
            base_rate = clf["actual"].mean()
            prec, rec, thr = precision_recall_curve(clf["actual"], clf["prob"])
            f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
            bi = int(np.nanargmax(f1))
            print(f"\n  [급확대 조기경보] n={len(clf)}, 실제 급확대 {int(clf['actual'].sum())}회 "
                  f"({base_rate:.1%})")
            print(f"    PR-AUC {ap:.3f}  (무작위 기준선 {base_rate:.3f}, "
                  f"{'개선' if ap > base_rate else '악화'} {ap/base_rate:.2f}배)")
            print(f"    최적 F1 지점: Precision {prec[bi]:.3f}, Recall {rec[bi]:.3f}, "
                  f"임계 {thr[min(bi, len(thr)-1)]:.3f}")
            print(f"    Brier {((clf['prob']-clf['actual'])**2).mean():.4f}")
            clf.to_parquet(PROC / f"credit_clf_{tgt}.parquet")
            summary.append({"bucket": label, "pr_auc": ap, "base": base_rate,
                            "hit": hit, "corr": corr})
        print()

    # 위기구간 case study
    print("=" * 60)
    print("위기구간 case study (기획서 5.7)")
    print("=" * 60)
    for name, a, b in [("2020 코로나", "2020-02-01", "2020-06-30"),
                       ("2022 레고랜드", "2022-09-01", "2023-01-31")]:
        w = panel.loc[a:b, ["cs_aa3y_bp", "cs_bbb3y_bp"]]
        if w.empty:
            continue
        print(f"\n  [{name}]  {a} ~ {b}")
        for c, lbl in BUCKETS.items():
            v = w[c]
            print(f"    {lbl}: 시작 {v.iloc[0]:6.1f} -> 최대 {v.max():6.1f}bp "
                  f"(확대폭 {v.max()-v.iloc[0]:+6.1f}bp, 최대일 {v.idxmax():%Y-%m-%d})")

    pd.DataFrame(summary).to_csv(PROC / "credit_summary.csv", index=False)
    print(f"\n저장: {PROC}")
