"""Nelson-Siegel 커브 적합 + PCA 교차검증.

핵심 설계 (계획 Block 2-1)
- lambda를 고정하면 loading이 상수가 되어 beta1/2/3는 '선형' 최소제곱으로 풀린다.
  매일 비선형 최적화를 돌리지 않으므로 4,000일 적합이 초 단위로 끝나고 발산 위험도 없다.
- lambda는 전체 표본 SSE를 최소화하는 값을 그리드서치로 한 번 정해 고정한다.
- 만기별 관측 가능 여부가 날마다 다르다(2년물 2021~, 30년물 2012~).
  따라서 매일 '그날 관측된 만기만' 써서 적합한다. 최소 4개 만기 필요.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PROC = ROOT / "data" / "processed"

# 커브 구성 만기 (변수명, 연 단위 만기)
CURVE = [
    ("ktb1y_pct", 1.0),
    ("ktb2y_pct", 2.0),
    ("ktb3y_pct", 3.0),
    ("ktb5y_pct", 5.0),
    ("ktb10y_pct", 10.0),
    ("ktb20y_pct", 20.0),
    ("ktb30y_pct", 30.0),
]
MIN_TENORS = 4


def ns_loadings(tau: np.ndarray, lam: float) -> np.ndarray:
    """NS 설계행렬 [1, slope_loading, curvature_loading]."""
    x = lam * tau
    # x->0 극한에서 (1-exp(-x))/x -> 1. tau는 항상 >=1이므로 실제로는 안전하나 방어적으로 처리.
    with np.errstate(divide="ignore", invalid="ignore"):
        l2 = np.where(x < 1e-8, 1.0, (1.0 - np.exp(-x)) / x)
    l3 = l2 - np.exp(-x)
    return np.column_stack([np.ones_like(tau), l2, l3])


def fit_day(y: np.ndarray, tau: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """하루치 선형 최소제곱. (beta[3], SSE) 반환."""
    X = ns_loadings(tau, lam)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, float(resid @ resid)


def fit_all(panel: pd.DataFrame, lam: float) -> pd.DataFrame:
    """전 기간 NS 적합. beta1/2/3, RMSE(bp), 사용만기수를 반환."""
    cols = [c for c, _ in CURVE if c in panel.columns]
    taus = np.array([t for c, t in CURVE if c in panel.columns])
    Y = panel[cols].to_numpy(float)

    out = np.full((len(panel), 5), np.nan)
    for i in range(len(panel)):
        row = Y[i]
        ok = ~np.isnan(row)
        if ok.sum() < MIN_TENORS:
            continue
        beta, sse = fit_day(row[ok], taus[ok], lam)
        out[i, :3] = beta
        out[i, 3] = np.sqrt(sse / ok.sum()) * 100     # 적합오차 bp
        out[i, 4] = ok.sum()

    return pd.DataFrame(
        out,
        index=panel.index,
        columns=["beta1_level", "beta2_slope", "beta3_curv", "fit_rmse_bp", "n_tenors"],
    )


def choose_lambda(panel: pd.DataFrame, grid: np.ndarray | None = None) -> tuple[float, pd.DataFrame]:
    """전체 표본 평균 적합 RMSE를 최소화하는 lambda 선택."""
    if grid is None:
        grid = np.arange(0.20, 0.85, 0.01)
    rows = []
    for lam in grid:
        f = fit_all(panel, float(lam))
        rows.append({"lambda": float(lam), "mean_rmse_bp": f["fit_rmse_bp"].mean()})
    tab = pd.DataFrame(rows)
    best = float(tab.loc[tab["mean_rmse_bp"].idxmin(), "lambda"])
    return best, tab


def pca_factors(panel: pd.DataFrame, n: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    """검증용 PCA. 전 기간 관측되는 만기만 사용(2년·30년 제외)."""
    cols = [c for c, _ in CURVE if c in panel.columns and panel[c].isna().mean() < 0.01]
    X = panel[cols].dropna()
    Xc = X - X.mean()
    U, S, Vt = np.linalg.svd(Xc.to_numpy(), full_matrices=False)
    scores = U[:, :n] * S[:n]
    evr = (S**2 / (S**2).sum())[:n]
    return pd.DataFrame(scores, index=X.index, columns=[f"pc{i+1}" for i in range(n)]), evr


if __name__ == "__main__":
    panel = pd.read_parquet(PROC / "panel.parquet")

    print("[1] lambda 그리드서치")
    lam, tab = choose_lambda(panel)
    print(tab.iloc[(tab["mean_rmse_bp"]).argsort()[:5]].to_string(index=False))
    print(f"\n선택된 lambda = {lam:.2f}")
    print(f"  -> 곡률 로딩 최대 만기 ~= {1.7933 / lam:.2f}년\n")

    print("[2] 전 기간 NS 적합")
    factors = fit_all(panel, lam)
    print(f"  적합일수 {factors['beta1_level'].notna().sum()} / {len(panel)}")
    print(f"  평균 적합오차 {factors['fit_rmse_bp'].mean():.2f}bp, "
          f"최대 {factors['fit_rmse_bp'].max():.2f}bp")
    print(factors[["beta1_level", "beta2_slope", "beta3_curv"]].describe().round(3).to_string())

    print("\n[3] PCA 교차검증 (기획서 3.3-2)")
    pcs, evr = pca_factors(panel)
    print(f"  설명력 PC1 {evr[0]:.1%}, PC2 {evr[1]:.1%}, PC3 {evr[2]:.1%} "
          f"(누적 {evr.sum():.1%})")
    j = factors.join(pcs, how="inner").dropna()
    print(f"\n  {'':<14}{'PC1':>8}{'PC2':>8}{'PC3':>8}")
    for b in ["beta1_level", "beta2_slope", "beta3_curv"]:
        c = [j[b].corr(j[f"pc{i}"]) for i in (1, 2, 3)]
        print(f"  {b:<14}" + "".join(f"{v:>8.3f}" for v in c))

    pairs = [("beta1_level", "pc1"), ("beta2_slope", "pc2"), ("beta3_curv", "pc3")]
    print()
    for b, p in pairs:
        r = abs(j[b].corr(j[p]))
        print(f"  [{'PASS' if r > 0.9 else 'CHECK'}] |corr({b},{p})| = {r:.3f} (기준 >0.9)")

    factors.to_parquet(PROC / "ns_factors.parquet")
    pd.Series({"lambda": lam}).to_json(PROC / "ns_config.json")
    print(f"\n저장: {PROC / 'ns_factors.parquet'}")
