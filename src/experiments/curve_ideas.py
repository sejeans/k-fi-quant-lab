"""금리커브 예측 개선 아이디어 일괄 테스트 (h=20영업일, NS 요인 -> 만기별 금리 변화).

선택 편향을 막기 위해 평가 구간을 둘로 나눈다.
- dev   : 2019-01 ~ 봉인 직전. 아이디어 비교는 여기서 한다.
- sealed: 마지막 2년. dev에서 이긴 아이디어만 여기서 확인한다.
지표는 만기별 RMSE의 RW 대비 개선율(%). 겹치는 20일 표본이라 1%p 안팎 차이는 노이즈로 본다.
"""
from __future__ import annotations
import sys as _s
if hasattr(_s.stdout, "reconfigure"):
    _s.stdout.reconfigure(encoding="utf-8", errors="replace")
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import ElasticNetCV, RidgeCV, HuberRegressor, LinearRegression, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.features.nelson_siegel import CURVE, ns_loadings
from src.models.curve_forecast import FACTORS, make_features, enet_model
from src.models import walkforward as wf

PROC = ROOT / "data" / "processed"; H = 20
panel = pd.read_parquet(PROC / "panel.parquet"); fac = pd.read_parquet(PROC / "ns_factors.parquet")
lam = float(pd.read_json(PROC / "ns_config.json", typ="series")["lambda"])
taus = np.array([t for _, t in CURVE]); L = ns_loadings(taus, lam)
SEAL = fac.index.max() - pd.DateOffset(years=2); DEV0 = pd.Timestamp("2019-01-01")

X0 = make_features(panel, fac)

def extra(X):
    X = X.copy()
    for f in FACTORS:
        X[f"{f}_d60"] = fac[f].diff(60); X[f"{f}_d120"] = fac[f].diff(120)
        X[f"{f}_pct"] = fac[f].rolling(500).rank(pct=True)
    X["gap_d20"] = panel["ktb1y_base_gap_bp"].diff(20)
    X["gap_d60"] = panel["ktb1y_base_gap_bp"].diff(60)
    X["cd_msb_d20"] = panel["cd_msb_bp"].diff(20)
    X["slope1s10s"] = panel["slope_1s10s_bp"]; X["bfly"] = panel["bfly_3s5s10s_bp"]
    X["call_gap"] = panel["call_base_gap_bp"]
    X["base_chg60"] = panel["base_rate_pct"].diff(60) * 100
    return X
X1 = extra(X0)
own = lambda f: [f, f"{f}_d5", f"{f}_d20", f"{f}_z60"]

class Shrink:                         # 예측 크기를 a배로 축소 (RW 방향으로 수축)
    def __init__(self, base, a): self.m, self.a = base(), a
    def fit(self, X, y): self.m.fit(X, y); return self
    def predict(self, X): return self.a * self.m.predict(X)
class Avg:                            # 모델 평균
    def __init__(self, makers): self.ms = [m() for m in makers]
    def fit(self, X, y): [m.fit(X, y) for m in self.ms]; return self
    def predict(self, X): return np.mean([m.predict(X) for m in self.ms], axis=0)
class Roll:                           # 최근 N행만 학습 (walk_forward가 expanding이라 래퍼에서 자름)
    def __init__(self, base, n): self.m, self.n = base(), n
    def fit(self, X, y): self.m.fit(X[-self.n:], y[-self.n:]); return self
    def predict(self, X): return self.m.predict(X)

ridge = lambda: RidgeCV(alphas=np.logspace(0, 4, 12))
huber = lambda: HuberRegressor(alpha=10.0, max_iter=300)
pls = lambda: PLSRegression(n_components=2)
class PLS1(PLSRegression):
    def predict(self, X): return super().predict(X).ravel()
gbr = lambda: GradientBoostingRegressor(n_estimators=80, max_depth=2, learning_rate=0.03, subsample=0.7, random_state=42)
ols_own = LinearRegression

# (이름, 특징세트 함수(f)->DataFrame, 모델 생성자)
IDEAS = {
    "0 기준: ENet 기본특징":        (lambda f: X0, enet_model),
    "1 AR(자기 4특징)":             (lambda f: X0[own(f)], LinearRegression),
    "2 ENet 예측 x0.5 수축":        (lambda f: X0, lambda: Shrink(enet_model, .5)),
    "3 Ridge 강정규화":             (lambda f: X0, ridge),
    "4 Huber(이상치 강건)":         (lambda f: X0, huber),
    "5 PLS 2성분":                  (lambda f: X0, lambda: PLS1(n_components=2)),
    "6 ENet 확장특징":              (lambda f: X1, enet_model),
    "7 Ridge 확장특징":             (lambda f: X1, ridge),
    "8 ENet 최근3년 롤링":          (lambda f: X0, lambda: Roll(enet_model, 750)),
    "9 ENet 최근5년 롤링":          (lambda f: X0, lambda: Roll(enet_model, 1250)),
    "10 ENet+AR 평균":              (lambda f: X0, None),   # 아래서 처리
    "11 GBM 얕은 트리":             (lambda f: X0, gbr),
    "12 Ridge 확장+0.5수축":        (lambda f: X1, lambda: Shrink(ridge, .5)),
    "13 Huber 확장특징":            (lambda f: X1, huber),
}

def run(name, featf, maker):
    out = {}
    for f in FACTORS:
        y = fac[f].shift(-H) - fac[f]
        if name.startswith("10"):
            a = wf.walk_forward(X0, y, DEV0, H, enet_model)
            b = wf.walk_forward(X0[own(f)], y, DEV0, H, LinearRegression)
            r = a.copy(); r["pred"] = (a["pred"] + b["pred"].reindex(a.index)) / 2
        else:
            r = wf.walk_forward(featf(f), y, DEV0, H, maker)
        out[f] = r[r["actual"].notna()]
    return out

def evaluate(res, lo, hi):
    idx = res[FACTORS[0]].loc[lo:hi].index
    for f in FACTORS[1:]: idx = idx.intersection(res[f].loc[lo:hi].index)
    P = np.column_stack([res[f].loc[idx, "pred"] for f in FACTORS]); A = np.column_stack([res[f].loc[idx, "actual"] for f in FACTORS])
    dp, da = P @ L.T * 100, A @ L.T * 100
    row = {}
    for t in (1.0, 3.0, 10.0):
        k = list(taus).index(t); rw = np.sqrt((da[:, k] ** 2).mean()); m = np.sqrt(((da[:, k] - dp[:, k]) ** 2).mean())
        row[f"{t:g}y"] = (rw - m) / rw * 100
    for j, f in enumerate(FACTORS):
        rw = np.sqrt((A[:, j] ** 2).mean()); m = np.sqrt(((A[:, j] - P[:, j]) ** 2).mean())
        row[f[:5] + "_" + f.split("_")[1][:3]] = (rw - m) / rw * 100
    row["n"] = len(idx)
    return row

if __name__ == "__main__":
    rows = []
    for name, (ff, mk) in IDEAS.items():
        res = run(name, ff, mk)
        for per, (lo, hi) in {"dev": (DEV0, SEAL - pd.Timedelta(days=1)), "sealed": (SEAL, fac.index.max())}.items():
            rows.append({"idea": name, "period": per, **evaluate(res, lo, hi)})
        print(name, "done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(PROC / "curve_ideas.csv", index=False)
    pd.set_option("display.width", 200)
    for per in ("dev", "sealed"):
        print(f"\n== {per} (RW 대비 RMSE 개선 %, +가 좋음) =="); print(df[df.period == per].drop(columns="period").round(2).to_string(index=False))
