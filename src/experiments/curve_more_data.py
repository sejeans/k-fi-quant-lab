"""표본 늘리기 실험: (a) 한국 2010~ (현행)  (b) 한국 2001~  (c) 한국 2001~ + 미국 1962~ 합동 학습.

공정 비교를 위해 세 경우 모두 NS를 같은 4개 만기(1·3·5·10년)와 같은 lambda로 적합한다.
(만기 구성이 기간마다 다르면 요인 값에 단절이 생긴다.)
평가 구간·purge 규칙은 curve_ideas와 같다. 미국 행은 그 정답(20영업일 뒤)이 i시점 이전에
확정된 것만 학습에 쓴다.
"""
import sys as _s
if hasattr(_s.stdout, "reconfigure"): _s.stdout.reconfigure(encoding="utf-8", errors="replace")
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import src.features.panel as P
from src.features.nelson_siegel import ns_loadings
from src.models.curve_forecast import FACTORS, make_features, enet_model

PROC = ROOT / "data" / "processed"; EXT = ROOT / "data" / "raw_ext"; H = 20; REFIT = 20
lam = float(pd.read_json(PROC / "ns_config.json", typ="series")["lambda"])
TAUS = np.array([1.0, 3.0, 5.0, 10.0]); L = ns_loadings(TAUS, lam)
END = pd.read_parquet(PROC / "ns_factors.parquet").index.max()      # 현행 표본과 같은 끝점
SEAL = END - pd.DateOffset(years=2)
PERIODS = {"dev19": (pd.Timestamp("2019-01-01"), SEAL - pd.Timedelta(days=1)), "sealed": (SEAL, END),
           "dev13": (pd.Timestamp("2013-01-01"), SEAL - pd.Timedelta(days=1))}

def ns4(Y):
    B = np.linalg.lstsq(L, Y.T, rcond=None)[0].T          # 4만기 전부 있는 날만
    return B

def factors_from(yields: pd.DataFrame) -> pd.DataFrame:
    y = yields.dropna()
    return pd.DataFrame(ns4(y.to_numpy(float)), index=y.index, columns=FACTORS)

def own(f): return [f, f"{f}_d5", f"{f}_d20", f"{f}_z60"]
FONLY = sum((own(f) for f in FACTORS), [])

# ---------- 데이터 ----------
P.RAW = EXT
kr_panel = P.build_panel().loc[:END]
kr_fac = factors_from(kr_panel[["ktb1y_pct", "ktb3y_pct", "ktb5y_pct", "ktb10y_pct"]])
kr_X = make_features(kr_panel.loc[kr_fac.index], kr_fac).dropna()
us = pd.read_parquet(EXT / "us_treasury.parquet").dropna()
us_fac = factors_from(us)
us_X = make_features(pd.DataFrame(index=us_fac.index, data={c: 0.0 for c in
                     ["cd_msb_bp", "cp_cd_bp", "ktb1y_base_gap_bp", "ktb3y_vol20_bp"]}), us_fac)[FONLY].dropna()
us_end = pd.Series(us_fac.index, index=us_fac.index).shift(-H).reindex(us_X.index)   # 미국 정답 확정일

def target(fac, X): return pd.concat({f: fac[f].shift(-H) - fac[f] for f in FACTORS}, axis=1).reindex(X.index)
kr_Y, us_Y = target(kr_fac, kr_X), target(us_fac, us_X)

def run(kr_start, cols, pool_us, eval_from):
    Xk = kr_X.loc[kr_start:, cols]; Yk = kr_Y.loc[Xk.index]
    idx = Xk.index; pos = np.where(idx >= eval_from)[0]
    pred = {f: np.full(len(pos), np.nan) for f in FACTORS}
    models = {}
    for k, i in enumerate(pos):
        if k % REFIT == 0:
            tr = Xk.iloc[:i - H + 1]; ytr = Yk.iloc[:i - H + 1]
            m = ytr.notna().all(axis=1); tr, ytr = tr[m], ytr[m]
            if pool_us:
                ok = us_end.notna() & (us_end <= idx[i]) & us_Y.notna().all(axis=1)
                tr = pd.concat([tr, us_X.loc[ok, cols]]); ytr = pd.concat([ytr, us_Y.loc[ok]])
            sc = StandardScaler().fit(tr)
            Z = sc.transform(tr)
            for f in FACTORS:
                en = enet_model().fit(Z, ytr[f])
                oc = [cols.index(c) for c in own(f)]
                ar = LinearRegression().fit(Z[:, oc], ytr[f])
                models[f] = (en, ar, oc)
            scaler = sc
        z = scaler.transform(Xk.iloc[i:i + 1])
        for f in FACTORS:
            en, ar, oc = models[f]
            pred[f][k] = (en.predict(z)[0] + ar.predict(z[:, oc])[0]) / 2   # ENet+AR 평균
    return {f: pd.DataFrame({"actual": Yk[f].iloc[pos].to_numpy(), "pred": pred[f]}, index=idx[pos]).dropna()
            for f in FACTORS}

def evaluate(res, lo, hi, shr):
    idx = res[FACTORS[0]].loc[lo:hi].index
    for f in FACTORS[1:]: idx = idx.intersection(res[f].index)
    Pm = np.column_stack([res[f].loc[idx, "pred"] * s for f, s in zip(FACTORS, shr)])
    A = np.column_stack([res[f].loc[idx, "actual"] for f in FACTORS])
    dp, da = Pm @ L.T * 100, A @ L.T * 100
    row = {}
    for t in (1.0, 3.0, 10.0):
        k = list(TAUS).index(t); rw = np.sqrt((da[:, k] ** 2).mean()); m = np.sqrt(((da[:, k] - dp[:, k]) ** 2).mean())
        row[f"{t:g}y"] = (rw - m) / rw * 100
        row[f"{t:g}y_hit"] = (np.sign(dp[:, k]) == np.sign(da[:, k]))[dp[:, k] != 0].mean() * 100 if (dp[:, k] != 0).any() else np.nan
    row["n"] = len(idx)
    return row

if __name__ == "__main__":
    print(f"한국 4만기 요인: {kr_fac.index.min():%Y-%m-%d}~ ({len(kr_X)}행), 미국: {us_fac.index.min():%Y-%m-%d}~ ({len(us_X)}행)")
    allc = list(kr_X.columns)
    CASES = {
        "a 한국2010~ 전체특징":      ("2010-01-01", allc, False),
        "b 한국2001~ 전체특징":      ("2001-01-01", allc, False),
        "a' 한국2010~ 요인특징":     ("2010-01-01", FONLY, False),
        "b' 한국2001~ 요인특징":     ("2001-01-01", FONLY, False),
        "c 한국2001~+미국 요인특징": ("2001-01-01", FONLY, True),
    }
    rows = []
    for name, (st, cols, pool) in CASES.items():
        ev0 = PERIODS["dev13"][0] if name[0] != "a" else PERIODS["dev19"][0]
        res = run(st, cols, pool, ev0)
        for tag, shr in {"평균": (1, 1, 1), "기울기만": (0, 1, 0)}.items():
            for per, (lo, hi) in PERIODS.items():
                if per == "dev13" and name[0] == "a": continue      # 2010~ 학습은 2013 평가 시 표본이 너무 짧음(최소 750일 미달)
                rows.append({"case": name, "post": tag, "period": per, **evaluate(res, lo, hi, shr)})
        print(name, "done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(PROC / "curve_more_data.csv", index=False)
    pd.set_option("display.width", 220)
    for per in PERIODS:
        print(f"\n== {per} (RW 대비 RMSE 개선 %, 방향 적중 %) ==")
        print(df[df.period == per].drop(columns="period").round(1).to_string(index=False))
