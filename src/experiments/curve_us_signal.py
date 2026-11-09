"""(1) 미국 금리 전일 변화를 특징으로 추가  (2) 1년물 성과의 유의성 검정.

정보 시점: 한국 t일 장 시작 전에 확정된 미국 종가는 미국 t-1일(달력) 이전 값이다.
따라서 한국 t일 행에는 '달력상 t-1일 이전의 마지막 미국 관측'을 붙인다(asof).

유의성: 20일 예측은 매일 겹치므로 일별 적중률을 그대로 이항검정하면 과장된다.
- 겹치지 않는 표본: 20일 간격으로 뽑은 표본으로 이항검정. 시작 오프셋 20개의 중앙값을 보고.
- Diebold-Mariano: 손실차(RW 제곱오차 - 모델 제곱오차)의 평균을 Newey-West(lag=H-1) 분산으로 검정.
"""
import sys as _s
if hasattr(_s.stdout, "reconfigure"): _s.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from scipy import stats
import src.experiments.curve_more_data as M
from src.experiments.curve_more_data import FACTORS, L, TAUS, PERIODS, H, PROC

us = pd.read_parquet(M.EXT / "us_treasury.parquet").dropna()
usf = pd.DataFrame(index=us.index)
for c, nm in [("DGS1", "us1"), ("DGS3", "us3"), ("DGS10", "us10")]:
    for d in (1, 5, 20):
        usf[f"{nm}_d{d}"] = us[c].diff(d)
usf["us_slope_d20"] = (us["DGS10"] - us["DGS1"]).diff(20)
kr = M.kr_X.copy()
key = pd.DataFrame({"t": kr.index, "cut": kr.index - pd.Timedelta(days=1)})
aligned = pd.merge_asof(key.sort_values("cut"), usf.reset_index().rename(columns={"observation_date": "cut"}).sort_values("cut"),
                        on="cut", direction="backward").set_index("t").drop(columns="cut")
assert (aligned.index == kr.index).all()
US = list(usf.columns)
M.kr_X = kr.join(aligned).dropna()
M.kr_Y = M.kr_Y.reindex(M.kr_X.index)
BASE = [c for c in kr.columns]

def yields(res, shr, lo, hi):
    idx = res[FACTORS[0]].loc[lo:hi].index
    for f in FACTORS[1:]: idx = idx.intersection(res[f].index)
    Pm = np.column_stack([res[f].loc[idx, "pred"] * s for f, s in zip(FACTORS, shr)])
    A = np.column_stack([res[f].loc[idx, "actual"] for f in FACTORS])
    return pd.DataFrame(Pm @ L.T * 100, index=idx, columns=TAUS), pd.DataFrame(A @ L.T * 100, index=idx, columns=TAUS)

def tests(p, a):
    p, a = p.to_numpy(), a.to_numpy()
    d = a**2 - (a - p)**2                          # +면 모델이 RW보다 오차가 작음
    n = len(d); dc = d - d.mean()
    v = dc @ dc / n
    for l in range(1, H):
        v += 2 * (1 - l / H) * (dc[l:] @ dc[:-l]) / n
    dm_p = 2 * (1 - stats.norm.cdf(abs(d.mean() / np.sqrt(v / n))))
    hits, ps = [], []
    for o in range(H):
        pp, aa = p[o::H], a[o::H]; m = pp != 0
        k = int((np.sign(pp[m]) == np.sign(aa[m])).sum()); hits.append(k / m.sum() * 100)
        ps.append(stats.binomtest(k, m.sum(), .5, alternative="greater").pvalue)
    return {"DM_p": dm_p, "독립n": n // H, "독립적중%": np.median(hits), "이항p(중앙)": np.median(ps)}

if __name__ == "__main__":
    CASES = {"b 한국2001~ 기본특징": ("2001-01-01", BASE), "b+US 한국2001~ +미국전일": ("2001-01-01", BASE + US),
             "a 한국2010~ 기본특징": ("2010-01-01", BASE), "a+US 한국2010~ +미국전일": ("2010-01-01", BASE + US)}
    POST = {"평균": (1, 1, 1), "기울기만": (0, 1, 0), "수준+기울기": (1, 1, 0)}
    rows, sig = [], []
    for name, (st, cols) in CASES.items():
        ev0 = PERIODS["dev13"][0] if name[0] == "b" else PERIODS["dev19"][0]
        res = M.run(st, cols, False, ev0)
        for tag, shr in POST.items():
            for per, (lo, hi) in PERIODS.items():
                if per == "dev13" and name[0] == "a": continue
                p, a = yields(res, shr, lo, hi)
                row = {"case": name, "post": tag, "period": per}
                for t in (1.0, 3.0, 10.0):
                    rw = np.sqrt((a[t]**2).mean()); m = np.sqrt(((a[t] - p[t])**2).mean())
                    row[f"{t:g}y"] = (rw - m) / rw * 100
                    if tag == "기울기만" or name.endswith("미국전일"):
                        sig.append({"case": name, "post": tag, "period": per, "만기": f"{t:g}y",
                                    "개선%": row[f"{t:g}y"], **tests(p[t], a[t])})
                rows.append(row)
        print(name, "done", flush=True)
    df, sg = pd.DataFrame(rows), pd.DataFrame(sig)
    df.to_csv(PROC / "curve_us_signal.csv", index=False); sg.to_csv(PROC / "curve_significance.csv", index=False)
    pd.set_option("display.width", 220)
    for per in PERIODS:
        print(f"\n== {per} RW 대비 RMSE 개선 % =="); print(df[df.period == per].drop(columns="period").round(1).to_string(index=False))
    print("\n== 유의성 (DM_p: 손실차 검정, 이항p: 겹치지 않는 표본 방향 적중) ==")
    print(sg.round(4).to_string(index=False))
