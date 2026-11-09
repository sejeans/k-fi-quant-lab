"""2차: 1차에서 살아남은 'ENet+AR 평균'을 만기별로 수축/혼합하는 후처리 아이디어."""
import sys as _s
if hasattr(_s.stdout, "reconfigure"): _s.stdout.reconfigure(encoding="utf-8", errors="replace")
from src.experiments.curve_ideas import *   # noqa

key = "10 ENet+AR 평균"; res = run(key, *IDEAS[key])
def combo(shr):                                   # shr: 요인별 수축계수
    return {f: res[f].assign(pred=res[f]["pred"] * shr[i]) for i, f in enumerate(FACTORS)}
V = {"ENet+AR 평균(수축 없음)": (1, 1, 1),
     "수준 0 / 기울기·곡률 유지": (0, 1, 1),
     "수준 0 / 기울기 유지 / 곡률 0": (0, 1, 0),
     "전체 x0.7": (.7, .7, .7),
     "수준 x0.5 / 나머지 유지": (.5, 1, 1),
     "수준만 유지(단기물 전용)": (1, 0, 0),
     "전부 0 (=RW)": (0, 0, 0)}
rows = []
for n, s in V.items():
    for per, (lo, hi) in {"dev": (DEV0, SEAL - pd.Timedelta(days=1)), "sealed": (SEAL, fac.index.max())}.items():
        rows.append({"idea": n, "period": per, **evaluate(combo(s), lo, hi)})
df = pd.DataFrame(rows); df.to_csv(PROC / "curve_ideas2.csv", index=False)
pd.set_option("display.width", 200)
for per in ("dev", "sealed"):
    print(f"\n== {per} =="); print(df[df.period == per].drop(columns="period").round(2).to_string(index=False))
