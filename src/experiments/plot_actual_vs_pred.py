"""실제 vs 예측 20영업일 변화 시계열 (2019~, 봉인 구간 음영).

예측은 curve_ideas와 같은 조건(walk-forward, purge)으로 2019년부터 다시 계산해
data/processed/curve_pred_2019.parquet에 저장한다(있으면 재사용).
"""
import sys as _s
if hasattr(_s.stdout, "reconfigure"): _s.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.experiments.curve_ideas import *   # noqa
import src.reporting.figures                # 한글 폰트 설정 재사용
from sklearn.linear_model import LinearRegression

FIG = ROOT / "reports" / "figures"
CACHE = PROC / "curve_pred_2019.parquet"
INK, MUTED = "#0b0b0b", "#52514e"
C_FINAL, C_ENET = "#2a78d6", "#eb6834"
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})

if CACHE.exists():
    pr = pd.read_parquet(CACHE)
else:
    cols = {}
    for f in FACTORS:
        y = fac[f].shift(-H) - fac[f]
        en = wf.walk_forward(X0, y, DEV0, H, enet_model)
        ar = wf.walk_forward(X0[own(f)], y, DEV0, H, LinearRegression)
        cols[(f, "actual")], cols[(f, "enet")], cols[(f, "ar")] = en["actual"], en["pred"], ar["pred"].reindex(en.index)
    pr = pd.DataFrame(cols).dropna()
    pr.columns = [f"{a}|{b}" for a, b in pr.columns]
    pr.to_parquet(CACHE)

A = np.column_stack([pr[f"{f}|actual"] for f in FACTORS])
EN = np.column_stack([pr[f"{f}|enet"] for f in FACTORS])
AV = np.column_stack([(pr[f"{f}|enet"] + pr[f"{f}|ar"]) / 2 for f in FACTORS])
FINAL = AV * np.array([0, 1, 0])
to_y = lambda B: pd.DataFrame(B @ L.T * 100, index=pr.index, columns=taus)
ya, yf, ye = to_y(A), to_y(FINAL), to_y(EN)

def imp(a, p, lo, hi):
    a, p = a.loc[lo:hi], p.loc[lo:hi]
    rw = np.sqrt((a**2).mean()); return (rw - np.sqrt(((a - p)**2).mean())) / rw * 100

def shade(ax):
    ax.axvspan(SEAL, pr.index.max(), color="#e9e8e4", zorder=0, lw=0)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)

dev_hi = SEAL - pd.Timedelta(days=1)

# ---- 그림 1: 만기별 금리 ----
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
for ax, t in zip(axes, (1.0, 3.0, 10.0)):
    shade(ax)
    ax.plot(ya.index, ya[t], color=INK, lw=1.0, label="실제")
    ax.plot(ye.index, ye[t], color=C_ENET, lw=1.0, alpha=0.8, label="처음 모델 (ENet, 세 요인 모두)")
    ax.plot(yf.index, yf[t], color=C_FINAL, lw=1.6, label="최종 모델 (기울기 요인만)")
    ax.set_title(f"국고 {t:g}년 20영업일 금리 변화(bp)   RW 대비 RMSE 개선: 최종 dev {imp(ya[t], yf[t], DEV0, dev_hi):+.1f}% / 봉인 "
                 f"{imp(ya[t], yf[t], SEAL, None):+.1f}%,  처음 dev {imp(ya[t], ye[t], DEV0, dev_hi):+.1f}% / 봉인 {imp(ya[t], ye[t], SEAL, None):+.1f}%",
                 loc="left", fontsize=9.5, color=INK)
    ax.set_ylabel("bp", color=MUTED)
axes[0].text(SEAL, axes[0].get_ylim()[1], " 봉인 구간", va="top", fontsize=9, color=MUTED)
axes[0].legend(loc="upper left", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0, 1.32))
fig.suptitle("실제 vs 예측: 만기별 20영업일 금리 변화 (2019~, 날짜 = 예측 시점)", x=0.01, ha="left", fontsize=12, y=1.0)
fig.tight_layout(); fig.savefig(FIG / "07_actual_vs_pred_yields.png", bbox_inches="tight", dpi=130); plt.close(fig)

# ---- 그림 2: 요인별 ----
names = {"beta1_level": "수준 요인", "beta2_slope": "기울기 요인 (최종 모델에서 사용)", "beta3_curv": "곡률 요인"}
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
for j, (ax, f) in enumerate(zip(axes, FACTORS)):
    shade(ax)
    a, e, v = pd.Series(A[:, j] * 100, pr.index), pd.Series(EN[:, j] * 100, pr.index), pd.Series(AV[:, j] * 100, pr.index)
    ax.plot(a.index, a, color=INK, lw=1.0, label="실제")
    ax.plot(e.index, e, color=C_ENET, lw=1.0, alpha=0.8, label="ENet")
    ax.plot(v.index, v, color=C_FINAL, lw=1.6, label="ENet+AR 평균")
    ax.set_title(f"{names[f]} 20영업일 변화(bp)   RW 대비 RMSE 개선: ENet dev {imp(a, e, DEV0, dev_hi):+.1f}% / 봉인 {imp(a, e, SEAL, None):+.1f}%,  "
                 f"ENet+AR dev {imp(a, v, DEV0, dev_hi):+.1f}% / 봉인 {imp(a, v, SEAL, None):+.1f}%", loc="left", fontsize=9.5, color=INK)
    ax.set_ylabel("bp", color=MUTED)
axes[0].text(SEAL, axes[0].get_ylim()[1], " 봉인 구간", va="top", fontsize=9, color=MUTED)
axes[0].legend(loc="upper left", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0, 1.32))
fig.suptitle("실제 vs 예측: NS 요인별 20영업일 변화 (2019~, 날짜 = 예측 시점)", x=0.01, ha="left", fontsize=12, y=1.0)
fig.tight_layout(); fig.savefig(FIG / "08_actual_vs_pred_factors.png", bbox_inches="tight", dpi=130); plt.close(fig)
print("저장:", FIG / "07_actual_vs_pred_yields.png", FIG / "08_actual_vs_pred_factors.png")
