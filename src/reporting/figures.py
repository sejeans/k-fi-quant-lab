"""리포트용 차트 일괄 생성. 모든 그림은 reports/figures/에 저장된다."""

from __future__ import annotations

import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):      # Windows cp949 콘솔에서 한글/기호 깨짐 방지
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"

# 한글 폰트: 윈도우 맑은 고딕
for cand in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def fig_curve_3d(panel: pd.DataFrame) -> None:
    """만기별 금리 시계열 + 최근 커브 형태."""
    tenors = [("ktb1y_pct", 1), ("ktb2y_pct", 2), ("ktb3y_pct", 3), ("ktb5y_pct", 5),
              ("ktb10y_pct", 10), ("ktb20y_pct", 20), ("ktb30y_pct", 30)]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    for c, t in tenors:
        ax[0].plot(panel.index, panel[c], lw=0.8, label=f"{t}년")
    ax[0].set_title("국고채 만기별 수익률")
    ax[0].set_ylabel("%")
    ax[0].legend(ncol=4, fontsize=8)

    for lbl, dt in [("현재", panel.index[-1]),
                    ("1년 전", panel.index[max(0, len(panel) - 250)]),
                    ("3년 전", panel.index[max(0, len(panel) - 750)])]:
        xs = [t for c, t in tenors if not np.isnan(panel.loc[dt, c])]
        ys = [panel.loc[dt, c] for c, t in tenors if not np.isnan(panel.loc[dt, c])]
        ax[1].plot(xs, ys, "o-", label=f"{lbl} ({dt:%Y-%m-%d})")
    ax[1].set_title("일드커브 형태 비교")
    ax[1].set_xlabel("만기(년)")
    ax[1].set_ylabel("%")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "01_curve.png", bbox_inches="tight")
    plt.close(fig)


def fig_ns_factors(panel: pd.DataFrame, f: pd.DataFrame) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(f.index, f["beta1_level"], lw=0.9, label="β1 Level")
    ax[0].plot(f.index, f["beta2_slope"], lw=0.9, label="β2 Slope")
    ax[0].plot(f.index, f["beta3_curv"], lw=0.9, label="β3 Curvature")
    ax[0].set_title("Nelson-Siegel 요인 (λ 고정 선형적합)")
    ax[0].legend(ncol=3, fontsize=9)
    ax[1].plot(f.index, f["fit_rmse_bp"], lw=0.7, color="crimson")
    ax[1].set_title(f"일별 커브 적합오차 (평균 {f['fit_rmse_bp'].mean():.2f}bp)")
    ax[1].set_ylabel("bp")
    fig.tight_layout()
    fig.savefig(FIG / "02_ns_factors.png", bbox_inches="tight")
    plt.close(fig)


def fig_rmse_heatmap() -> None:
    """만기 x 예측시계 RMSE 개선율 heatmap."""
    frames = []
    for h in (5, 20):
        f = PROC / f"curve_rmse_h{h}.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
    if not frames:
        return
    d = pd.concat(frames)
    piv = d.pivot(index="tenor", columns="h", values="improve_pct")
    order = ["ktb1y_pct", "ktb2y_pct", "ktb3y_pct", "ktb5y_pct",
             "ktb10y_pct", "ktb20y_pct", "ktb30y_pct"]
    piv = piv.reindex([o for o in order if o in piv.index])

    fig, ax = plt.subplots(figsize=(5.5, 4))
    v = np.abs(piv.to_numpy()).max()
    im = ax.imshow(piv, cmap="RdYlGn", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [f"{c}일" for c in piv.columns])
    ax.set_yticks(range(len(piv)), [i.replace("ktb", "").replace("_pct", "") for i in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.iloc[i,j]:+.1f}%", ha="center", va="center", fontsize=9)
    ax.set_title("Random Walk 대비 RMSE 개선율\n(양수=모델 우위)")
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    fig.savefig(FIG / "03_rmse_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def fig_backtest() -> None:
    f = PROC / "backtest_bfly_3s5s10s_bp.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for cx in sorted(d["cost_x"].unique()):
        w = d[d["cost_x"] == cx].sort_values("signal_q")
        ax[0].plot(w["signal_q"], w["ann_ret_%"], "o-", label=f"비용 {cx:.0f}x")
        ax[1].plot(w["signal_q"], w["sharpe"], "o-", label=f"비용 {cx:.0f}x")
    for a, t in zip(ax, ["연환산 수익률(%)", "Sharpe"]):
        a.axhline(0, color="k", lw=0.8, ls="--")
        a.set_xlabel("신호 임계 분위수")
        a.set_title(f"3-5-10 버터플라이 — {t}")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "04_backtest.png", bbox_inches="tight")
    plt.close(fig)


def fig_policy() -> None:
    f = PROC / "policy_predictions.parquet"
    if not f.exists():
        return
    r = pd.read_parquet(f)
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(r["meeting_date"], r["p_인상"], "o-", ms=3, color="crimson", label="인상")
    ax[0].plot(r["meeting_date"], r["p_인하"], "o-", ms=3, color="royalblue", label="인하")
    ax[0].plot(r["meeting_date"], r["p_동결"], "-", lw=0.8, color="gray", label="동결")
    ax[0].set_title("금통위 결정 확률 예측 (walk-forward)")
    ax[0].legend(ncol=3, fontsize=9)

    mark = {"인상": ("^", "crimson"), "인하": ("v", "royalblue"), "동결": (".", "lightgray")}
    for lbl, (m, c) in mark.items():
        w = r[r["actual"] == lbl]
        ax[1].scatter(w["meeting_date"], [lbl] * len(w), marker=m, color=c, s=40)
    ax[1].set_title("실제 결정")
    fig.tight_layout()
    fig.savefig(FIG / "05_policy.png", bbox_inches="tight")
    plt.close(fig)


def fig_credit(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(panel.index, panel["cs_aa3y_bp"], lw=0.8, color="darkgreen")
    ax[0].set_title("AA- 3년 크레딧 스프레드")
    ax[0].set_ylabel("bp")
    ax[1].plot(panel.index, panel["cs_bbb3y_bp"], lw=0.8, color="darkred")
    ax[1].set_title("BBB- 3년 크레딧 스프레드")
    ax[1].set_ylabel("bp")
    for a in ax:
        for s, e, lb in [("2020-02-01", "2020-06-30", "코로나"),
                         ("2022-09-01", "2023-01-31", "레고랜드")]:
            a.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.15, color="orange")
    fig.tight_layout()
    fig.savefig(FIG / "06_credit.png", bbox_inches="tight")
    plt.close(fig)


def fig_policy_timeline(panel: pd.DataFrame) -> None:
    """금통위: 시장금리 기반 모델의 예측 확률 vs 실제 결정 (시계열)."""
    f = PROC / "policy_predictions.parquet"
    if not f.exists():
        return
    r = pd.read_parquet(f).sort_values("meeting_date")
    INK, MUTED, BASE = "#0b0b0b", "#52514e", "#9a9993"
    UP, DN = "#eb6834", "#2a78d6"
    t0 = r["meeting_date"].min() - pd.Timedelta(days=60)
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.4, 1]})
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)

    # (1) 기준금리와 국고 1년: 모델 입력인 시장 기대
    w = panel.loc[t0:]
    ax[0].plot(w.index, w["ktb1y_pct"], color=MUTED, lw=1.0, label="국고 1년 금리 (시장 기대)")
    ax[0].step(w.index, w["base_rate_pct"], where="post", color=INK, lw=1.8, label="기준금리")
    ax[0].set_ylabel("%", color=MUTED)
    ax[0].set_title("국고 1년 금리가 기준금리보다 먼저 움직인다 (모델의 핵심 입력 = 두 금리의 차이)",
                    loc="left", fontsize=10.5)
    ax[0].legend(loc="upper left", frameon=False, ncol=1, fontsize=9, bbox_to_anchor=(1.005, 1))

    # (2) 예측 확률: 위 = 인상, 아래 = 인하. 실제 결정 표시
    x, wd = r["meeting_date"], 25
    ax[1].bar(x, r["p_인상"], width=wd, color=UP, label="예측 P(인상)")
    ax[1].bar(x, -r["p_인하"], width=wd, color=DN, label="예측 P(인하)")
    ax[1].step(x, r["base_인상"], where="mid", color=BASE, lw=1, ls="--", label="기준선 (과거 빈도)")
    ax[1].step(x, -r["base_인하"], where="mid", color=BASE, lw=1, ls="--")
    up, dn = r[r["actual"] == "인상"], r[r["actual"] == "인하"]
    ax[1].scatter(up["meeting_date"], [1.12] * len(up), marker="v", s=55, color=UP,
                  edgecolor=INK, lw=0.6, zorder=3, label="실제 인상")
    ax[1].scatter(dn["meeting_date"], [-1.12] * len(dn), marker="^", s=55, color=DN,
                  edgecolor=INK, lw=0.6, zorder=3, label="실제 인하")
    ax[1].axhline(0, color=MUTED, lw=0.8)
    ax[1].set_ylim(-1.25, 1.25)
    ax[1].set_yticks([-1, -0.5, 0, 0.5, 1], ["100%", "50%", "0", "50%", "100%"])
    ax[1].set_ylabel("인하 확률  |  인상 확률", color=MUTED)
    ax[1].set_title("회의별 예측 확률과 실제 결정 (walk-forward, 각 회의 전 정보만 사용)",
                    loc="left", fontsize=10.5)
    ax[1].legend(loc="upper left", frameon=False, ncol=1, fontsize=9, bbox_to_anchor=(1.005, 1))
    for d, txt in [("2021-08-26", "첫 인상\nP=17%"), ("2024-10-11", "첫 인하\nP=37%")]:
        row = r[r["meeting_date"] == pd.Timestamp(d)]
        if not row.empty:
            y = 0.55 if "인상" in txt else -0.75
            ax[1].annotate(txt, (pd.Timestamp(d), 0.2 if y > 0 else -0.4), xytext=(pd.Timestamp(d), y),
                           ha="center", fontsize=8.5, color=INK,
                           arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    # (3) 실제 결정에 부여한 확률: 모델 vs 기준선
    pm = r.apply(lambda q: q[f"p_{q['actual']}"], axis=1)
    pb = r.apply(lambda q: q[f"base_{q['actual']}"], axis=1)
    ll_m, ll_b = -np.log(pm.clip(1e-15)).mean(), -np.log(pb.clip(1e-15)).mean()
    ax[2].vlines(x, pb, pm, color=np.where(pm >= pb, "#1baf7a", "#e34948"), lw=1.6)
    ax[2].scatter(x, pb, s=14, color=BASE, zorder=3, label="기준선")
    ax[2].scatter(x, pm, s=18, color=INK, zorder=3, label="모델")
    ax[2].set_ylim(-0.03, 1.05)
    ax[2].set_ylabel("실제 결정의 예측 확률", color=MUTED)
    win = (pm > pb).mean() * 100
    ax[2].set_title(f"실제 결정에 부여한 확률 (초록 = 모델이 기준선보다 높음, {win:.0f}% 회의)   "
                    f"Log Loss 모델 {ll_m:.3f} vs 기준선 {ll_b:.3f}", loc="left", fontsize=10.5)
    ax[2].legend(loc="upper left", frameon=False, ncol=1, fontsize=9, bbox_to_anchor=(1.005, 1))

    fig.suptitle(f"금통위 결정 예측: 시장금리만 사용한 모델 vs 실제 ({len(r)}회, "
                 f"{r['meeting_date'].min():%Y.%m}~{r['meeting_date'].max():%Y.%m})",
                 x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "09_policy_actual_vs_pred.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PROC / "panel.parquet")
    factors = pd.read_parquet(PROC / "ns_factors.parquet")

    fig_curve_3d(panel)
    fig_ns_factors(panel, factors)
    fig_rmse_heatmap()
    fig_backtest()
    fig_policy()
    fig_policy_timeline(panel)
    fig_credit(panel)

    made = sorted(FIG.glob("*.png"))
    print(f"차트 {len(made)}개 생성:")
    for m in made:
        print(f"  {m.name}  ({m.stat().st_size/1024:.0f}KB)")
