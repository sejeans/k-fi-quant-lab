"""K-FI Quant Lab 통합 대시보드.

기획서 7장: 최상단에 5줄 요약을 고정하고, 상세는 탭으로 내린다.
실행: streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
CLASSES = ["인하", "동결", "인상"]

st.set_page_config(page_title="K-FI Quant Lab", layout="wide")


@st.cache_data
def load_all():
    d = {"panel": pd.read_parquet(PROC / "panel.parquet"),
         "factors": pd.read_parquet(PROC / "ns_factors.parquet")}
    for k, f in [("policy", "policy_predictions.parquet"),
                 ("bfly", "fc_direct_bfly_3s5s10s_bp_h20.parquet"),
                 ("slope", "fc_direct_slope_3s10s_bp_h20.parquet")]:
        p = PROC / f
        d[k] = pd.read_parquet(p) if p.exists() else None
    for k, f in [("bt", "backtest_bfly_3s5s10s_bp.csv"),
                 ("rmse20", "curve_rmse_h20.csv"), ("rmse5", "curve_rmse_h5.csv"),
                 ("credit", "credit_summary.csv")]:
        p = PROC / f
        d[k] = pd.read_csv(p) if p.exists() else None
    return d


D = load_all()
panel, factors = D["panel"], D["factors"]
asof = panel.index[-1]

st.title("K-FI Quant Lab")
st.caption(f"한국 채권시장 통합 예측 시스템 · 기준일 {asof:%Y-%m-%d} · "
           f"데이터 한국은행 ECOS · 표본 {len(panel):,}영업일")

# ==================== 5줄 요약 (기획서 7장) ====================
st.subheader("운용 요약")
c = st.columns(5)

# 1. 금통위
if D["policy"] is not None:
    last = D["policy"].iloc[-1]
    up, hold, cut = last["p_인상"], last["p_동결"], last["p_인하"]
    top = max([("인상", up), ("동결", hold), ("인하", cut)], key=lambda x: x[1])
    c[0].metric("① 다음 금통위", top[0], f"{top[1]:.0%}")
    c[0].caption(f"인상 {up:.0%} / 동결 {hold:.0%} / 인하 {cut:.0%}")
else:
    c[0].metric("① 다음 금통위", "N/A")

# 2. 금리 수준
c[1].metric("② 국고 3년 / 10년",
            f"{panel['ktb3y_pct'].iloc[-1]:.2f}%",
            f"10년 {panel['ktb10y_pct'].iloc[-1]:.2f}%", delta_color="off")
c[1].caption(f"20일 변화 {(panel['ktb3y_pct'].iloc[-1]-panel['ktb3y_pct'].iloc[-21])*100:+.1f}bp")

# 3. 커브
if D["bfly"] is not None:
    b = D["bfly"]["enet"].iloc[-1]
    st_bfly = "곡률 확대" if b > 0 else "곡률 축소"
    c[2].metric("③ 커브 전망 (20일)", st_bfly, f"{b:+.1f}bp")
    c[2].caption(f"3s10s는 예측력 없음 → 거래 대상 제외")
else:
    c[2].metric("③ 커브 전망", "N/A")

# 4. 크레딧
c[3].metric("④ AA- 3년 스프레드", f"{panel['cs_aa3y_bp'].iloc[-1]:.0f}bp",
            f"{panel['cs_aa3y_bp'].iloc[-1]-panel['cs_aa3y_bp'].iloc[-21]:+.1f}bp (20일)")
z = ((panel["cs_aa3y_bp"] - panel["cs_aa3y_bp"].rolling(250).mean())
     / panel["cs_aa3y_bp"].rolling(250).std()).iloc[-1]
sig = "경계" if z > 1.5 else ("주의" if z > 0.5 else "정상")
c[3].caption(f"신호등 **{sig}** (z={z:+.2f})")

# 5. 포지션
if D["bt"] is not None:
    best = D["bt"][D["bt"]["cost_x"] == 1.0].sort_values("sharpe").iloc[-1]
    ok2x = (D["bt"][(D["bt"]["cost_x"] == 2.0) & (D["bt"]["ann_ret_%"] > 0)]).shape[0] > 0
    c[4].metric("⑤ 제안 포지션", "관망" if not ok2x else "선별 진입")
    c[4].caption("비용 2배에서 수익 소멸 → 실거래 권고 안 함")

st.info(
    "**해석 주의** · 모든 성과는 최근 2년을 봉인한 walk-forward 결과다. "
    "커브 전략은 통계적 예측력(방향 63.5%)은 있으나 거래비용 2배에서 수익이 사라져 "
    "운용 신호로 채택하지 않았다. 금통위 모델은 한국은행 문서를 사용하지 않은 "
    "시장가격 기반 버전이다.",
    icon="ℹ️")

# ==================== 탭 ====================
t1, t2, t3, t4, t5 = st.tabs(["일드커브", "커브 예측·전략", "금통위", "크레딧", "데이터"])

with t1:
    col1, col2 = st.columns([2, 1])
    tenors = [("ktb1y_pct", 1), ("ktb2y_pct", 2), ("ktb3y_pct", 3), ("ktb5y_pct", 5),
              ("ktb10y_pct", 10), ("ktb20y_pct", 20), ("ktb30y_pct", 30)]
    fig = go.Figure()
    for lbl, off in [("현재", 0), ("1개월 전", 21), ("1년 전", 250), ("3년 전", 750)]:
        i = max(0, len(panel) - 1 - off)
        dt = panel.index[i]
        xs = [t for c_, t in tenors if not np.isnan(panel[c_].iloc[i])]
        ys = [panel[c_].iloc[i] for c_, t in tenors if not np.isnan(panel[c_].iloc[i])]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers",
                                 name=f"{lbl} ({dt:%y-%m-%d})"))
    fig.update_layout(title="국고채 일드커브", xaxis_title="만기(년)",
                      yaxis_title="수익률(%)", height=420)
    col1.plotly_chart(fig, use_container_width=True)

    col2.markdown("**Nelson-Siegel 요인 (현재)**")
    ns = factors.iloc[-1]
    col2.metric("β1 Level", f"{ns['beta1_level']:.3f}")
    col2.metric("β2 Slope", f"{ns['beta2_slope']:.3f}")
    col2.metric("β3 Curvature", f"{ns['beta3_curv']:.3f}")
    col2.metric("적합오차", f"{ns['fit_rmse_bp']:.2f}bp")
    col2.caption(f"전 기간 평균 적합오차 {factors['fit_rmse_bp'].mean():.2f}bp")

    f2 = go.Figure()
    for c_, nm in [("beta1_level", "β1 Level"), ("beta2_slope", "β2 Slope"),
                   ("beta3_curv", "β3 Curvature")]:
        f2.add_trace(go.Scatter(x=factors.index, y=factors[c_], name=nm, line=dict(width=1)))
    f2.update_layout(title="NS 요인 시계열", height=340)
    st.plotly_chart(f2, use_container_width=True)

with t2:
    st.markdown("#### 만기별 예측 성과 (Random Walk 대비 RMSE 개선율)")
    if D["rmse20"] is not None:
        m = D["rmse20"][["tenor", "rw_rmse_bp", "model_rmse_bp", "improve_pct", "hit_pct"]].copy()
        m["tenor"] = m["tenor"].str.replace("ktb", "").str.replace("_pct", "")
        m.columns = ["만기", "RW RMSE(bp)", "모델 RMSE(bp)", "개선율(%)", "방향정확도(%)"]
        st.dataframe(m.style.format({"RW RMSE(bp)": "{:.2f}", "모델 RMSE(bp)": "{:.2f}",
                                     "개선율(%)": "{:+.2f}", "방향정확도(%)": "{:.1f}"})
                     .background_gradient(subset=["개선율(%)"], cmap="RdYlGn"),
                     use_container_width=True, hide_index=True)
        st.caption("20영업일 예측. 단기물에서만 RW를 유의하게 이기고, 장기물은 이기지 못한다.")

    st.markdown("#### 전략 후보 선별")
    rows = []
    for k, lbl in [("slope", "3s10s 슬로프"), ("bfly", "3-5-10 버터플라이")]:
        if D[k] is None:
            continue
        d = D[k]
        rw = np.sqrt((d["actual"] ** 2).mean())
        md = np.sqrt(((d["actual"] - d["enet"]) ** 2).mean())
        hit = (np.sign(d["enet"]) == np.sign(d["actual"])).mean() * 100
        imp = (rw - md) / rw * 100
        rows.append({"전략": lbl, "corr": d["enet"].corr(d["actual"]),
                     "방향정확도(%)": hit, "RW대비 RMSE(%)": imp,
                     "판정": "거래" if (imp > 0 and hit > 50) else "거래 안 함"})
    if rows:
        st.dataframe(pd.DataFrame(rows).style.format(
            {"corr": "{:+.3f}", "방향정확도(%)": "{:.1f}", "RW대비 RMSE(%)": "{:+.2f}"}),
            use_container_width=True, hide_index=True)

    if D["bt"] is not None:
        st.markdown("#### 버터플라이 백테스트 — 거래비용 스트레스")
        f3 = go.Figure()
        for cx in sorted(D["bt"]["cost_x"].unique()):
            w = D["bt"][D["bt"]["cost_x"] == cx].sort_values("signal_q")
            f3.add_trace(go.Scatter(x=w["signal_q"], y=w["sharpe"],
                                    mode="lines+markers", name=f"비용 {cx:.0f}배"))
        f3.add_hline(y=0, line_dash="dash", line_color="gray")
        f3.update_layout(xaxis_title="신호 임계 분위수", yaxis_title="Sharpe", height=380)
        st.plotly_chart(f3, use_container_width=True)
        st.warning("비용 1배에서만 양(+)의 Sharpe가 나오고 2배부터 손실이다. "
                   "**비용 가정 민감도가 지나치게 높아 운용 신호로 채택하지 않는다.**")

with t3:
    if D["policy"] is not None:
        p = D["policy"]
        f4 = go.Figure()
        f4.add_trace(go.Scatter(x=p["meeting_date"], y=p["p_인상"], name="인상",
                                line=dict(color="crimson")))
        f4.add_trace(go.Scatter(x=p["meeting_date"], y=p["p_인하"], name="인하",
                                line=dict(color="royalblue")))
        f4.add_trace(go.Scatter(x=p["meeting_date"], y=p["p_동결"], name="동결",
                                line=dict(color="gray", width=1)))
        for _, r in p[p["actual"] != "동결"].iterrows():
            f4.add_vline(x=r["meeting_date"], line_dash="dot", line_width=1,
                         line_color="crimson" if r["actual"] == "인상" else "royalblue")
        f4.update_layout(title="금통위 결정 확률 (walk-forward, 점선=실제 변경)",
                         yaxis_title="확률", height=420)
        st.plotly_chart(f4, use_container_width=True)

        st.markdown("#### 최근 10회")
        t = p.tail(10)[["meeting_date", "actual", "p_인하", "p_동결", "p_인상"]].copy()
        t["meeting_date"] = t["meeting_date"].dt.strftime("%Y-%m-%d")
        t.columns = ["회의일", "실제", "인하", "동결", "인상"]
        st.dataframe(t.style.format({"인하": "{:.1%}", "동결": "{:.1%}", "인상": "{:.1%}"}),
                     use_container_width=True, hide_index=True)
        st.caption("한국은행 문서(결정문·의사록)를 사용하지 않은 시장가격 기반 모델이다. "
                   "NLP·LLM 결합은 확장 과제.")

with t4:
    col1, col2 = st.columns(2)
    for col, c_, lbl in [(col1, "cs_aa3y_bp", "AA- 3년"), (col2, "cs_bbb3y_bp", "BBB- 3년")]:
        f5 = go.Figure()
        f5.add_trace(go.Scatter(x=panel.index, y=panel[c_], name=lbl,
                                line=dict(width=1)))
        f5.update_layout(title=f"{lbl} 크레딧 스프레드", yaxis_title="bp", height=340)
        col.plotly_chart(f5, use_container_width=True)
        z_ = ((panel[c_] - panel[c_].rolling(250).mean())
              / panel[c_].rolling(250).std()).iloc[-1]
        col.metric(f"{lbl} 현재", f"{panel[c_].iloc[-1]:.1f}bp", f"z={z_:+.2f}")

    if D["credit"] is not None:
        st.dataframe(D["credit"], use_container_width=True, hide_index=True)
    st.caption("ECOS 공개데이터에 등급×만기 매트릭스가 없어 2개 버킷만 다룬다. "
               "ECM 오차수정계수가 양수로 나와 20일 시계에서는 평균회귀보다 모멘텀이 우세했다.")

with t5:
    st.markdown("#### 패널 요약")
    info = pd.DataFrame({
        "변수": panel.columns,
        "결측률(%)": [panel[c].isna().mean() * 100 for c in panel.columns],
        "최신값": [panel[c].dropna().iloc[-1] if panel[c].notna().any() else np.nan
                for c in panel.columns],
    })
    st.dataframe(info.style.format({"결측률(%)": "{:.2f}", "최신값": "{:.3f}"}),
                 use_container_width=True, hide_index=True, height=420)
    st.caption(f"기간 {panel.index[0]:%Y-%m-%d} ~ {asof:%Y-%m-%d} · "
               f"{len(panel):,}행 × {panel.shape[1]}열 · 출처 한국은행 ECOS")
