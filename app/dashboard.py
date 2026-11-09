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

from src.features.nelson_siegel import CURVE, ns_loadings  # noqa: E402

PROC = ROOT / "data" / "processed"
NS_F = [("beta1_level", "수준"), ("beta2_slope", "기울기"), ("beta3_curv", "곡률")]
NS_COLOR = {"수준": "#4C78A8", "기울기": "#F58518", "곡률": "#54A24B"}
FIG = ROOT / "reports" / "figures"
CLASSES = ["인하", "동결", "인상"]

st.set_page_config(page_title="K-FI Quant Lab", layout="wide")


@st.cache_data
def load_all():
    d = {"panel": pd.read_parquet(PROC / "panel.parquet"),
         "factors": pd.read_parquet(PROC / "ns_factors.parquet")}
    for k, f in [("policy", "policy_predictions.parquet"),
                 ("policy_next", "policy_next.parquet"),
                 ("bfly_live", "live_direct_bfly_3s5s10s_bp_h20.parquet"),
                 ("bfly", "fc_direct_bfly_3s5s10s_bp_h20.parquet"),
                 ("slope", "fc_direct_slope_3s10s_bp_h20.parquet")]:
        p = PROC / f
        d[k] = pd.read_parquet(p) if p.exists() else None
    for k, f in [("bt", "backtest_bfly_3s5s10s_bp.csv"),
                 ("rmse20", "curve_rmse_h20.csv"), ("rmse5", "curve_rmse_h5.csv"),
                 ("credit", "credit_summary.csv")]:
        p = PROC / f
        d[k] = pd.read_csv(p) if p.exists() else None
    d["fc"] = {(h, f): pd.read_parquet(PROC / f"fc_{f}_h{h}.parquet")
               for h in (5, 20) for f, _ in NS_F
               if (PROC / f"fc_{f}_h{h}.parquet").exists()}
    d["live"] = {(h, f): pd.read_parquet(PROC / f"live_{f}_h{h}.parquet")
                 for h in (5, 20) for f, _ in NS_F
                 if (PROC / f"live_{f}_h{h}.parquet").exists()}
    d["lam"] = float(pd.read_json(PROC / "ns_config.json", typ="series")["lambda"])
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
if D["policy_next"] is not None:
    last = D["policy_next"].iloc[0]
    up, hold, cut = last["p_인상"], last["p_동결"], last["p_인하"]
    top = max([("인상", up), ("동결", hold), ("인하", cut)], key=lambda x: x[1])
    c[0].metric(f"① 다음 금통위 ({last['meeting_date']:%m/%d})", top[0], f"{top[1]:.0%}",
                delta_color="off")
    c[0].caption(f"인상 {up:.0%} / 동결 {hold:.0%} / 인하 {cut:.0%}")
else:
    c[0].metric("① 다음 금통위", "N/A")

# 2. 금리 수준
c[1].metric("② 국고 3년 / 10년",
            f"{panel['ktb3y_pct'].iloc[-1]:.2f}%",
            f"10년 {panel['ktb10y_pct'].iloc[-1]:.2f}%", delta_color="off")
c[1].caption(f"20일 변화 {(panel['ktb3y_pct'].iloc[-1]-panel['ktb3y_pct'].iloc[-21])*100:+.1f}bp")

# 3. 커브
if D["bfly_live"] is not None and not D["bfly_live"].empty:
    b = D["bfly_live"]["enet"].iloc[-1]
    st_bfly = "5년 벨리 약세" if b > 0 else "5년 벨리 강세"
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
    "버터플라이 전략은 방향 적중률이 50%를 넘지만 통계적으로 유의하지 않고 거래비용 2배에서 "
    "수익이 사라져 운용 신호로 채택하지 않았다. 금통위 모델은 한국은행 문서를 사용하지 않은 "
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
    st.markdown("#### 금리커브 전망: 수준·기울기·곡률")
    h = st.radio("예측 기간", [20, 5], format_func=lambda x: f"{x}영업일", horizontal=True)
    fc = {nm: D["fc"].get((h, f)) for f, nm in NS_F}
    if any(v is None for v in fc.values()):
        st.warning("NS 요인 예측 파일이 없다. src/models/curve_forecast.py를 먼저 실행할 것.")
    else:
        src = {nm: D["live"].get((h, f)) for f, nm in NS_F}            # 정답 없는 최신 예측
        if any(v is None or v.empty for v in src.values()):
            src = fc                                                   # 없으면 마지막 평가 행
        common = src["수준"].index.intersection(src["기울기"].index).intersection(src["곡률"].index)
        base = common[-1]
        db = np.array([src[nm].loc[base, "enet"] for _, nm in NS_F])  # 요인 변화 예측(%p)
        taus = np.array([t for _, t in CURVE])
        contrib = ns_loadings(taus, D["lam"]) * db * 100              # (만기, 요인) bp
        total = contrib.sum(axis=1)
        j = {t: k for k, (_, t) in enumerate(CURVE)}                  # 만기(년) → 행 번호
        d3, d10 = total[j[3.0]], total[j[10.0]]
        lvl, spr = (d3 + d10) / 2, d10 - d3

        # 시나리오: 금리 방향(불/베어) × 3s10s 방향(스티프/플랫). 1bp 미만은 보합 처리
        if abs(lvl) < 1 and abs(spr) < 1:
            scen, desc = "보합", "커브 형태 변화가 1bp 미만"
        elif abs(spr) < 1:
            scen = "불 평행이동" if lvl < 0 else "베어 평행이동"
            desc = "전 구간 비슷하게 " + ("하락(강세)" if lvl < 0 else "상승(약세)")
        else:
            scen = ("불" if lvl < 0 else "베어") + (" 스티프닝" if spr > 0 else " 플래트닝")
            lead = "단기물" if (lvl < 0) == (spr > 0) else "장기물"
            desc = f"{lead} 주도 " + ("강세(금리 하락)" if lvl < 0 else "약세(금리 상승)")

        i0 = panel.index.get_loc(base)
        target = panel.index[i0 + h] if i0 + h < len(panel) else None
        st.markdown(f"### {scen}  \n{desc} · 국고 3년 **{d3:+.1f}bp**, 10년 **{d10:+.1f}bp**, "
                    f"3s10s **{spr:+.1f}bp**")
        st.caption(f"예측 기준일 {base:%Y-%m-%d} → 목표일 "
                   + (f"{target:%Y-%m-%d} (이미 실현된 구간, 아래 차트에 실제 변화 표시)"
                      if target is not None else f"{h}영업일 후"))

        hit = {nm: (np.sign(fc[nm]["enet"]) == np.sign(fc[nm]["actual"])).mean() * 100
               for _, nm in NS_F}
        lvl_f = contrib[0, 0]
        slp_f = contrib[j[10.0], 1] - contrib[j[3.0], 1]
        cur_f = 2 * contrib[j[3.0], 2] - contrib[j[1.0], 2] - contrib[j[10.0], 2]
        k1, k2, k3 = st.columns(3)
        k1.metric("수준 · 전 구간 평행이동",
                  "금리 상승(약세)" if lvl_f > 0 else "금리 하락(강세)",
                  f"{lvl_f:+.1f}bp", delta_color="inverse")
        k1.caption(f"모든 만기를 같은 폭으로 움직이는 부분 · 방향 적중 {hit['수준']:.0f}%")
        k2.metric("기울기 · 3s10s 스프레드",
                  "스티프닝" if slp_f > 0 else "플래트닝",
                  f"{slp_f:+.1f}bp", delta_color="off")
        k2.caption(f"10년-3년 금리차 변화 중 기울기 요인 몫 · 방향 적중 {hit['기울기']:.0f}%")
        k3.metric("곡률 · 1-3-10 버터플라이",
                  "벨리(중기물) 약세" if cur_f > 0 else "벨리(중기물) 강세",
                  f"{cur_f:+.1f}bp", delta_color="off")
        k3.caption(f"3년물이 1·10년물 대비 얼마나 더 오르나(+) · 방향 적중 {hit['곡률']:.0f}%")

        lbl = [f"{t:g}년" for t in taus]
        fz = go.Figure()
        for k, (_, nm) in enumerate(NS_F):
            fz.add_trace(go.Bar(x=lbl, y=contrib[:, k], name=nm, marker_color=NS_COLOR[nm]))
        fz.add_trace(go.Scatter(x=lbl, y=total, name="예측 합계", mode="markers+lines",
                                marker=dict(symbol="diamond", size=10, color="black"),
                                line=dict(color="black", width=1)))
        if target is not None:
            cols = [c for c, _ in CURVE]
            real = (panel[cols].iloc[i0 + h] - panel[cols].iloc[i0]).to_numpy() * 100
            fz.add_trace(go.Scatter(x=lbl, y=real, name="실제 변화", mode="markers",
                                    marker=dict(symbol="x", size=10, color="crimson")))
        fz.add_hline(y=0, line_color="gray", line_width=1)
        fz.update_layout(barmode="relative", height=400, yaxis_title="금리 변화(bp)",
                         title=f"만기별 {h}영업일 금리 변화 예측 (요인별 분해)",
                         legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fz, use_container_width=True)
        st.caption(f"+는 금리 상승(채권 가격 하락). NS 곡률 요인은 {1.7933 / D['lam']:.1f}년 "
                   "근처에서 가장 크게 움직여 3-5-10이 아닌 1-3-10 버터플라이로 쟀다. "
                   "요인별 예측을 합치면 신호가 뒤집힐 수 있어 매매 신호는 아래 직접 예측 모델을 쓴다.")

    st.divider()
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
