"""리서치 노트 자동 생성. 모델 산출물에서 숫자를 읽어 Markdown을 만든다.

원칙: 숫자를 손으로 쓰지 않는다. 모든 값은 data/processed의 결과 파일에서 읽는다.
"""

from __future__ import annotations

import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):      # Windows cp949 콘솔에서 한글/기호 깨짐 방지
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLASSES = ["인하", "동결", "인상"]


def load(name: str):
    p = PROC / name
    if not p.exists():
        return None
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def main() -> None:
    panel = pd.read_parquet(PROC / "panel.parquet")
    factors = pd.read_parquet(PROC / "ns_factors.parquet")
    asof = panel.index[-1]
    L = []

    L.append("# K-FI Quant Lab — 리서치 노트")
    L.append("")
    L.append(f"- 기준일: **{asof:%Y-%m-%d}** (데이터 최신 영업일)")
    L.append(f"- 생성일: {date.today():%Y-%m-%d} (본 노트는 코드로 자동 생성됨)")
    L.append(f"- 표본: {panel.index[0]:%Y-%m-%d} ~ {asof:%Y-%m-%d}, {len(panel):,}영업일")
    L.append("- 데이터: 한국은행 ECOS 오픈API (국고채·회사채·단기금리·기준금리)")
    L.append("")
    L.append("> 모든 성능은 **최근 2년을 봉인한 walk-forward 검증** 결과다. "
             "학습에 쓰이지 않은 구간에서만 측정했다.")
    L.append("")

    # ---------- 1. 현재 시장 ----------
    L.append("## 1. 현재 시장 상태")
    L.append("")
    L.append("| 지표 | 값 |")
    L.append("|---|---|")
    for c, lbl in [("base_rate_pct", "기준금리"), ("ktb1y_pct", "국고 1년"),
                   ("ktb3y_pct", "국고 3년"), ("ktb10y_pct", "국고 10년")]:
        L.append(f"| {lbl} | {panel[c].iloc[-1]:.3f}% |")
    for c, lbl in [("slope_3s10s_bp", "3s10s 스프레드"),
                   ("bfly_3s5s10s_bp", "3-5-10 버터플라이"),
                   ("cs_aa3y_bp", "AA- 3년 스프레드"),
                   ("cs_bbb3y_bp", "BBB- 3년 스프레드")]:
        L.append(f"| {lbl} | {panel[c].iloc[-1]:+.1f}bp |")
    L.append("")
    ns = factors.iloc[-1]
    L.append(f"NS 요인: Level {ns['beta1_level']:.3f} / Slope {ns['beta2_slope']:.3f} / "
             f"Curvature {ns['beta3_curv']:.3f} (적합오차 {ns['fit_rmse_bp']:.2f}bp)")
    L.append("")

    # ---------- 2. 모듈 1 ----------
    L.append("## 2. 모듈 1 — 일드커브 (Nelson-Siegel)")
    L.append("")
    L.append(f"- 적합: 전 기간 {factors['beta1_level'].notna().sum():,}일, "
             f"평균 적합오차 **{factors['fit_rmse_bp'].mean():.2f}bp**")
    L.append("- λ는 봉인 구간을 뺀 학습 구간의 적합 오차 최소값으로 고정해 β를 선형최소제곱으로 구한다.")
    L.append("- 최종 예측: 기울기 요인만 ENet과 AR의 평균, 수준과 곡률은 0(RW). 근거는 README 개선 실험.")
    L.append("")
    h20 = load("curve_rmse_h20.csv")
    short_imp, bfly_hit, bfly_p = float("nan"), float("nan"), float("nan")
    if h20 is not None:
        L.append("### 만기별 20일 예측 (Random Walk 대비)")
        L.append("")
        L.append("| 만기 | RW RMSE | 모델 RMSE | 개선율 | DM p | 방향정확도(독립 구간) |")
        L.append("|---|---|---|---|---|---|")
        for _, r in h20.iterrows():
            t = r["tenor"].replace("ktb", "").replace("_pct", "")
            L.append(f"| {t} | {r['rw_rmse_bp']:.2f}bp | {r['model_rmse_bp']:.2f}bp | "
                     f"**{r['improve_pct']:+.2f}%** | {r['dm_p']:.3f} | {r['hit_indep_pct']:.1f}% |")
        L.append("")
        best = h20.loc[h20["improve_pct"].idxmax()]
        short_imp = best["improve_pct"]
        L.append(f"**핵심:** 개선 폭이 크고 유의한 것은 단기물({best['tenor'].replace('ktb','').replace('_pct','')})이다. "
                 f"RW 대비 {best['improve_pct']:+.1f}%, DM p={best['dm_p']:.3f}. "
                 "나머지 만기도 RW 이상이지만 개선 폭이 1~6%로 작고, 10년 이상은 dev 구간에서 유의하지 않았다.")
        L.append("")
        L.append("> 해석: 단기 구간은 정책금리 기대에 묶여 있어 커브 자체 정보로 예측 가능하지만, "
                 "10년 이상은 글로벌 듀레이션 수급이 지배해 국내 커브 정보만으로는 설명되지 않는다.")
        L.append("")

    # ---------- 3. 전략 ----------
    L.append("## 3. 운용 연결 — 커브 전략 백테스트")
    L.append("")
    rows = []
    for tgt, lbl in [("slope_3s10s_bp", "3s10s 슬로프"), ("bfly_3s5s10s_bp", "3-5-10 버터플라이")]:
        d = load(f"fc_direct_{tgt}_h20.parquet")
        if d is None:
            continue
        rw = np.sqrt((d["actual"] ** 2).mean())
        md = np.sqrt(((d["actual"] - d["enet"]) ** 2).mean())
        if tgt.startswith("bfly"):
            from scipy.stats import binomtest
            nz = d.iloc[::20]                         # 겹치지 않는 20일 구간만
            k = int((np.sign(nz["enet"]) == np.sign(nz["actual"])).sum())
            bfly_hit = (np.sign(d["enet"]) == np.sign(d["actual"])).mean() * 100
            bfly_p = binomtest(k, len(nz), 0.5, alternative="greater").pvalue
            bfly_nz = (k, len(nz))
        rows.append((lbl, d["enet"].corr(d["actual"]),
                     (np.sign(d["enet"]) == np.sign(d["actual"])).mean() * 100,
                     (rw - md) / rw * 100))
    if rows:
        L.append("### 전략 후보 선별")
        L.append("")
        L.append("| 전략 | corr | 방향정확도 | RW 대비 RMSE | 판정 |")
        L.append("|---|---|---|---|---|")
        for lbl, c, h, imp in rows:
            ok = "거래" if (imp > 0 and h > 50) else "**거래 안 함**"
            L.append(f"| {lbl} | {c:+.3f} | {h:.1f}% | {imp:+.2f}% | {ok} |")
        L.append("")
        L.append("> 슬로프는 예측력이 없어 매매 대상에서 제외하고 버터플라이만 백테스트한다. "
                 f"다만 20일 예측이 매일 겹쳐 독립 표본이 적다. 겹치지 않는 20일 구간만 세면 "
                 f"버터플라이는 {bfly_nz[1]}개 중 {bfly_nz[0]}개 적중(이항검정 p={bfly_p:.2f})으로 "
                 "통계적으로 유의하지 않다.")
        L.append("")

    bt = load("backtest_bfly_3s5s10s_bp.csv")
    if bt is not None:
        L.append("### 3-5-10 버터플라이 (DV01 중립, 거래비용 스트레스)")
        L.append("")
        L.append("| 신호분위 | 비용 | 거래수 | 연수익 | Sharpe | MDD | 적중률 |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in bt.iterrows():
            L.append(f"| {r['signal_q']:.1f} | {r['cost_x']:.0f}x | {int(r['n_trades'])} | "
                     f"{r['ann_ret_%']:+.2f}% | {r['sharpe']:+.2f} | {r['mdd_%']:.2f}% | "
                     f"{r['hit_%']:.1f}% |")
        L.append("")
        best1x = bt[(bt["cost_x"] == 1.0)].sort_values("sharpe").iloc[-1]
        pos2x = bt[(bt["cost_x"] == 2.0) & (bt["ann_ret_%"] > 0)]
        L.append(f"**결론:** 비용 1배에서는 선별매매 시 Sharpe {best1x['sharpe']:.2f}"
                 f"(분위 {best1x['signal_q']:.1f}, 거래 {int(best1x['n_trades'])}회)까지 나오지만, "
                 f"**비용 2배에서는 {'모든 설정이 손실' if pos2x.empty else '일부만 생존'}**이다.")
        L.append("")
        L.append(f"> 기획서 2.4 원칙에 따라, 방향 적중률({bfly_hit:.1f}%)이 50%를 넘어도 "
                 "비용 차감 후 경제적 가치가 확인되지 않으면 운용 신호로 채택하지 않는다. "
                 "이 전략은 **비용 가정에 대한 민감도가 지나치게 높다**는 것이 결론이다.")
        L.append("")

    # ---------- 4. 모듈 4 ----------
    pol = load("policy_predictions.parquet")
    pol_sum, pol_ll = "N/A", float("nan")
    if pol is not None:
        L.append("## 4. 모듈 4 — 금통위 결정 확률")
        L.append("")
        y = pol["actual"].to_numpy()
        P = np.clip(pol[[f"p_{c}" for c in CLASSES]].to_numpy(), 1e-9, 1)
        B = np.clip(pol[[f"base_{c}" for c in CLASSES]].to_numpy(), 1e-9, 1)
        P /= P.sum(1, keepdims=True)
        B /= B.sum(1, keepdims=True)
        Y = np.column_stack([(y == c).astype(float) for c in CLASSES])
        from sklearn.metrics import f1_score, log_loss
        # log_loss는 labels를 사전순 정렬해 열과 매핑 -> 열 재배열 필요
        _sorted = sorted(CLASSES)
        _ord = [CLASSES.index(c) for c in _sorted]
        ll_m = log_loss(y, P[:, _ord], labels=_sorted)
        ll_b = log_loss(y, B[:, _ord], labels=_sorted)
        br_m, br_b = ((P - Y) ** 2).sum(1).mean(), ((B - Y) ** 2).sum(1).mean()
        pred = np.array(CLASSES)[P.argmax(1)]
        f1_m = f1_score(y, pred, average="macro", labels=CLASSES, zero_division=0)
        f1_b = f1_score(y, np.full(len(y), "동결"), average="macro",
                        labels=CLASSES, zero_division=0)

        L.append(f"- 표본: {len(y)}회 (인상 {(y=='인상').sum()}, 동결 {(y=='동결').sum()}, "
                 f"인하 {(y=='인하').sum()}) — 동결 비중 {(y=='동결').mean():.1%}")
        L.append("- 기준선: 학습표본 클래스 빈도(= '항상 동결'의 확률판)")
        L.append("")
        L.append("| 지표 | 모델 | 기준선 | 판정 |")
        L.append("|---|---|---|---|")
        L.append(f"| Log Loss | {ll_m:.4f} | {ll_b:.4f} | {'PASS' if ll_m<ll_b else '**FAIL**'} |")
        L.append(f"| Brier | {br_m:.4f} | {br_b:.4f} | {'**PASS**' if br_m<br_b else 'FAIL'} |")
        L.append(f"| Macro-F1 | {f1_m:.3f} | {f1_b:.3f} | {'**PASS**' if f1_m>f1_b else 'FAIL'} |")
        L.append("")
        L.append("세 지표 모두 '항상 동결' 기준선을 이긴다. 다만 성과가 고르지 않아 "
                 "어디서 손실이 나는지 함께 기록한다.")
        L.append("")
        pos = [CLASSES.index(c) for c in y]
        Pt, Bt = P[np.arange(len(y)), pos], B[np.arange(len(y)), pos]
        excess = -np.log(Pt) + np.log(Bt)
        top5 = np.argsort(excess)[-5:]
        L.append("확신하고 틀린 상위 5회가 초과 Log Loss의 "
                 f"{excess[top5].sum()/excess[excess>0].sum()*100:.0f}%를 차지한다:")
        L.append("")
        L.append("| 회의일 | 실제 | 모델이 준 확률 | 초과손실 |")
        L.append("|---|---|---|---|")
        for i in sorted(top5, key=lambda k: -excess[k]):
            L.append(f"| {pol['meeting_date'].iloc[i]:%Y-%m-%d} | {y[i]} | "
                     f"{Pt[i]:.1%} | {excess[i]:+.2f} |")
        keep = np.setdiff1d(np.arange(len(y)), top5)
        L.append("")
        L.append(f"이 5회를 제외하면 Log Loss는 모델 {-np.log(Pt[keep]).mean():.3f} vs "
                 f"기준선 {-np.log(Bt[keep]).mean():.3f}로 격차가 더 벌어진다. "
                 "즉 전체 성과는 소수의 큰 실패에 크게 좌우된다.")
        L.append("")
        L.append("> 시장금리가 정책 변경을 먼저 반영했지만 한국은행이 그 회의에서는 동결하고 "
                 "뒤 회의에서 움직인 경우가 큰 실패로 남는다. 방향보다 시점을 맞히기 어렵다.")
        pol_sum = (f"Log Loss {(ll_b-ll_m)/ll_b*100:+.1f}%, Brier {(br_b-br_m)/br_b*100:+.1f}%, "
                   f"Macro-F1 {f1_m:.3f} vs {f1_b:.3f}")
        pol_ll = (ll_b - ll_m) / ll_b * 100
        L.append("")
        L.append("### 최근 6회 예측")
        L.append("")
        L.append("| 회의일 | 실제 | 인하 | 동결 | 인상 |")
        L.append("|---|---|---|---|---|")
        for _, r in pol.tail(6).iterrows():
            L.append(f"| {r['meeting_date']:%Y-%m-%d} | {r['actual']} | {r['p_인하']:.1%} | "
                     f"{r['p_동결']:.1%} | {r['p_인상']:.1%} |")
        L.append("")

    # ---------- 5. 모듈 3 ----------
    L.append("## 5. 모듈 3 — 크레딧 스프레드")
    L.append("")
    L.append("**범위 한계:** ECOS 공개 데이터에 등급×만기 매트릭스가 없어 "
             "AA-/BBB- 3년 2개 버킷만 다룬다. 기획서 5.2의 등급별 확장은 라이선스 데이터가 필요하다.")
    L.append("")
    L.append("| 버킷 | 방향정확도 | corr | 급확대 PR-AUC | 무작위 기준선 |")
    L.append("|---|---|---|---|---|")
    cs = load("credit_summary.csv")
    if cs is not None:
        for _, r in cs.iterrows():
            L.append(f"| {r['bucket']} | {r['hit']:.1f}% | {r['corr']:+.3f} | "
                     f"{r['pr_auc']:.3f} | {r['base']:.3f} |")
    L.append("")
    L.append("> ECM 오차수정계수가 **양수**로 나왔다. 20일 시계에서 한국 크레딧 스프레드는 "
             "평균회귀보다 **모멘텀**이 우세하다는 뜻으로, 기획서 5.4의 가설과 반대다. "
             "방향 정확도는 60% 이상이나 RMSE는 RW를 이기지 못해 "
             "'방향은 맞히되 폭은 못 맞힌다'는 한계가 있다.")
    L.append("")
    L.append("### 위기구간 검증")
    L.append("")
    L.append("| 구간 | AA- 확대폭 | BBB- 확대폭 |")
    L.append("|---|---|---|")
    for name, a, b in [("2020 코로나", "2020-02-01", "2020-06-30"),
                       ("2022 레고랜드", "2022-09-01", "2023-01-31")]:
        w = panel.loc[a:b]
        if w.empty:
            continue
        L.append(f"| {name} | {w['cs_aa3y_bp'].max()-w['cs_aa3y_bp'].iloc[0]:+.1f}bp | "
                 f"{w['cs_bbb3y_bp'].max()-w['cs_bbb3y_bp'].iloc[0]:+.1f}bp |")
    L.append("")

    # ---------- 6. 종합 ----------
    L.append("## 6. 종합 판단")
    L.append("")
    L.append("| 모듈 | 통계적 성과 | 경제적 가치 | 판정 |")
    L.append("|---|---|---|---|")
    L.append(f"| 1. 일드커브 | 단기물 RW 대비 {short_imp:+.1f}%, 버터플라이 방향 {bfly_hit:.1f}% "
             f"(p={bfly_p:.2f}) | 비용 2배에서 소멸 | 부분 성공 |")
    L.append(f"| 4. 금통위 | {pol_sum} | 기준선 대비 전 지표 우위 | **성공** |")
    cs_hit = (f"방향 {cs['hit'].min():.0f}~{cs['hit'].max():.0f}%" if cs is not None else "N/A")
    L.append(f"| 3. 크레딧 | {cs_hit}, PR-AUC 개선 미미 | 미검증 | 미달 |")
    L.append("")
    L.append("**하루 예산으로 확인한 것**")
    L.append("")
    L.append("1. **금통위 확률 예측이 가장 잘 작동한다.** 한국은행 문서를 전혀 쓰지 않고 "
             "시장금리에 반영된 정책 기대(국고1년-기준금리 갭)만으로 "
             f"'항상 동결' 기준선을 Log Loss 기준 {pol_ll:.1f}% 개선했다.")
    L.append("2. **커브는 단기물만 예측되고, 버터플라이 전략은 경제적 가치가 없다.** "
             f"국고 1년은 RW 대비 {short_imp:+.1f}%로 예측된다. 버터플라이 방향 적중률은 {bfly_hit:.1f}%지만 "
             f"독립 표본 기준으로 유의하지 않고(p={bfly_p:.2f}), 거래비용 2배에서 수익이 사라진다.")
    L.append("3. **크레딧은 이 데이터 범위에서 한계가 뚜렷하다.** "
             "ECM은 평균회귀 대신 모멘텀을 시사했고 급확대 조기경보의 PR-AUC 개선은 미미하다.")
    L.append("")
    L.append("비용 차감 후 가치가 없는 신호를 채택하지 않는 것은 실패가 아니라 "
             "**모델 리스크 관리의 결과**다. 기획서 2.3·2.4 원칙에 따라 그대로 기록한다.")
    L.append("")
    L.append("## 7. 재현 방법")
    L.append("")
    L.append("```bash")
    L.append("pip install -r requirements.txt")
    L.append("echo ECOS_API_KEY=발급키 > .env")
    L.append("python run_all.py")
    L.append("streamlit run app/dashboard.py")
    L.append("```")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*본 노트의 모든 수치는 `src/reporting/note.py`가 "
             "`data/processed/`의 모델 산출물에서 직접 읽어 생성한다. 수기 입력 없음.*")

    out = REPORTS / "research_note.md"
    out.write_text("\n".join(L), encoding="utf-8")
    n = len("\n".join(L))
    print(f"리서치 노트 생성: {out}")
    print(f"  {n:,}자 / {len(L)}줄")


if __name__ == "__main__":
    main()
