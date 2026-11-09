"""모듈 4: 금통위 인상/동결/인하 확률 예측 (다항 로짓).

설계 (기획서 6.5-6.6 / 계획 Block 3)
- 표본이 적다(실제 회의 163회, 그중 동결 79%). 딥러닝 안 쓴다. Multinomial Logit.
- 필수 기준선: '항상 동결'. 동결이 79%라 정확도만 보면 속는다.
  -> Log Loss / Brier / Macro-F1로 평가한다.
- 검증: 회의 순서를 지킨 expanding window. shuffle 금지.
- 특징은 회의 '직전 영업일'까지의 시장 데이터만 사용(look-ahead 차단).

핵심 아이디어
- 시장은 정책을 선반영한다. 국고1년-기준금리 갭이 크면 인상 기대가 반영된 것.
  텍스트 없이도 이 갭이 강력한 예측변수가 된다.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

CLASSES = ["인하", "동결", "인상"]
MIN_TRAIN = 60          # 최소 60회 학습 후 예측 시작
SEED = 42


def build_features(meetings: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """각 회의 직전 영업일 기준 특징."""
    rows = []
    idx = panel.index

    for _, m in meetings.iterrows():
        dt = m["meeting_date"]
        prior = idx[idx < dt]                       # 회의 당일 제외 = look-ahead 차단
        if len(prior) < 260:
            continue
        t = prior[-1]
        w = panel.loc[:t]

        base = w["base_rate_pct"].iloc[-1]
        gap1y = w["ktb1y_base_gap_bp"].iloc[-1]     # 시장에 반영된 정책 기대
        gap3y = w["ktb3y_base_gap_bp"].iloc[-1]
        call_gap = w["call_base_gap_bp"].iloc[-1]

        rows.append({
            "meeting_date": dt,
            "label": m["label"],
            "base_rate": base,
            "gap_1y_bp": gap1y,
            "gap_3y_bp": gap3y,
            "call_gap_bp": call_gap,
            # 기대의 변화(최근 20/60일)
            "gap1y_d20": gap1y - w["ktb1y_base_gap_bp"].iloc[-21],
            "gap1y_d60": gap1y - w["ktb1y_base_gap_bp"].iloc[-61],
            # 커브·유동성
            "slope_3s10s_bp": w["slope_3s10s_bp"].iloc[-1],
            "cd_msb_bp": w["cd_msb_bp"].iloc[-1],
            "cs_aa3y_bp": w["cs_aa3y_bp"].iloc[-1],
            "vol20": w["ktb3y_vol20_bp"].iloc[-1],
            # 금리 모멘텀
            "ktb3y_d60": (w["ktb3y_pct"].iloc[-1] - w["ktb3y_pct"].iloc[-61]) * 100,
            # 정책 관성: 직전 12개월 누적 변화
            "base_d250_bp": (base - w["base_rate_pct"].iloc[-251]) * 100,
        })
    return pd.DataFrame(rows).dropna().reset_index(drop=True)


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """회의 순서를 지킨 expanding window 예측."""
    feat_cols = [c for c in df.columns if c not in ("meeting_date", "label")]
    X = df[feat_cols].to_numpy(float)
    y = df["label"].to_numpy()

    out = []
    for i in range(MIN_TRAIN, len(df)):
        Xtr, ytr = X[:i], y[:i]
        if len(np.unique(ytr)) < 2:
            continue
        sc = StandardScaler().fit(Xtr)
        # C=0.05 무가중: 진단 결과 이 조합이 Log Loss/Brier 모두에서 최선이었다.
        # class_weight="balanced"는 소수클래스 확률을 부풀려 Log Loss를 크게 악화시킨다.
        clf = LogisticRegression(
            max_iter=2000, C=0.05, random_state=SEED,
        ).fit(sc.transform(Xtr), ytr)

        p = clf.predict_proba(sc.transform(X[i : i + 1]))[0]
        prob = dict(zip(clf.classes_, p))
        # 학습에 없던 클래스는 0
        row = {"meeting_date": df["meeting_date"].iloc[i], "actual": y[i]}
        for c in CLASSES:
            row[f"p_{c}"] = prob.get(c, 0.0)
        # 기준선: 학습표본의 클래스 빈도 (= '항상 동결'의 확률판)
        freq = pd.Series(ytr).value_counts(normalize=True)
        for c in CLASSES:
            row[f"base_{c}"] = float(freq.get(c, 0.0))
        out.append(row)
    return pd.DataFrame(out)


def evaluate(res: pd.DataFrame) -> None:
    y = res["actual"].to_numpy()
    P = res[[f"p_{c}" for c in CLASSES]].to_numpy()
    B = res[[f"base_{c}" for c in CLASSES]].to_numpy()
    # 정규화 + 수치안정
    P = np.clip(P / P.sum(1, keepdims=True), 1e-9, 1)
    B = np.clip(B / B.sum(1, keepdims=True), 1e-9, 1)

    # sklearn의 log_loss는 labels를 사전순으로 정렬해 확률 열과 매핑한다.
    # CLASSES 순서(인하/동결/인상)로 만든 P를 그대로 넘기면 열이 어긋나므로 재배열한다.
    _sorted = sorted(CLASSES)
    _ord = [CLASSES.index(c) for c in _sorted]
    ll_m = log_loss(y, P[:, _ord], labels=_sorted)
    ll_b = log_loss(y, B[:, _ord], labels=_sorted)
    Y = np.column_stack([(y == c).astype(float) for c in CLASSES])
    br_m = ((P - Y) ** 2).sum(1).mean()
    br_b = ((B - Y) ** 2).sum(1).mean()

    pred_m = np.array(CLASSES)[P.argmax(1)]
    always_hold = np.full(len(y), "동결")

    print(f"  표본 {len(y)}회 (인상 {(y=='인상').sum()}, 동결 {(y=='동결').sum()}, "
          f"인하 {(y=='인하').sum()})\n")
    print(f"  {'지표':<14}{'모델':>10}{'기준선':>10}{'개선':>10}")
    print("  " + "-" * 44)
    print(f"  {'Log Loss':<14}{ll_m:>10.4f}{ll_b:>10.4f}{(ll_b-ll_m)/ll_b*100:>9.1f}%")
    print(f"  {'Brier':<14}{br_m:>10.4f}{br_b:>10.4f}{(br_b-br_m)/br_b*100:>9.1f}%")
    print(f"  {'정확도':<14}{(pred_m==y).mean()*100:>9.1f}%"
          f"{(always_hold==y).mean()*100:>9.1f}%")
    print(f"  {'Macro-F1':<14}{f1_score(y,pred_m,average='macro',labels=CLASSES,zero_division=0):>10.3f}"
          f"{f1_score(y,always_hold,average='macro',labels=CLASSES,zero_division=0):>10.3f}")

    f1_m = f1_score(y, pred_m, average="macro", labels=CLASSES, zero_division=0)
    f1_b = f1_score(y, always_hold, average="macro", labels=CLASSES, zero_division=0)
    print(f"\n  [{'PASS' if ll_m < ll_b else 'FAIL'}] Log Loss "
          f"{'개선' if ll_m < ll_b else '악화'} (기획서 6.8 성공기준)")
    print(f"  [{'PASS' if br_m < br_b else 'FAIL'}] Brier    "
          f"{'개선' if br_m < br_b else '악화'}")
    print(f"  [{'PASS' if f1_m > f1_b else 'FAIL'}] Macro-F1 "
          f"{'개선' if f1_m > f1_b else '악화'}")

    # 지표가 엇갈리는 이유를 수치로 남긴다 (해석 없이 넘어가지 않는다)
    pos = [CLASSES.index(c) for c in y]
    Pt = P[np.arange(len(y)), pos]
    Bt = B[np.arange(len(y)), pos]
    excess = -np.log(Pt) + np.log(Bt)
    top5 = np.argsort(excess)[-5:]
    print(f"\n  [Log Loss 악화 원인] 확신하고 틀린 상위 5회가 초과손실의 "
          f"{excess[top5].sum() / excess[excess > 0].sum() * 100:.0f}% 차지")
    for i in sorted(top5, key=lambda k: -excess[k]):
        print(f"    {res['meeting_date'].iloc[i]:%Y-%m-%d}  실제={y[i]}  "
              f"모델확률={Pt[i]:>5.1%}  초과손실={excess[i]:+.2f}")
    keep = np.setdiff1d(np.arange(len(y)), top5)
    print(f"    -> 이 5회 제외 시 Log Loss 모델 {-np.log(Pt[keep]).mean():.3f} "
          f"vs 기준선 {-np.log(Bt[keep]).mean():.3f}")

    # calibration: 인상 확률 구간별 실제 인상 비율
    print("\n  [인상 확률 calibration]")
    ph = res["p_인상"].to_numpy()
    bins = [0, .1, .2, .3, .5, 1.01]
    print(f"  {'구간':<12}{'n':>5}{'평균예측':>10}{'실제':>8}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (ph >= lo) & (ph < hi)
        if m.sum() == 0:
            continue
        print(f"  {f'{lo:.0%}-{hi:.0%}':<12}{m.sum():>5}{ph[m].mean():>9.1%}"
              f"{(y[m]=='인상').mean():>8.1%}")


if __name__ == "__main__":
    np.random.seed(SEED)
    panel = pd.read_parquet(PROC / "panel.parquet")
    meetings = pd.read_parquet(PROC / "policy_meetings.parquet")

    df = build_features(meetings, panel)
    print(f"특징 생성: {len(df)}회 "
          f"({df['meeting_date'].min():%Y-%m-%d} ~ {df['meeting_date'].max():%Y-%m-%d})\n")

    res = walk_forward(df)
    print(f"walk-forward 예측 {len(res)}회 "
          f"({res['meeting_date'].min():%Y-%m-%d} ~ {res['meeting_date'].max():%Y-%m-%d})\n")
    evaluate(res)

    print("\n  [최근 8회 예측]")
    print(f"  {'회의일':<13}{'실제':>6}{'인하':>8}{'동결':>8}{'인상':>8}")
    for _, r in res.tail(8).iterrows():
        print(f"  {r['meeting_date']:%Y-%m-%d}{r['actual']:>7}"
              f"{r['p_인하']:>8.1%}{r['p_동결']:>8.1%}{r['p_인상']:>8.1%}")

    res.to_parquet(PROC / "policy_predictions.parquet", index=False)

    # 다음 회의 예측: 전체 표본으로 학습, 데이터 최신일까지의 시장가격으로 예측
    from src.features.policy_events import MEETINGS_CSV
    cal = pd.to_datetime(pd.read_csv(MEETINGS_CSV)["date"])
    upcoming = cal[cal > panel.index.max()]
    if upcoming.empty:
        print("\n다음 회의 일정이 configs/mpc_meetings.csv에 없다. 일정을 추가할 것.")
    else:
        nxt = build_features(pd.DataFrame({"meeting_date": [upcoming.iloc[0]], "label": ["미정"]}),
                             panel)
        feat = [c for c in df.columns if c not in ("meeting_date", "label")]
        sc = StandardScaler().fit(df[feat])
        clf = LogisticRegression(max_iter=2000, C=0.05, random_state=SEED).fit(
            sc.transform(df[feat]), df["label"])
        prob = dict(zip(clf.classes_, clf.predict_proba(sc.transform(nxt[feat]))[0]))
        out = pd.DataFrame([{"meeting_date": upcoming.iloc[0], "asof": panel.index.max(),
                             **{f"p_{c}": prob.get(c, 0.0) for c in CLASSES}}])
        out.to_parquet(PROC / "policy_next.parquet", index=False)
        print(f"\n  [다음 회의 {upcoming.iloc[0]:%Y-%m-%d}] 시장 기준일 {panel.index.max():%Y-%m-%d}  "
              + "  ".join(f"{c} {prob.get(c, 0.0):.1%}" for c in CLASSES))
    print(f"\n저장: {PROC / 'policy_predictions.parquet'}")
