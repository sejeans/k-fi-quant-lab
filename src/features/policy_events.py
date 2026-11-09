"""기준금리 일별 시계열 + 실제 금통위 회의일 -> 이벤트·라벨 생성.

- 회의일: configs/mpc_meetings.csv. 한국은행 '통화정책방향 결정회의' 연도별 일정
  (bok.or.kr/portal/singl/crncyPolicyDrcMtg/listYear.do?mtgSe=A&menuNo=200755)에서 수집.
  2020-03-16 임시회의 포함. 매년 초 새 일정을 추가한다.
- 라벨: ECOS 722Y001 기준금리(일별 계단함수)의 변경일을 회의일에 붙인다.
  변경은 보통 회의 당일 발효, 2020-03 임시회의만 다음 날 발효 -> 회의일 + 3일 이내 변경을 그 회의 결과로 본다.
- 이전 버전은 '매월 둘째 목요일' 근사로 가짜 동결 회의를 만들었다(186회 중 21회가 20일 미만 간격).
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PROC = ROOT / "data" / "processed"


def detect_changes(base: pd.Series) -> pd.DataFrame:
    """기준금리 변경 이벤트 추출."""
    b = base.dropna()
    d = b.diff()
    chg = d[d.abs() > 1e-9]
    return pd.DataFrame({
        "date": chg.index,
        "change_bp": (chg.to_numpy() * 100).round(0),
        "rate_after": b.reindex(chg.index).to_numpy(),
    }).reset_index(drop=True)


MEETINGS_CSV = ROOT / "configs" / "mpc_meetings.csv"
EFFECT_LAG_DAYS = 3


def build_meetings(base: pd.Series) -> pd.DataFrame:
    """실제 회의일에 기준금리 변경을 붙여 라벨링. 데이터 기간 안의 회의만 남긴다."""
    b = base.dropna()
    changes = detect_changes(b)
    mtg = pd.to_datetime(pd.read_csv(MEETINGS_CSV)["date"])
    mtg = mtg[(mtg >= b.index.min()) & (mtg <= b.index.max())]

    rows = []
    for dt in mtg:
        hit = changes[(changes["date"] >= dt)
                      & (changes["date"] <= dt + pd.Timedelta(days=EFFECT_LAG_DAYS))]
        ch = float(hit["change_bp"].sum())
        rows.append({"meeting_date": dt, "change_bp": ch,
                     "rate": float(b.loc[:dt + pd.Timedelta(days=EFFECT_LAG_DAYS)].iloc[-1])})

    df = pd.DataFrame(rows)
    # 모든 변경이 어느 회의에 붙었는지 확인 (캘린더 누락 탐지)
    missed = changes[~changes["date"].isin(
        [c for dt in mtg for c in changes["date"]
         if dt <= c <= dt + pd.Timedelta(days=EFFECT_LAG_DAYS)])]
    assert missed.empty, f"회의에 매칭되지 않은 기준금리 변경: {missed['date'].dt.date.tolist()}"

    df["label"] = np.select(
        [df["change_bp"] > 0, df["change_bp"] < 0], ["인상", "인하"], default="동결"
    )
    return df.reset_index(drop=True)


if __name__ == "__main__":
    panel = pd.read_parquet(PROC / "panel.parquet")
    base = panel["base_rate_pct"]

    changes = detect_changes(base)
    print(f"기준금리 변경 이벤트: {len(changes)}건 "
          f"({changes['date'].min():%Y-%m-%d} ~ {changes['date'].max():%Y-%m-%d})")
    print(changes.tail(10).to_string(index=False))

    meetings = build_meetings(base)
    print(f"\n회의 캘린더(근사): {len(meetings)}회")
    print(meetings["label"].value_counts().to_string())
    print(f"\n동결 비중 {(meetings['label'] == '동결').mean():.1%} "
          f"<- '항상 동결' 기준선의 정확도")
    print("\n최근 12회:")
    print(meetings.tail(12).to_string(index=False))

    meetings.to_parquet(PROC / "policy_meetings.parquet", index=False)
    print(f"\n저장: {PROC / 'policy_meetings.parquet'}")
