"""기준금리 일별 시계열 -> 금통위 이벤트·라벨 생성.

근거 (사전 확인)
- ECOS 722Y001 기준금리는 일별 계단함수다. 값이 바뀌는 날 = 정책 변경 발효일.
- 따라서 변화점 탐지만으로 '결정일 + 방향(인상/동결/인하) + 변화폭(bp)'을 얻는다.
  한국은행 결정문 PDF/HWP 파싱 없이 라벨링이 끝난다.

한계 (반드시 명시)
- 이 방식은 '변경이 있었던 회의'만 직접 관측한다. 동결 회의는 달력에서 따로 구성해야 한다.
- 금통위 통방회의는 연 8회(2017년 이후)지만 그 이전에는 매월이었다.
  실제 회의일 목록을 하드코딩하지 않고, '변경일 + 정기 회의 주기'로 근사한다.
  -> 근사이므로 결과 해석 시 이 점을 밝힌다.
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


def build_meetings(base: pd.Series) -> pd.DataFrame:
    """금통위 회의 캘린더 근사 + 라벨.

    구성 규칙
    - 변경일은 실제 회의일이다(그날 결정이 발효).
    - 동결 회의는 변경일 사이를 메우되, 2017년 이후 연 8회 / 이전 연 12회 주기를 가정한다.
      매월 둘째주 목요일 근처를 회의일로 근사한다.
    """
    b = base.dropna()
    changes = detect_changes(b)
    change_dates = set(changes["date"])

    # 정기 회의 근사: 각 월의 둘째주 목요일
    months = pd.date_range(b.index.min(), b.index.max(), freq="MS")
    cand = []
    for m in months:
        thursdays = pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="W-THU")
        if len(thursdays) >= 2:
            cand.append(thursdays[1])
    cand = pd.DatetimeIndex(cand)

    # 2017년 이후는 연 8회: 1,2,4,5,7,8,10,11월만 남긴다(실제 개최월과 근사)
    keep = [d for d in cand if d.year < 2017 or d.month in (1, 2, 4, 5, 7, 8, 10, 11)]
    meetings = sorted(set(keep) | change_dates)

    rows = []
    for dt in meetings:
        if dt not in b.index:
            # 거래일이 아니면 직전 거래일로 스냅
            prior = b.index[b.index <= dt]
            if len(prior) == 0:
                continue
            dt_use = prior[-1]
        else:
            dt_use = dt

        if dt in change_dates:
            ch = float(changes.loc[changes["date"] == dt, "change_bp"].iloc[0])
        else:
            ch = 0.0
        rows.append({"meeting_date": dt_use, "change_bp": ch,
                     "rate": float(b.loc[dt_use])})

    df = pd.DataFrame(rows).drop_duplicates("meeting_date").sort_values("meeting_date")
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
