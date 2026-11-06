"""한국은행 ECOS 오픈API 클라이언트.

설계 메모
- 실키는 1회 호출 상한이 넉넉해 15년치 일별 시리즈를 한 번에 받는다(실측 4,123행/1.8초).
  그래도 상한을 모르는 통계표가 있으므로 페이징을 남겨둔다.
- 받은 원본은 data/raw에 그대로 저장하고 다시 호출하지 않는다(기획서 2.2: 원천/가공 분리).
- ECOS는 에러도 HTTP 200으로 주고 body의 최상위 키가 RESULT가 된다. 이 경우를 예외로 승격한다.
"""

from __future__ import annotations

import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):      # Windows cp949 콘솔에서 한글/기호 깨짐 방지
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://ecos.bok.or.kr/api"
PAGE = 10000          # 1회 호출 요청 행수
MAX_RETRY = 3
TIMEOUT = 90


class EcosError(RuntimeError):
    """ECOS가 RESULT 코드로 돌려준 오류."""


def get_key() -> str:
    """.env 또는 환경변수에서 인증키를 읽는다."""
    key = os.environ.get("ECOS_API_KEY")
    if key:
        return key
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("ECOS_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise EcosError("ECOS_API_KEY를 찾을 수 없습니다. .env를 확인하세요.")


def _call(url: str) -> dict:
    last = None
    for attempt in range(MAX_RETRY):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as exc:                     # 네트워크/파싱 실패는 재시도
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise EcosError(f"호출 실패: {last}")


def fetch_series(
    stat_code: str,
    item_code: str,
    start: str,
    end: str,
    cycle: str = "D",
    key: str | None = None,
) -> pd.DataFrame:
    """단일 시리즈를 DataFrame(date, value)으로 반환한다."""
    key = key or get_key()
    rows: list[dict] = []
    start_row = 1

    while True:
        url = (
            f"{BASE}/StatisticSearch/{key}/json/kr/"
            f"{start_row}/{start_row + PAGE - 1}/"
            f"{stat_code}/{cycle}/{start}/{end}/{item_code}"
        )
        payload = _call(url)

        if "RESULT" in payload:                      # ECOS 오류 응답
            res = payload["RESULT"]
            code = res.get("CODE", "")
            if code == "INFO-200" and rows:          # 더 받을 데이터 없음
                break
            raise EcosError(f"{stat_code}/{item_code} {code}: {res.get('MESSAGE','')}")

        block = payload.get("StatisticSearch", {})
        got = block.get("row", [])
        rows.extend(got)
        total = int(block.get("list_total_count", len(rows)))

        if len(rows) >= total or not got:
            break
        start_row += PAGE

    if not rows:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(rows)[["TIME", "DATA_VALUE"]]
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    # ECOS는 결측을 빈 문자열로 준다 -> NaN으로 승격
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
