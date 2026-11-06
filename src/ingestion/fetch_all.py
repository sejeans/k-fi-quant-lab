"""configs/series.yaml의 전 시리즈를 수집해 data/raw에 저장한다.

캐시 원칙: raw에 파일이 있으면 재호출하지 않는다(--force로 무시).
"""

from __future__ import annotations

import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):      # Windows cp949 콘솔에서 한글/기호 깨짐 방지
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ingestion.ecos import EcosError, fetch_series, get_key  # noqa: E402

RAW = ROOT / "data" / "raw"
CFG = ROOT / "configs" / "series.yaml"


def load_config() -> dict:
    return yaml.safe_load(CFG.read_text(encoding="utf-8"))


def main(force: bool = False) -> int:
    cfg = load_config()
    start = cfg["meta"]["start_date"]
    end = date.today().strftime("%Y%m%d")
    key = get_key()
    RAW.mkdir(parents=True, exist_ok=True)

    print(f"수집 기간 {start} ~ {end}\n")
    ok, failed = [], []

    for s in cfg["series"]:
        name = s["variable_name"]
        out = RAW / f"{name}.parquet"

        if out.exists() and not force:
            df = pd.read_parquet(out)
            print(f"  cache {name:<16} {len(df):>5}행")
            ok.append(name)
            continue

        try:
            df = fetch_series(s["stat_code"], s["item_code"], start, end, key=key)
            if df.empty:
                raise EcosError("빈 응답")
            df.to_parquet(out, index=False)
            print(
                f"  OK    {name:<16} {len(df):>5}행  "
                f"{df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}  "
                f"last={df['value'].iloc[-1]}"
            )
            ok.append(name)
        except Exception as exc:
            # 폴백(계획 Block 1): 실패 시리즈는 건너뛰고 나머지로 진행
            print(f"  FAIL  {name:<16} {exc}")
            failed.append(name)

    print(f"\n성공 {len(ok)} / 실패 {len(failed)}")
    if failed:
        print("실패 시리즈:", ", ".join(failed))
    return 0 if len(ok) >= 10 else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="캐시 무시하고 재수집")
    raise SystemExit(main(p.parse_args().force))
