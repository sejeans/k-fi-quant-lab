"""raw 시리즈를 일별 패널로 결합한다.

정합 규칙 (기획서 2.2)
- 영업일 축은 '국고채 3년이 관측된 날' = 실제 채권시장 개장일로 정의한다.
  달력 영업일(B)을 쓰면 국내 공휴일이 결측으로 남는다.
- 금리 결측은 최대 5영업일 ffill. 그 이상 비면 NaN으로 남겨 모델에서 제외한다.
- 기준금리는 정의상 다음 변경 전까지 유지되므로 무제한 ffill.
- 파생 스프레드는 bp 단위로 만들고 변수명에 _bp를 붙인다.
"""

from __future__ import annotations

import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):      # Windows cp949 콘솔에서 한글/기호 깨짐 방지
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
CFG = ROOT / "configs" / "series.yaml"

FFILL_LIMIT = 5


def build_panel() -> pd.DataFrame:
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    frames = {}
    for s in cfg["series"]:
        f = RAW / f"{s['variable_name']}.parquet"
        if f.exists():
            frames[s["variable_name"]] = pd.read_parquet(f).set_index("date")["value"]

    if "ktb3y_pct" not in frames:
        raise RuntimeError("ktb3y_pct가 없습니다. 커브 기준축을 만들 수 없습니다.")

    # 거래일 축 = 국고채 3년 관측일
    trading_days = frames["ktb3y_pct"].dropna().index
    panel = pd.DataFrame(index=trading_days).sort_index()
    panel.index.name = "date"

    for name, ser in frames.items():
        col = ser.reindex(panel.index)
        limit = None if name == "base_rate_pct" else FFILL_LIMIT
        panel[name] = col.ffill(limit=limit)

    # ---- 파생변수 (단위를 이름에 명시) ----
    # 커브 형태
    panel["slope_3s10s_bp"] = (panel["ktb10y_pct"] - panel["ktb3y_pct"]) * 100
    panel["slope_1s10s_bp"] = (panel["ktb10y_pct"] - panel["ktb1y_pct"]) * 100
    panel["bfly_3s5s10s_bp"] = (
        2 * panel["ktb5y_pct"] - panel["ktb3y_pct"] - panel["ktb10y_pct"]
    ) * 100

    # 크레딧 스프레드 (모듈 3 타깃)
    panel["cs_aa3y_bp"] = (panel["corp_aa3y_pct"] - panel["ktb3y_pct"]) * 100
    panel["cs_bbb3y_bp"] = (panel["corp_bbb3y_pct"] - panel["ktb3y_pct"]) * 100

    # 유동성·단기 스프레드
    panel["cd_msb_bp"] = (panel["cd91_pct"] - panel["msb1y_pct"]) * 100
    panel["cp_cd_bp"] = (panel["cp91_pct"] - panel["cd91_pct"]) * 100

    # 정책 관련 (모듈 4 입력)
    panel["ktb1y_base_gap_bp"] = (panel["ktb1y_pct"] - panel["base_rate_pct"]) * 100
    panel["ktb3y_base_gap_bp"] = (panel["ktb3y_pct"] - panel["base_rate_pct"]) * 100
    panel["call_base_gap_bp"] = (panel["call_on_pct"] - panel["base_rate_pct"]) * 100

    # 변동성 (20일 실현, bp 단위 일간변화 기준)
    panel["ktb3y_vol20_bp"] = (
        panel["ktb3y_pct"].diff().mul(100).rolling(20).std()
    )
    return panel


def report(panel: pd.DataFrame) -> None:
    print(f"패널 {panel.shape[0]}행 x {panel.shape[1]}열")
    print(f"기간 {panel.index.min():%Y-%m-%d} ~ {panel.index.max():%Y-%m-%d}\n")
    print(f"{'변수':<22}{'결측률':>8}  {'최신값':>10}")
    print("-" * 44)
    for c in panel.columns:
        miss = panel[c].isna().mean() * 100
        last = panel[c].dropna()
        lastv = f"{last.iloc[-1]:.3f}" if len(last) else "N/A"
        print(f"{c:<22}{miss:>7.2f}%  {lastv:>10}")

    # 완료 기준: 국고채 3/5/10년 결측률 5% 미만
    print()
    for c in ["ktb3y_pct", "ktb5y_pct", "ktb10y_pct"]:
        miss = panel[c].isna().mean() * 100
        flag = "PASS" if miss < 5 else "FAIL"
        print(f"[{flag}] {c} 결측률 {miss:.2f}% (기준 <5%)")


if __name__ == "__main__":
    PROC.mkdir(parents=True, exist_ok=True)
    p = build_panel()
    p.to_parquet(PROC / "panel.parquet")
    report(p)
    print(f"\n저장: {PROC / 'panel.parquet'}")
