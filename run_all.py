"""전체 파이프라인 1커맨드 실행.

    python run_all.py            # 캐시 사용
    python run_all.py --force    # ECOS 재수집

각 단계는 독립 실행 가능한 모듈이며, 실패 시 이후 단계를 중단한다.
"""

from __future__ import annotations

import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):      # Windows cp949 콘솔에서 한글/기호 깨짐 방지
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("데이터 수집 (ECOS)", "src/ingestion/fetch_all.py"),
    ("패널 구축", "src/features/panel.py"),
    ("Nelson-Siegel 적합 + PCA 검증", "src/features/nelson_siegel.py"),
    ("모듈1: 요인 예측 (walk-forward)", "src/models/curve_forecast.py"),
    ("모듈1: 슬로프/버터플라이 직접 예측", "src/models/slope_forecast.py"),
    ("모듈1: 커브 전략 백테스트", "src/backtest/curve_strategy.py"),
    ("모듈4: 금통위 이벤트 추출", "src/features/policy_events.py"),
    ("모듈4: 금통위 확률 모델", "src/models/policy_prob.py"),
    ("모듈3: 크레딧 스프레드", "src/models/credit_model.py"),
    ("차트 생성", "src/reporting/figures.py"),
    ("리서치 노트 생성", "src/reporting/note.py"),
]


def main(force: bool) -> int:
    print("=" * 64)
    print("K-FI Quant Lab — 전체 파이프라인")
    print("=" * 64)
    t0 = time.time()

    for i, (label, script) in enumerate(STEPS, 1):
        print(f"\n[{i}/{len(STEPS)}] {label}")
        print("-" * 64)
        cmd = [sys.executable, str(ROOT / script)]
        if force and "fetch_all" in script:
            cmd.append("--force")
        t = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"\n실패: {script} (exit {r.returncode})")
            return r.returncode
        print(f"  ... {time.time()-t:.1f}초")

    print("\n" + "=" * 64)
    print(f"완료 — 총 {time.time()-t0:.1f}초")
    print("=" * 64)
    print("\n결과물:")
    print("  reports/research_note.md   리서치 노트 (자동 생성)")
    print("  reports/figures/*.png      차트 6종")
    print("  data/processed/*           모델 산출물")
    print("\n대시보드: streamlit run app/dashboard.py")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="ECOS 데이터 재수집")
    raise SystemExit(main(p.parse_args().force))
