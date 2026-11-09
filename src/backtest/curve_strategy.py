"""모듈 1 운용 연결: 커브 전략 백테스트 (버터플라이 중심).

전략 선택 근거 (진단 결과)
- 3s10s 슬로프는 이 표본에서 예측력이 없다(corr +0.040, 방향 47.9%). 따라서 매매하지 않는다.
- 3-5-10 버터플라이는 RW를 이긴다(corr +0.438, 방향 63.3%, RW 대비 RMSE +10.1%).
  단, 겹치지 않는 20일 구간 기준 24개 중 14개 적중(p=0.27)이라 유의하지 않다.
  NS 곡률 요인은 2.2년 만기에서 가장 크게 움직여 3-5-10 버터플라이와 직접 대응하지 않는다.
- 두 전략을 모두 백테스트해 '예측력 있는 쪽만 거래한다'는 판단 과정을 남긴다.

리스크 중립화 (기획서 3.5)
- 버터플라이: 몸통(5년)과 날개(3년/10년)의 DV01을 맞춘다. 50:50 배분으로
  날개 합계 DV01 = 몸통 DV01. 금리 수준(level) 노출을 제거하고 곡률만 취한다.
- 손익: dP/P = -D*dy + 0.5*C*dy^2 (볼록성 2차항 포함)
- 거래비용 1/2/3배 스트레스, 임계치 이하 무거래.
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

H = 20
COST_BP_PER_LEG = 0.5      # 편도 비용 가정(bp of yield)
NOTIONAL = 100.0


def bond_risk(tenor: float, ytm_pct: float) -> tuple[float, float]:
    """par 채권(연 1회 이표) 수정듀레이션·볼록성."""
    y = ytm_pct / 100.0
    n = int(round(tenor))
    t = np.arange(1, n + 1, dtype=float)
    cf = np.full(n, y * 100.0)
    cf[-1] += 100.0
    pv = cf * (1.0 + y) ** (-t)
    price = pv.sum()
    mod = ((t * pv).sum() / price) / (1.0 + y)
    conv = ((t * (t + 1.0) * pv).sum() / price) / (1.0 + y) ** 2
    return mod, conv


def leg_return(mod: float, conv: float, dy_bp: float) -> float:
    dy = dy_bp / 10000.0
    return (-mod * dy + 0.5 * conv * dy**2) * 100.0


def backtest_butterfly(sig: pd.DataFrame, panel: pd.DataFrame,
                       thresh_bp: pd.Series, cost_mult: float) -> dict:
    """버터플라이 = 2*5년 - 3년 - 10년. 예측 상승(곡률 확대) 시 몸통 숏/날개 롱.

    - 매일 신규 진입분은 자본의 1/H. H영업일 보유 후 청산(H개 트랜치 중첩).
    - 손익은 일별 평가손익(MTM)으로 쌓는다. 이전 버전은 20일 보유손익을 진입일 하루에
      몰아 넣고 일간 수익률처럼 연율화해 변동성을 과소, Sharpe 크기를 과대 계산했다.
    - thresh_bp는 시점별 임계치(그 날까지의 신호만으로 계산). NaN이면 거래하지 않는다.
    """
    full = panel.index
    y3, y5, y10 = (panel[c].ffill().to_numpy() for c in ("ktb3y_pct", "ktb5y_pct", "ktb10y_pct"))
    pos = full.get_indexer(sig.index)
    pred, th = sig["enet"].to_numpy(), thresh_bp.reindex(sig.index).to_numpy()

    daily = np.zeros(len(full))                      # 일별 포트폴리오 수익(%)
    trade_pnl = []
    cost_bp = COST_BP_PER_LEG * cost_mult            # 편도
    for k, i in enumerate(pos):
        s = pred[k]
        if not (np.isfinite(s) and np.isfinite(th[k])) or abs(s) < th[k]:
            continue
        j = i + H
        if j >= len(full):                           # 청산일이 데이터 밖이면 진입하지 않는다
            continue

        m3, c3 = bond_risk(3.0, y3[i])
        m5, c5 = bond_risk(5.0, y5[i])
        m10, c10 = bond_risk(10.0, y10[i])
        # DV01 중립: 몸통 5년 명목 = NOTIONAL, 날개는 각각 절반의 DV01을 담당
        n5 = NOTIONAL
        n3, n10 = 0.5 * n5 * m5 / m3, 0.5 * n5 * m5 / m10
        # 버터플라이 상승 예상(s>0) = 몸통 금리가 날개 대비 상승 -> 몸통 숏 / 날개 롱
        side = -1.0 if s > 0 else 1.0                # side=+1: 몸통 롱, 날개 숏

        leg_cost = (n3 * m3 + n5 * m5 + n10 * m10) / NOTIONAL * (cost_bp / 10000.0) * 100.0
        daily[i] -= leg_cost / H                     # 진입 비용
        daily[j] -= leg_cost / H                     # 청산 비용
        tot = -2 * leg_cost
        for t in range(i + 1, j + 1):
            d3, d5, d10 = ((y[t] - y[t - 1]) * 100 for y in (y3, y5, y10))
            pnl = side * (n5 * leg_return(m5, c5, d5) - n3 * leg_return(m3, c3, d3)
                          - n10 * leg_return(m10, c10, d10)) / NOTIONAL
            daily[t] += pnl / H
            tot += pnl
        trade_pnl.append(tot)

    lo, hi = pos.min(), pos.max() + H                # 평가 구간: 첫 신호일 ~ 마지막 청산일
    r = pd.Series(daily[lo:hi + 1], index=full[lo:hi + 1])
    ann = 252
    mu, sd = r.mean() * ann, r.std() * np.sqrt(ann)
    cum = (1 + r / 100).cumprod()
    mdd = ((cum / cum.cummax()) - 1).min() * 100
    tp = np.array(trade_pnl)
    return {
        "thresh_bp": float(np.nanmean(th)), "cost_x": cost_mult, "n_trades": len(tp),
        "ann_ret_%": mu, "ann_vol_%": sd, "sharpe": mu / sd if sd > 0 else np.nan,
        "mdd_%": mdd, "hit_%": (tp > 0).mean() * 100 if len(tp) else np.nan,
        "_curve": cum,
    }


if __name__ == "__main__":
    panel = pd.read_parquet(PROC / "panel.parquet")

    print("=" * 62)
    print("전략 후보 예측력 점검 (거래 여부 판단)")
    print("=" * 62)
    keep = {}
    for tgt, label in [("slope_3s10s_bp", "3s10s 슬로프"), ("bfly_3s5s10s_bp", "3-5-10 버터플라이")]:
        f = PROC / f"fc_direct_{tgt}_h{H}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        corr = d["enet"].corr(d["actual"])
        hit = (np.sign(d["enet"]) == np.sign(d["actual"])).mean() * 100
        rw = np.sqrt((d["actual"] ** 2).mean())
        md = np.sqrt(((d["actual"] - d["enet"]) ** 2).mean())
        imp = (rw - md) / rw * 100
        ok = imp > 0 and hit > 50
        print(f"  {label:<18} corr={corr:+.3f}  방향={hit:5.1f}%  RW대비={imp:+6.2f}%  "
              f"-> {'거래' if ok else '거래 안 함'}")
        if ok:
            keep[tgt] = (label, d)

    if not keep:
        print("\n예측력이 확인된 전략이 없습니다. 백테스트를 중단합니다.")
        raise SystemExit(0)

    for tgt, (label, sig) in keep.items():
        print(f"\n{'=' * 62}\n{label} 백테스트  (n={len(sig)}, "
              f"{sig.index.min():%Y-%m-%d} ~ {sig.index.max():%Y-%m-%d})\n{'=' * 62}")
        print("임계치는 그 날까지 나온 신호 절대값의 분위수로 정한다(예측 std가 실제보다 작아 고정bp는 부적절).")
        print(f"{'분위':>6}{'임계bp':>8}{'비용x':>7}{'거래수':>8}{'연수익%':>9}{'연변동%':>9}"
              f"{'Sharpe':>8}{'MDD%':>8}{'적중%':>7}")
        rows, curves = [], {}
        qs = [0.0, 0.5, 0.7, 0.8, 0.9]
        for q in qs:
            # 임계치는 그 날까지 나온 신호의 분위수(확장 윈도우). 최소 H개 쌓인 뒤부터 거래
            th = (sig["enet"].abs().expanding(min_periods=H).quantile(q) if q > 0
                  else pd.Series(0.0, index=sig.index))
            for cx in [1.0, 2.0, 3.0]:
                r = backtest_butterfly(sig, panel, th, cx)
                r["signal_q"] = q
                curves[(q, cx)] = r.pop("_curve")
                rows.append(r)
                print(f"{q:>6.1f}{r['thresh_bp']:>8.2f}{r['cost_x']:>7.0f}{r['n_trades']:>8d}"
                      f"{r['ann_ret_%']:>9.2f}{r['ann_vol_%']:>9.2f}{r['sharpe']:>8.2f}"
                      f"{r['mdd_%']:>8.2f}{r['hit_%']:>7.1f}")
        pd.DataFrame(rows).to_csv(PROC / f"backtest_{tgt}.csv", index=False)
        curves[(0.9, 1.0)].rename("cum").to_frame().to_parquet(PROC / f"equity_{tgt}.parquet")

    print(f"\n저장: {PROC}")
