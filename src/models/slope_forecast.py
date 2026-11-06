"""3s10s 슬로프 변화를 '직접' 예측한다.

왜 직접 예측하는가 (진단 결과)
- NS 요인은 직교가 아니다. 20일 변화 기준 corr(b2,b3)=-0.657, corr(b1,b2)=-0.652.
- 요인별로 따로 예측한 뒤 재결합하면 요인 간 예측오차가 상호작용해 신호가 뒤집힌다.
  실측: corr(pred_b2, act_b3) = -0.411 이고 3s10s를 실제로 움직이는 건 b3(corr -0.702)라
  결합 신호의 상관이 -0.239로 역전됐다.
- 따라서 '거래하는 대상'인 슬로프를 타깃으로 직접 학습한다.
  요인 분해는 커브 해석/시각화용으로 유지하고, 매매신호는 여기서 만든다.

검증은 curve_forecast.py와 동일: expanding walk-forward, 최근 2년 봉인, shuffle 없음.
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
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PROC = ROOT / "data" / "processed"

H = 20
MIN_TRAIN = 750
REFIT_EVERY = 20
HOLDOUT_YEARS = 2
SEED = 42

TARGETS = {
    "slope_3s10s_bp": "3s10s 슬로프",
    "bfly_3s5s10s_bp": "3-5-10 버터플라이",
}


def make_features(panel: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """t시점까지의 정보만. 커브 자체 + 정책·유동성 변수."""
    X = pd.DataFrame(index=panel.index)

    for c in ["slope_3s10s_bp", "slope_1s10s_bp", "bfly_3s5s10s_bp"]:
        s = panel[c]
        X[c] = s
        X[f"{c}_d5"] = s.diff(5)
        X[f"{c}_d20"] = s.diff(20)
        # 평균회귀 신호: 60일 z-score
        X[f"{c}_z60"] = (s - s.rolling(60).mean()) / s.rolling(60).std()
        X[f"{c}_z250"] = (s - s.rolling(250).mean()) / s.rolling(250).std()

    X["level_3y"] = panel["ktb3y_pct"]
    X["ktb1y_base_gap_bp"] = panel["ktb1y_base_gap_bp"]
    X["cd_msb_bp"] = panel["cd_msb_bp"]
    X["cp_cd_bp"] = panel["cp_cd_bp"]
    X["cs_aa3y_bp"] = panel["cs_aa3y_bp"]
    X["vol20"] = panel["ktb3y_vol20_bp"]
    X["curv_ns"] = factors["beta3_curv"].reindex(panel.index)
    return X


def walk_forward(X: pd.DataFrame, y: pd.Series, split: pd.Timestamp) -> pd.DataFrame:
    df = X.join(y.rename("y")).dropna()
    Xv, yv, idx = df.drop(columns="y").to_numpy(), df["y"].to_numpy(), df.index
    pos = np.where(idx >= split)[0]
    pos = pos[pos >= MIN_TRAIN]
    if len(pos) == 0:
        return pd.DataFrame()

    pred = np.full(len(pos), np.nan)
    model = scaler = None
    for k, i in enumerate(pos):
        if k % REFIT_EVERY == 0:
            scaler = StandardScaler().fit(Xv[:i])
            model = ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=SEED,
                max_iter=5000, n_jobs=-1,
            ).fit(scaler.transform(Xv[:i]), yv[:i])
        pred[k] = model.predict(scaler.transform(Xv[i : i + 1]))[0]

    return pd.DataFrame({"actual": yv[pos], "rw": 0.0, "enet": pred}, index=idx[pos])


if __name__ == "__main__":
    np.random.seed(SEED)
    panel = pd.read_parquet(PROC / "panel.parquet")
    factors = pd.read_parquet(PROC / "ns_factors.parquet")
    X = make_features(panel, factors)
    split = panel.index.max() - pd.DateOffset(years=HOLDOUT_YEARS)

    print(f"최종 테스트 구간(봉인): {split:%Y-%m-%d} ~ {panel.index.max():%Y-%m-%d}")
    print(f"타깃: {H}영업일 후 변화량 (bp)\n")

    for tgt, label in TARGETS.items():
        y = panel[tgt].shift(-H) - panel[tgt]
        res = walk_forward(X, y, split)
        if res.empty:
            print(f"[{label}] 표본 부족")
            continue

        rw_rmse = np.sqrt((res["actual"] ** 2).mean())
        md_rmse = np.sqrt(((res["actual"] - res["enet"]) ** 2).mean())
        corr = res["enet"].corr(res["actual"])
        hit = (np.sign(res["enet"]) == np.sign(res["actual"])).mean() * 100

        print(f"[{label}]  n={len(res)}")
        print(f"  RW RMSE      {rw_rmse:7.2f}bp")
        print(f"  모델 RMSE    {md_rmse:7.2f}bp   (개선 {(rw_rmse-md_rmse)/rw_rmse*100:+.2f}%)")
        print(f"  corr         {corr:+7.3f}")
        print(f"  방향정확도   {hit:7.1f}%")
        print(f"  예측std      {res['enet'].std():7.2f}bp (실제 {res['actual'].std():.2f}bp)\n")

        res.to_parquet(PROC / f"fc_direct_{tgt}_h{H}.parquet")

    print(f"저장: {PROC}")
