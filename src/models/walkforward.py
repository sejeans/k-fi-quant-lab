"""expanding window walk-forward 공통 함수 (curve_forecast, slope_forecast, credit_model).

- 학습은 정답이 확정된 행만 쓴다. 타깃이 'h영업일 뒤 변화'라서 i시점에는 i-h행까지만
  정답을 안다. 이전 버전은 i-1행까지 학습해 마지막 h행의 미래 정보가 섞였다(purge 누락).
- 정답이 아직 없는 최신 행도 예측한다(actual=NaN). 평가에는 split_live로 떼어낸 뒤 쓰고,
  떼어낸 행은 대시보드의 실시간 예측으로 저장한다.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

MIN_TRAIN = 750           # 약 3년치 최소 학습표본
REFIT_EVERY = 20          # 20영업일마다 재학습


def walk_forward(X: pd.DataFrame, y: pd.Series, split: pd.Timestamp, h: int,
                 make_model: Callable, proba: bool = False) -> pd.DataFrame:
    """split 이후 각 시점 i에 대해, 정답이 확정된 행(i-h까지)으로 학습한 모델의 예측을 낸다."""
    df = X.join(y.rename("y")).dropna(subset=list(X.columns))
    Xv, yv, idx = df[X.columns].to_numpy(float), df["y"].to_numpy(float), df.index
    pos = np.where(idx >= split)[0]
    pos = pos[pos >= MIN_TRAIN]

    pred = np.full(len(pos), np.nan)
    model = sc = None
    for k, i in enumerate(pos):
        if k % REFIT_EVERY == 0:
            tr = np.arange(i - h + 1)                 # 정답이 i시점까지 확정된 행
            tr = tr[~np.isnan(yv[tr])]
            if not (proba and len(np.unique(yv[tr])) < 2):
                sc = StandardScaler().fit(Xv[tr])
                model = make_model().fit(sc.transform(Xv[tr]), yv[tr])
        if model is None:
            continue
        x = sc.transform(Xv[i:i + 1])
        pred[k] = model.predict_proba(x)[0, 1] if proba else model.predict(x)[0]

    return pd.DataFrame({"actual": yv[pos], "pred": pred}, index=idx[pos])


def split_live(res: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(정답이 있는 평가용 행, 정답이 아직 없는 최신 예측 행)으로 나눈다."""
    live = res["actual"].isna()
    return res[~live], res[live]


if __name__ == "__main__":
    # 자기점검: i시점 학습에 쓰인 행의 정답이 i 이후 정보를 쓰지 않는지 확인
    n, h = 900, 20
    idx = pd.bdate_range("2020-01-01", periods=n)
    s = pd.Series(np.arange(n, dtype=float), index=idx)
    X = pd.DataFrame({"x": s})
    y = s.shift(-h) - s                               # 마지막 h행은 NaN
    seen = []

    class Spy:
        def fit(self, Xt, yt):
            seen.append(len(yt)); return self

        def predict(self, x):
            return np.zeros(1)

    r = walk_forward(X, y, idx[MIN_TRAIN], h, Spy)
    assert seen[0] == MIN_TRAIN - h + 1, seen[0]      # i=MIN_TRAIN에서 i-h행까지만 학습
    ev, live = split_live(r)
    assert len(live) == h and live["pred"].notna().all()
    print("walkforward 자기점검 통과")
