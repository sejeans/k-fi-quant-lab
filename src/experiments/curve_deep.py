"""딥러닝(LSTM / 1D-CNN / MLP)으로 NS 요인 20일 변화 예측. 평가 설계는 curve_ideas와 동일.

- 입력: 최근 SEQ영업일의 기본특징(make_features) 시퀀스. 학습구간 통계로만 표준화.
- 출력: 3요인 20일 변화 동시 예측(다중출력), 타깃도 학습구간 std로 스케일.
- purge: i시점 재학습에는 정답이 확정된 행(i-H까지)만. 마지막 15%를 검증으로 떼어 조기종료.
- 시드 3개 평균. REFIT일마다 재학습.
"""
import sys as _s
if hasattr(_s.stdout, "reconfigure"): _s.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
import torch, torch.nn as nn
from src.experiments.curve_ideas import *   # noqa
torch.set_num_threads(8)
SEQ, REFIT, SEEDS, EPOCHS = 40, 60, 3, 60

feat = X0.copy()
Y = pd.concat({f: fac[f].shift(-H) - fac[f] for f in FACTORS}, axis=1)
ok = feat.notna().all(axis=1)
feat, Y = feat[ok], Y[ok]
Xv, Yv, idx = feat.to_numpy(np.float32), Y.to_numpy(np.float32), feat.index

class LSTM(nn.Module):
    def __init__(s, k): super().__init__(); s.r = nn.LSTM(k, 32, batch_first=True); s.d = nn.Dropout(.3); s.o = nn.Linear(32, 3)
    def forward(s, x): h, _ = s.r(x); return s.o(s.d(h[:, -1]))
class CNN(nn.Module):
    def __init__(s, k):
        super().__init__()
        s.c = nn.Sequential(nn.Conv1d(k, 32, 5, padding=2), nn.ReLU(), nn.Conv1d(32, 32, 5, dilation=2, padding=4), nn.ReLU(),
                            nn.AdaptiveAvgPool1d(1))
        s.d = nn.Dropout(.3); s.o = nn.Linear(32, 3)
    def forward(s, x): return s.o(s.d(s.c(x.transpose(1, 2)).squeeze(-1)))
class MLP(nn.Module):
    def __init__(s, k): super().__init__(); s.n = nn.Sequential(nn.Linear(k, 64), nn.ReLU(), nn.Dropout(.3), nn.Linear(64, 3))
    def forward(s, x): return s.n(x[:, -1])

def seqs(Xs, rows):
    return np.stack([Xs[r - SEQ + 1:r + 1] for r in rows])

def fit_predict(Net, tr, te_rows, Xs, Ys, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    nv = int(len(tr) * .15); va, trn = tr[-nv:], tr[:-nv - H]          # 검증 앞에도 H행 간격(purge)
    xt, yt = torch.tensor(seqs(Xs, trn)), torch.tensor(Ys[trn])
    xv, yv = torch.tensor(seqs(Xs, va)), torch.tensor(Ys[va])
    net = Net(Xs.shape[1]); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    best, state, bad = 1e9, None, 0
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(xt))
        for b in range(0, len(xt), 128):
            j = perm[b:b + 128]; opt.zero_grad(); nn.functional.mse_loss(net(xt[j]), yt[j]).backward(); opt.step()
        net.eval()
        with torch.no_grad(): v = nn.functional.mse_loss(net(xv), yv).item()
        if v < best - 1e-4: best, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8: break
    net.load_state_dict(state); net.eval()
    with torch.no_grad(): return net(torch.tensor(seqs(Xs, te_rows))).numpy()

def run_deep(Net):
    pos = np.where(idx >= DEV0)[0]; pos = pos[pos >= wf.MIN_TRAIN]
    pred = np.full((len(pos), 3), np.nan)
    for k0 in range(0, len(pos), REFIT):
        i = pos[k0]; block = pos[k0:k0 + REFIT]
        tr = np.arange(SEQ, i - H + 1); tr = tr[~np.isnan(Yv[tr]).any(axis=1)]
        mu, sd = Xv[:i - H + 1].mean(0), Xv[:i - H + 1].std(0) + 1e-6
        ysd = Yv[tr].std(0)
        Xs, Ys = (Xv - mu) / sd, Yv / ysd
        Ys = np.nan_to_num(Ys)
        p = np.mean([fit_predict(Net, tr, block, Xs.astype(np.float32), Ys.astype(np.float32), s) for s in range(SEEDS)], axis=0)
        pred[k0:k0 + len(block)] = p * ysd
    out = {}
    for j, f in enumerate(FACTORS):
        r = pd.DataFrame({"actual": Yv[pos, j], "pred": pred[:, j]}, index=idx[pos])
        out[f] = r[r["actual"].notna()]
    return out

if __name__ == "__main__":
    rows = []
    P = {"dev": (DEV0, SEAL - pd.Timedelta(days=1)), "sealed": (SEAL, fac.index.max())}
    for name, Net in {"LSTM": LSTM, "1D-CNN": CNN, "MLP": MLP}.items():
        t = time.time(); res = run_deep(Net)
        for tag, shr in {"": (1, 1, 1), " +기울기만": (0, 1, 0)}.items():
            r2 = {f: res[f].assign(pred=res[f]["pred"] * shr[i]) for i, f in enumerate(FACTORS)}
            for per, (lo, hi) in P.items():
                rows.append({"idea": name + tag, "period": per, **evaluate(r2, lo, hi)})
        print(f"{name} {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(rows); df.to_csv(PROC / "curve_deep.csv", index=False)
    pd.set_option("display.width", 200)
    for per in P:
        print(f"\n== {per} =="); print(df[df.period == per].drop(columns="period").round(2).to_string(index=False))
