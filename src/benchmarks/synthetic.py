"""
5.2  Synthetic ground-truth tests: an independent check of "does the explainer find the right features".

(a) CORRUPTION TEST (autoencoder family). Take real normal rows the model never trained on, multiply 3
    known features by a fixed factor (10x per the roadmap; 100x also reported), and ask each explainer to
    explain the corrupted row's reconstruction error. The truth is the 3 features we corrupted: the
    error rose because of them. Only features that are non-zero in that row are eligible (zero x 10 = 0
    changes nothing). Metric: precision@3 = share of the explainer's top-3 that are the corrupted features.
    Chance is 3/p. Two controls anchor the scale: a random ranking, and "largest |z|" (which ignores the
    model entirely).

(b) PLANTED-SIGNAL TEST (supervised models). For a supervised model there is no independent way to know
    which features it "should" use. So we train a small MLP on a synthetic target that depends on exactly
    three chosen features,   y = 2*u_a + 1.5*sin(1.5*u_b) + u_a*u_c   (+ small noise),   verify it learned
    it (held-out R^2, and the share of permutation importance carried by the three), and use those three
    as the truth. This tests the explainers on a model whose answer is known; it is a stand-in for
    TabNet / LSTM, not those models themselves. Features with a near-duplicate (|rho| >= 0.8) are not
    chosen as drivers, because credit could then legitimately go to the twin.
"""
import numpy as np
import torch
import torch.nn as nn

from src.benchmarks.faithfulness import rank_order


def corrupt_rows(X_raw, multiplier, n_corrupt=3, seed=0):
    """Multiply `n_corrupt` random NON-ZERO features of every row by `multiplier`. Returns (X_corrupt, truth (n, n_corrupt))."""
    X = np.asarray(X_raw, dtype=np.float64)
    rng = np.random.RandomState(seed)
    Xc = X.copy()
    truth = np.empty((len(X), n_corrupt), dtype=int)
    for i in range(len(X)):
        eligible = np.where(X[i] != 0)[0]
        cols = rng.choice(eligible, n_corrupt, replace=False)
        Xc[i, cols] *= multiplier
        truth[i] = cols
    return Xc, truth


def precision_at_k(A, truth, k=3, seed=0):
    """Mean share of the top-k features (by |attribution|, random tie-break) that are in the truth set."""
    order = rank_order(A, np.random.RandomState(seed))[:, :k]
    hits = np.array([len(set(order[i]) & set(truth[i])) for i in range(len(A))], dtype=float)
    return float(np.mean(hits / k)), hits / k


def bootstrap_ci(values, n_boot=2000, seed=0):
    v = np.asarray(values, dtype=float)
    r = np.random.RandomState(seed)
    m = [v[r.randint(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


class _Scalar(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x).squeeze(-1)


def planted_target(U, a, b, c):
    return 2.0 * U[:, a] + 1.5 * np.sin(1.5 * U[:, b]) + U[:, a] * U[:, c]


def make_planted_task(U_ref, seed=0, n_train=40000, n_hold=4000, epochs=30, max_twin_corr=0.8):
    """
    Build a planted-signal model. Returns dict(model, truth, U_hold, r2_holdout, planted_share_of_permutation_importance).
    """
    U_ref = np.asarray(U_ref, dtype=np.float32)
    rng = np.random.RandomState(seed)
    p = U_ref.shape[1]
    n_train = min(n_train, int(0.85 * len(U_ref)))
    n_hold = min(n_hold, len(U_ref) - n_train)
    idx = rng.permutation(len(U_ref))
    tr, ho = idx[:n_train], idx[n_train:n_train + n_hold]
    S = U_ref[rng.choice(len(U_ref), min(5000, len(U_ref)), replace=False)]
    C = np.corrcoef(S.T)
    np.fill_diagonal(C, 0)
    C = np.nan_to_num(C)
    nuniq = np.array([len(np.unique(S[:, j])) for j in range(p)])
    thr_used = None
    for thr in (max_twin_corr, 0.9):                       # rule R8: fall back to 0.9 only if fewer than 10 candidates at 0.8
        cand = np.where((np.abs(C).max(axis=0) < thr) & (nuniq >= 20) & (S.std(0) > 0.3))[0]
        thr_used = thr
        if len(cand) >= 10:
            break
    assert len(cand) >= 6, f"only {len(cand)} candidate driver features"
    a, b, c = rng.choice(cand, 3, replace=False)
    y = planted_target(U_ref, a, b, c) + rng.normal(scale=0.1, size=len(U_ref))
    ysd = float(y.std())

    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(p, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    Xt, yt = torch.tensor(U_ref[tr]), torch.tensor((y[tr] / ysd).astype(np.float32))
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            j = perm[i:i + 512]
            opt.zero_grad()
            loss = ((net(Xt[j]).squeeze(-1) - yt[j]) ** 2).mean()
            loss.backward()
            opt.step()
    model = _Scalar(net).eval()
    with torch.no_grad():
        pred = model(torch.tensor(U_ref[ho])).numpy() * ysd
    r2 = 1.0 - np.mean((pred - y[ho]) ** 2) / np.var(y[ho])

    # does the model really rely on the planted features?  permutation importance on held-out rows
    base_mse = np.mean((pred - y[ho]) ** 2)
    imp = np.zeros(p)
    Xh = U_ref[ho]
    with torch.no_grad():
        for j in range(p):
            Xp = Xh.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            imp[j] = max(0.0, np.mean((model(torch.tensor(Xp)).numpy() * ysd - y[ho]) ** 2) - base_mse)
    share = float(imp[[a, b, c]].sum() / (imp.sum() + 1e-12))
    return {"model": model, "truth": np.array([a, b, c]), "U_hold": Xh, "r2_holdout": float(r2),
            "planted_importance_share": share, "twin_corr_threshold": thr_used, "n_candidates": int(len(cand))}
