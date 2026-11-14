"""
5.1  Deletion / insertion faithfulness (AOPC).

For an instance x with attribution a, rank features by |a| and move through them in that order:
  deletion   replace the top-k features of x by a reference row b            (output should move FAST)
  insertion  start from b and put back the top-k features of x               (output should recover FAST)
The reference b is either the typical client (all zeros in the preprocessed space: "zero") or a random
real normal client ("donor"). Both are always reported, because each has a known bias (ARCHITECTURE.md
Phase 3): zero favours attributions proportional to the input, donor makes mixed, off-manifold rows.

Metric: the mean absolute output change over k = 1..K ("area over the perturbation curve"), reported as
a RATIO to the same quantity under random rankings (1.0 = no better than random) and as the share of
instances where the explainer beats its own random control.
"""
import numpy as np


def rank_order(A, rng):
    """Feature indices by descending |A| per row; exact ties (e.g. TabNet's zeros) broken at random."""
    A = np.abs(np.asarray(A, dtype=np.float64))
    return np.lexsort((rng.random_sample(A.shape), -A), axis=-1)


def _walk(f_np, U, base, order, K):
    n = len(U)
    ii = np.arange(n)
    cur_del, cur_ins = U.copy(), base.copy()
    f_del = np.empty((n, K))
    f_ins = np.empty((n, K))
    for k in range(K):
        cols = order[:, k]
        cur_del[ii, cols] = base[ii, cols]
        cur_ins[ii, cols] = U[ii, cols]
        f_del[:, k] = f_np(cur_del)
        f_ins[:, k] = f_np(cur_ins)
    return f_del, f_ins


def deletion_insertion(f_np, U, A, K, base, n_random=5, seed=0):
    """
    Returns per-instance effects and mean curves for one (attribution set, operator).
      del_effect[i] = mean_k |f(x_i) - f(x_i with top-k deleted)|
      ins_effect[i] = mean_k |f(base_i with top-k inserted) - f(base_i)|
    plus the same quantities under n_random random rankings (averaged).
    """
    U = np.asarray(U, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32)
    K = int(min(K, U.shape[1]))
    rng = np.random.RandomState(seed)
    fx, fb = f_np(U), f_np(base)

    def effects(order):
        f_del, f_ins = _walk(f_np, U, base, order, K)
        d = np.abs(fx[:, None] - f_del)
        s = np.abs(f_ins - fb[:, None])
        return d, s

    d, s = effects(rank_order(A, rng)[:, :K])
    rd, rs_ = [], []
    for r in range(n_random):
        dr, sr = effects(rank_order(rng.normal(size=U.shape), rng)[:, :K])
        rd.append(dr)
        rs_.append(sr)
    rd, rs_ = np.mean(rd, axis=0), np.mean(rs_, axis=0)
    out = {"del_effect": d.mean(1), "ins_effect": s.mean(1), "rand_del_effect": rd.mean(1), "rand_ins_effect": rs_.mean(1),
           "curve_del": d.mean(0), "curve_ins": s.mean(0), "curve_rand_del": rd.mean(0), "curve_rand_ins": rs_.mean(0)}
    return out


def summarize(res):
    """Scalar summary of one deletion_insertion() result."""
    eps = 1e-12
    return {"del_aopc": float(res["del_effect"].mean()), "del_over_random": float(res["del_effect"].mean() / (res["rand_del_effect"].mean() + eps)),
            "del_win_rate": float(np.mean(res["del_effect"] > res["rand_del_effect"])),
            "ins_aopc": float(res["ins_effect"].mean()), "ins_over_random": float(res["ins_effect"].mean() / (res["rand_ins_effect"].mean() + eps)),
            "ins_win_rate": float(np.mean(res["ins_effect"] > res["rand_ins_effect"]))}
