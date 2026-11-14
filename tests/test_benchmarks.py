"""
Tests for src/benchmarks — the measuring instrument for Phase 5.
Each test checks a piece against a case whose answer is known, because a bug here would
silently corrupt every benchmark number built on it.
"""
import sys
import numpy as np
import pytest

sys.path.insert(0, "/Users/somyabhadada/nodex8")
from src.benchmarks.faithfulness import rank_order, deletion_insertion, summarize
from src.benchmarks.synthetic import corrupt_rows, precision_at_k, make_planted_task, planted_target
from src.benchmarks.robustness import noise_stability, adversarial_search
from src.benchmarks import runtime


# ------------------------------------------------------------------ faithfulness
def test_rank_order_sorts_by_magnitude_and_breaks_ties_randomly():
    A = np.array([[0.1, -5.0, 2.0, 0.0]])
    assert list(rank_order(A, np.random.RandomState(0))[0][:3]) == [1, 2, 0]
    Z = np.zeros((4000, 6))
    first = rank_order(Z, np.random.RandomState(1))[:, 0]
    counts = np.bincount(first, minlength=6) / len(first)
    assert counts.min() > 0.12 and counts.max() < 0.22          # no systematic favouritism for low feature indices


def test_deletion_insertion_separates_true_from_wrong_and_random_attributions():
    rng = np.random.RandomState(0)
    w = np.array([5.0, 3.0, 1.0] + [0.0] * 17)
    f = lambda X: np.asarray(X) @ w
    U = rng.normal(size=(300, 20)).astype(np.float32)
    base = np.zeros_like(U)
    true_attr = U * w                                            # exact attribution of a linear model
    wrong_attr = U * (1.0 - (w > 0))                             # only the features the model ignores
    r_true = summarize(deletion_insertion(f, U, true_attr, 5, base, seed=1))
    r_wrong = summarize(deletion_insertion(f, U, wrong_attr, 5, base, seed=1))
    r_rand = summarize(deletion_insertion(f, U, rng.normal(size=U.shape), 5, base, seed=1))
    assert r_true["del_over_random"] > 3 and r_true["ins_over_random"] > 3
    assert r_wrong["del_over_random"] < 0.2                      # picks features the model ignores -> output barely moves
    assert 0.6 < r_rand["del_over_random"] < 1.6                 # a random ranking is about as good as the random control
    assert r_true["del_win_rate"] > 0.95 and r_wrong["del_win_rate"] < 0.1


def test_deletion_curve_is_cumulative_and_final_row_matches_manual_computation():
    f = lambda X: np.asarray(X).sum(axis=1)
    U = np.array([[3.0, 2.0, 1.0, 0.0]], dtype=np.float32)
    res = deletion_insertion(f, U, np.array([[3.0, 2.0, 1.0, 0.0]]), 3, np.zeros_like(U), n_random=1, seed=0)
    assert np.allclose(res["curve_del"], [3.0, 5.0, 6.0])        # deleting 3, then 3+2, then 3+2+1
    assert np.allclose(res["curve_ins"], [3.0, 5.0, 6.0])        # inserting them back from zero


# ------------------------------------------------------------------ synthetic ground truth
def test_corrupt_rows_changes_exactly_three_nonzero_features_by_the_multiplier():
    rng = np.random.RandomState(0)
    X = rng.exponential(size=(200, 30)) * (rng.rand(200, 30) > 0.6)   # ~60% zeros
    Xc, truth = corrupt_rows(X, 10, seed=3)
    assert truth.shape == (200, 3) and all(len(set(t)) == 3 for t in truth)
    for i in range(200):
        changed = np.where(Xc[i] != X[i])[0]
        assert set(changed) == set(truth[i])
        assert np.allclose(Xc[i, truth[i]], 10 * X[i, truth[i]]) and np.all(X[i, truth[i]] != 0)


def test_precision_at_3_is_one_for_perfect_chance_for_random_and_tie_safe():
    rng = np.random.RandomState(0)
    n, p = 3000, 100
    truth = np.array([rng.choice(p, 3, replace=False) for _ in range(n)])
    perfect = np.zeros((n, p))
    for i in range(n):
        perfect[i, truth[i]] = [3, 2, 1]
    assert precision_at_k(perfect, truth)[0] == 1.0
    assert abs(precision_at_k(rng.normal(size=(n, p)), truth)[0] - 3 / p) < 0.01
    # an explainer that outputs all zeros must score ~chance, NOT get a free ride from index order
    low_index_truth = np.tile([0, 1, 2], (n, 1))
    assert precision_at_k(np.zeros((n, p)), low_index_truth)[0] < 0.06


def test_planted_task_learns_the_planted_function_and_relies_on_the_three_features():
    rng = np.random.RandomState(0)
    U = rng.normal(size=(6000, 20)).astype(np.float32)
    U[:, 5] = U[:, 4] + 0.01 * rng.normal(size=6000)                  # a near-duplicate pair: must not be chosen as a driver
    task = make_planted_task(U, seed=1, n_train=4500, n_hold=1200, epochs=25)
    assert task["r2_holdout"] > 0.9
    assert task["planted_importance_share"] > 0.8
    assert not ({4, 5} & set(task["truth"]))
    with __import__("torch").no_grad():
        pred = task["model"](__import__("torch").tensor(task["U_hold"])).numpy()
    a, b, c = task["truth"]
    assert np.corrcoef(pred, planted_target(task["U_hold"], a, b, c))[0, 1] > 0.95


# ------------------------------------------------------------------ robustness
class _StubSuite:
    """f ignores feature 1; the 'explainer' is |x| (deterministic)."""
    def __init__(self):
        self.f_np = lambda X: np.asarray(X)[:, 0].astype(np.float64)

    def explain(self, name, U, seed=0, same_seed=False):
        return np.abs(np.asarray(U, dtype=np.float64))

    def is_stochastic(self, name):
        return False


def test_noise_stability_is_high_for_a_smooth_explainer_and_control_is_one_for_deterministic():
    suite = _StubSuite()
    U = np.random.RandomState(0).normal(size=(40, 25)) * 3
    res = noise_stability(suite, "abs", U, sigma=np.ones(25))
    assert res["tau_noise"].mean() > 0.97 and np.all(res["tau_control"] == 1.0)


def test_adversarial_search_finds_a_flip_the_model_cannot_see_and_beats_random_noise():
    suite = _StubSuite()
    x = np.tile(np.array([1.00, 0.97, 0.5, 0.2], dtype=np.float32), (4, 1))   # only the top pair (0 vs 1) can plausibly swap
    res = adversarial_search(suite, "abs", x, eps_in=0.05, eps_f=0.01, iters=25, cands=6, seed=0)
    # feature 1 is invisible to the model, so the search can lift it above feature 0 while the output stays within eps_f
    assert np.all(res["top1_changed"] == 1.0)
    assert res["tau_confirmed"].mean() < res["tau_random"].mean()              # directed search does more damage than random noise
    assert res["tau_confirmed"].mean() < 0.9 and np.all(res["feasible_share"] > 0.2)


def test_adversarial_search_respects_the_output_constraint():
    """With eps_f = 0 only perturbations that leave f exactly unchanged are allowed: feature 0 may not move."""
    suite = _StubSuite()
    x = np.zeros((2, 4), dtype=np.float32); x[:, 0] = 1.0; x[:, 1] = 1.0
    res = adversarial_search(suite, "abs", x, eps_in=0.05, eps_f=0.0, iters=10, cands=6, seed=1)
    assert np.all(res["feasible_share"] < 0.01)      # random perturbations essentially never keep f exactly fixed


# ------------------------------------------------------------------ runtime
def test_latency_reports_ordered_percentiles_and_memory_sees_a_known_allocation():
    import time
    lat = runtime.latency(lambda i: time.sleep(0.002), n=30, warmup=1)
    assert lat["n"] == 30 and 1.5 < lat["p50_ms"] < 10 and lat["p50_ms"] <= lat["p95_ms"] <= lat["max_ms"]
    mem = runtime.peak_memory(lambda i: np.ones(8_000_000), n=3)         # 64 MB per call
    assert mem["peak_python_mb"] > 55
