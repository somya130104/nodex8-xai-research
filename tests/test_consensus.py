"""
Tests for src/consensus.py — the weighted Kendall tau and the CLI.

ROADMAP.md 4.1: "a silent bug here invalidates every CLI number downstream", so the
tau is checked against (a) its defining properties, (b) the roadmap's explicit
top-swap vs tail-swap requirement, and (c) an independent implementation (scipy).
"""
import numpy as np
import pytest
from scipy.stats import weightedtau, kendalltau

from src.consensus import (
    weighted_kendall_tau, cli, GateThresholds, calibrate_thresholds, disputed_features,
    consensus_attribution, sign_agreement_topk, model_rashomon_index, kendalls_w,
)


def scores_from_order(order):
    """order[k] = feature at position k (0 = most important) -> importance scores."""
    s = np.zeros(len(order))
    s[np.asarray(order)] = np.arange(len(order), 0, -1)
    return s


# --------------------------------------------------------------- roadmap 4.1 required tests
def test_identical_rankings_give_one():
    a = np.random.default_rng(0).normal(size=40)
    assert weighted_kendall_tau(a, a) == pytest.approx(1.0)
    assert weighted_kendall_tau(a, 3 * a) == pytest.approx(1.0)            # scale-invariant


def test_reversed_rankings_give_minus_one():
    a = np.arange(1, 41, dtype=float)
    assert weighted_kendall_tau(a, 1.0 / a) == pytest.approx(-1.0)


def test_swap_at_top_costs_more_than_swap_in_tail():
    p = 100
    base = list(range(p))
    top = base.copy(); top[0], top[1] = top[1], top[0]
    tail = base.copy(); tail[98], tail[99] = tail[99], tail[98]
    t_top = weighted_kendall_tau(scores_from_order(base), scores_from_order(top))
    t_tail = weighted_kendall_tau(scores_from_order(base), scores_from_order(tail))
    assert t_top < t_tail < 1.0
    # unweighted Kendall cannot tell them apart — the reason for weighting
    u_top = weighted_kendall_tau(scores_from_order(base), scores_from_order(top), exponent=0)
    u_tail = weighted_kendall_tau(scores_from_order(base), scores_from_order(tail), exponent=0)
    assert u_top == pytest.approx(u_tail)


def test_top_disagreement_monotone_in_position():
    """Swapping adjacent features costs less the further down the ranking it happens."""
    p, base = 30, list(range(30))
    taus = []
    for k in range(0, p - 1, 4):
        o = base.copy(); o[k], o[k + 1] = o[k + 1], o[k]
        taus.append(weighted_kendall_tau(scores_from_order(base), scores_from_order(o)))
    assert all(x < y for x, y in zip(taus, taus[1:]))


# --------------------------------------------------------------- independent implementations
def test_matches_scipy_on_tie_free_input():
    rng = np.random.default_rng(1)
    for _ in range(25):
        p = int(rng.integers(5, 300))
        a = rng.normal(size=p)
        b = a + rng.normal(scale=rng.uniform(0.1, 3), size=p)
        # scipy works on raw values; pass non-negative values so |.| is a no-op
        a, b = np.abs(a) + 1e-3, np.abs(b) + 1e-3
        assert weighted_kendall_tau(a, b) == pytest.approx(weightedtau(a, b).statistic, abs=1e-12)


def test_exponent_zero_is_kendall_tau_b_with_ties():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 6, size=50).astype(float)          # heavy ties
    b = a + rng.integers(0, 3, size=50)
    assert weighted_kendall_tau(a, b, exponent=0) == pytest.approx(kendalltau(a, b).statistic)


# --------------------------------------------------------------- other properties
def test_symmetric_and_bounded():
    rng = np.random.default_rng(3)
    for _ in range(20):
        a, b = rng.normal(size=60), rng.normal(size=60)
        t = weighted_kendall_tau(a, b)
        assert t == pytest.approx(weighted_kendall_tau(b, a))
        assert -1 <= t <= 1


def test_ranks_by_magnitude_not_sign():
    a = np.array([5.0, -4.0, 3.0, -2.0, 1.0])
    assert weighted_kendall_tau(a, np.abs(a)) == pytest.approx(1.0)
    assert weighted_kendall_tau(a, -a) == pytest.approx(1.0)
    assert weighted_kendall_tau(a, -a, absolute=False) < 0


def test_invariant_to_monotone_transform():
    rng = np.random.default_rng(4)
    a, b = np.abs(rng.normal(size=80)), np.abs(rng.normal(size=80))
    assert weighted_kendall_tau(a, b) == pytest.approx(weighted_kendall_tau(a ** 3, np.log1p(b)))


def test_undefined_when_an_explainer_is_constant():
    assert np.isnan(weighted_kendall_tau(np.zeros(10), np.arange(10.0)))


def test_tied_tail_does_not_count_as_agreement():
    """Two explainers that agree only on 'everything else is zero' should not look like they agree."""
    p = 50
    a = np.zeros(p); a[[0, 1, 2]] = [3, 2, 1]
    b = np.zeros(p); b[[3, 4, 5]] = [3, 2, 1]
    assert weighted_kendall_tau(a, b) < 0.0
    assert weighted_kendall_tau(a, a) == pytest.approx(1.0)


# --------------------------------------------------------------- CLI
def test_cli_is_mean_of_pairwise_taus():
    rng = np.random.default_rng(5)
    A = {k: rng.normal(size=30) for k in ("shap", "lime", "lrd")}
    res = cli(A)
    assert len(res.pairwise) == 3
    assert res.score == pytest.approx(np.mean(list(res.pairwise.values())))
    assert res.pairwise[("shap", "lime")] == pytest.approx(weighted_kendall_tau(A["shap"], A["lime"]))


def test_cli_perfect_agreement_and_flags():
    a = np.arange(20, 0, -1.0)
    th = GateThresholds(red=0.2, amber=0.4)
    res = cli({"x": a, "y": 2 * a, "z": a + 0.5}, th)
    assert res.score == pytest.approx(1.0) and res.flag == "GREEN" and res.note == ""
    assert th.flag(0.1) == "RED" and th.flag(0.3) == "AMBER" and th.flag(np.nan) == "RED"


def test_cli_review_note_names_weakest_pair_and_disputed_feature():
    p = 20
    a = scores_from_order(list(range(p)))
    b = a.copy()
    c = scores_from_order([19] + list(range(19)))      # c puts feature 19 (last for a, b) at #1
    res = cli({"shap": a, "lime": b, "lrd": c}, GateThresholds(red=0.99, amber=0.995),
              feature_names=[f"feat{j}" for j in range(p)])
    assert res.flag == "RED"
    assert "lrd" in res.weakest_pair
    assert res.disputed[0][0] == "feat19"
    assert "REVIEW" in res.note and "feat19" in res.note


def test_disputed_features_orders_by_rank_gap():
    a = scores_from_order([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = scores_from_order([9, 1, 2, 3, 4, 5, 6, 7, 8, 0])
    d = disputed_features({"a": a, "b": b}, ("a", "b"), top_k=2)
    assert {d[0][0], d[1][0]} == {"f0", "f9"}


def test_calibrate_thresholds_quantiles():
    s = np.arange(100.0) / 100
    th = calibrate_thresholds(np.r_[s, np.nan])
    assert th.red == pytest.approx(np.quantile(s, 0.10))
    assert th.amber == pytest.approx(np.quantile(s, 0.25))
    assert np.mean([th.flag(x) == "RED" for x in s]) == pytest.approx(0.10, abs=0.01)


def test_consensus_and_sign_agreement():
    a = np.array([5.0, 4, 3, 2, 1])
    assert np.argmax(consensus_attribution({"x": a, "y": a, "z": -a})) == 0
    assert sign_agreement_topk({"x": a, "y": -a}, top_k=3) == 0.0          # all opposite
    assert sign_agreement_topk({"x": a, "y": a, "lrd": -np.abs(a)}, top_k=3, unsigned=("lrd",)) == 1.0
    assert np.isnan(sign_agreement_topk({"x": a, "lrd": np.abs(a)}, unsigned=("lrd",)))   # one signed explainer


def test_rashomon_and_kendalls_w():
    a = np.arange(10.0)
    r = model_rashomon_index({"m1": a, "m2": a, "m3": a})
    assert r["index"] == pytest.approx(1.0)
    assert kendalls_w(np.vstack([a, a, a])) == pytest.approx(1.0)
    assert kendalls_w(np.vstack([a, a[::-1]])) == pytest.approx(0.0)
