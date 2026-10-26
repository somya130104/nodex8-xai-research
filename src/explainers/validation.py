"""
Validation utilities shared by all three explainers.

replacement_effect — a basic faithfulness sanity check (ROADMAP.md 3.4 step 8;
    the full benchmark suite is Phase 5): if an explainer says features A are
    the ones responsible for the scalar's value, then overwriting A with values
    from a typical (normal) client should change the scalar a lot, while
    overwriting features the explainer ranked LAST should change it little.
    The same donor row is used for every ranking being compared, so the
    comparison is paired.

pairwise_agreement — mean weighted-tau agreement between explainers'
    attribution rankings. PLACEHOLDER: uses src.eval_utils.baseline_instance_tau
    (scipy's weighted tau). It is NOT the CLI; Phase 4 builds and unit-tests
    the real weighted Kendall tau and the per-instance CLI per the proposal's
    Equation 1.

nearest_real_distance — how far each generated perturbation is from the
    closest REAL normal client (the plausibility diagnostic for Manifold-LIME).
"""
import itertools
import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.eval_utils import baseline_instance_tau


def _topk_idx(attr_abs, k, largest=True):
    order = np.argsort(-attr_abs if largest else attr_abs, axis=1)
    return order[:, :k]


def replacement_effect(scalar_np, U, attributions, donors, k=5, n_random=10, seed=0):
    """
    scalar_np: callable (n, p) float32 array -> (n,) array (the scalar being explained).
    U: (n, p) instances. attributions: (n, p) (signed or not; ranked by |.|).
    donors: (m, p) real normal rows; one random donor is drawn per instance.
        donors=None switches to a "repair" test: the replaced columns are set to 0, the
        typical (normal-mean) value in the standardised / preprocessed coordinate systems
        used here. This avoids the side-effect of grafting one real client's values onto
        another (a mixed, off-manifold client), which itself raises reconstruction error
        — visible in the random-ranking control, where ANY replacement shifts the scalar.

    Returns dict with per-instance arrays:
      top_effect, bottom_effect, random_effect : mean |f(x_replaced) - f(x)| when the
          top-k / bottom-k / random-k ranked features are overwritten by the donor's values;
      top_reduction : mean (f(x) - f(x_replaced)) for the top-k (positive = error fell,
          i.e. "repairing" the flagged features made the client look more normal).
    """
    rng = np.random.RandomState(seed)
    n, p = U.shape
    donor_rows = np.zeros_like(U) if donors is None else donors[rng.randint(0, len(donors), n)]
    base = scalar_np(U.astype(np.float32))
    A = np.abs(attributions)

    def modified(idx_matrix):
        M = U.copy()
        rows = np.arange(n)[:, None]
        M[rows, idx_matrix] = donor_rows[rows, idx_matrix]
        return scalar_np(M.astype(np.float32))

    top = modified(_topk_idx(A, k, largest=True))
    bottom = modified(_topk_idx(A, k, largest=False))
    rand_effects = []
    for _ in range(n_random):
        ridx = np.array([rng.choice(p, k, replace=False) for _ in range(n)])
        rand_effects.append(np.abs(modified(ridx) - base))
    return dict(top_effect=np.abs(top - base), bottom_effect=np.abs(bottom - base),
                random_effect=np.mean(rand_effects, axis=0), top_reduction=base - top)


def summarize_replacement(res):
    t, b, r = res["top_effect"], res["bottom_effect"], res["random_effect"]
    return {
        "mean_top_effect": float(t.mean()),
        "mean_bottom_effect": float(b.mean()),
        "mean_random_effect": float(r.mean()),
        "top_over_bottom_ratio": float(t.mean() / (b.mean() + 1e-12)),
        "top_over_random_ratio": float(t.mean() / (r.mean() + 1e-12)),
        "frac_instances_top_gt_bottom": float(np.mean(t > b)),
        "mean_top_reduction": float(res["top_reduction"].mean()),
    }


def pairwise_agreement(attr_by_name, rows=None):
    """
    attr_by_name: {name: (n, p)}. Returns {(a, b): mean weighted-tau over instances}
    (restricted to `rows` if given). Placeholder tau — see module docstring.
    """
    names = list(attr_by_name)
    out = {}
    for a, b in itertools.combinations(names, 2):
        A, B = attr_by_name[a], attr_by_name[b]
        idx = range(len(A)) if rows is None else rows
        taus = [baseline_instance_tau(A[i], B[i]) for i in idx]
        taus = [t for t in taus if t is not None]
        out[(a, b)] = float(np.mean(taus)) if taus else float("nan")
    return out


def nearest_real_distance(Z, U_reference, n_subsample=10000, seed=0):
    """Euclidean distance from each row of Z to its nearest real normal row."""
    rng = np.random.RandomState(seed)
    ref = U_reference if len(U_reference) <= n_subsample else U_reference[rng.choice(len(U_reference), n_subsample, replace=False)]
    nn = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(ref)
    d, _ = nn.kneighbors(Z)
    return d[:, 0]
