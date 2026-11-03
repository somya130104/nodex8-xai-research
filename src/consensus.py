"""
The Consensus Logic Index (CLI) and its uncertainty gate (ROADMAP.md Phase 4).

weighted_kendall_tau  Agreement between two explainers' rankings of the SAME
    features for the SAME instance, with disagreements near the top of the
    ranking penalised more than disagreements in the tail.

    Definition (Vigna 2015, "A weighted correlation index for rankings with
    ties", hyperbolic additive weighting — the variant the proposal calls for):
      importance  s_j = |attribution_j|   (rank 0 = most important)
      pair weight w_ij = h(r_i) + h(r_j),   h(r) = 1 / (r + 1)^exponent
      tau_r = sum_{i<j} w_ij sgn(a_i - a_j) sgn(b_i - b_j)
              / sqrt( sum_{a_i != a_j} w_ij  *  sum_{b_i != b_j} w_ij )
    Ranks r come from one of the two rankings, so the index is computed twice
    (weights from a's ranking, then from b's) and the two are averaged — this
    makes it symmetric. A pair counts as soon as EITHER explainer ranks one
    of its features highly, so a feature one explainer puts at #1 and the
    other buries is penalised whichever explainer is "right".

    Ties (e.g. TabNet's exact-zero attributions, or features an explainer
    cannot see) are handled tau-b style: a tied pair contributes nothing to
    the numerator and is dropped from that explainer's normaliser; tied
    features share their average rank for the weights.

    On tie-free input this is IDENTICAL to scipy.stats.weightedtau's default
    (verified in tests/test_consensus.py — two independent implementations
    agreeing is the strongest check available). It is implemented here rather
    than taken from scipy because (1) scipy breaks ties in one ranking by the
    other ranking's values, which leaks information between the two explainers
    being compared, (2) the exponent is exposed for sensitivity analysis
    (exponent 0 = ordinary Kendall tau-b), and (3) the definition is then
    explicit in this repository, not an implementation detail of a library.

    Ranking is by |attribution|: the three explainers do not share a sign
    convention (LRD is a non-negative share of an error, IG and LIME are
    signed contributions), so "which features matter, in which order" is the
    only question all three answer. Direction agreement is reported separately.

cli  CLI(x) = 2 / (K (K - 1)) * sum_{i<j} tau_w(a_i(x), a_j(x))   (proposal, Eq. 1)
    for K explainers of ONE model on ONE instance. Never average across models —
    that is model_rashomon_index, a different question.

gate  GREEN / AMBER / RED from calibrated thresholds; RED = route to a human
    reviewer, with a note naming the explainer pair that disagreed most and the
    features they disagreed about.
"""
from dataclasses import dataclass, field
import itertools
import numpy as np
from scipy.stats import rankdata


# --------------------------------------------------------------------------- tau
def _signs(v):
    return np.sign(v[:, None] - v[None, :])


def _tau_one_side(Sa, Sb, ranks, exponent):
    h = 1.0 / (ranks + 1.0) ** exponent
    W = h[:, None] + h[None, :]
    den = np.sqrt((W * np.abs(Sa)).sum() * (W * np.abs(Sb)).sum())
    if den == 0.0:
        return np.nan
    return float((W * Sa * Sb).sum() / den)


def weighted_kendall_tau(attr_a, attr_b, exponent: float = 1.0, absolute: bool = True) -> float:
    """
    attr_a, attr_b : attribution (or importance) vectors over the same features,
        larger = more important (after abs() if absolute=True).
        To pass ranks where 1 = most important, pass -rank with absolute=False.
    exponent : 1.0 = hyperbolic weighting (primary); 0.0 = unweighted Kendall tau-b.
    Returns tau in [-1, 1], or NaN when undefined (an explainer gives every
    feature the same importance, e.g. an all-zero attribution).
    """
    a = np.asarray(attr_a, dtype=np.float64).ravel()
    b = np.asarray(attr_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    if absolute:
        a, b = np.abs(a), np.abs(b)
    if len(a) < 2 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return np.nan
    Sa, Sb = _signs(a), _signs(b)
    ra = rankdata(-a, method="average") - 1.0       # 0 = most important; ties share a rank
    rb = rankdata(-b, method="average") - 1.0
    t1 = _tau_one_side(Sa, Sb, ra, exponent)
    t2 = _tau_one_side(Sa, Sb, rb, exponent)
    if np.isnan(t1) or np.isnan(t2):
        return np.nan
    return 0.5 * (t1 + t2)


# --------------------------------------------------------------------------- CLI
@dataclass
class GateThresholds:
    """CLI < red -> RED (REVIEW);  red <= CLI < amber -> AMBER;  else GREEN."""
    red: float
    amber: float
    how: str = ""

    def flag(self, score) -> str:
        if score is None or np.isnan(score) or score < self.red:
            return "RED"                 # undefined agreement is treated as disagreement
        return "AMBER" if score < self.amber else "GREEN"


@dataclass
class CLIResult:
    score: float
    pairwise: dict                      # {(name_a, name_b): tau_w}
    flag: str = ""
    weakest_pair: tuple = ()
    disputed: list = field(default_factory=list)   # [(feature, {explainer: rank}), ...]
    note: str = ""


def importance_ranks(attr):
    """1 = most important (by |attr|); ties share their average rank."""
    return rankdata(-np.abs(np.asarray(attr, dtype=np.float64)), method="average")


def disputed_features(attrs: dict, pair, feature_names=None, top_k=5, max_items=3):
    """
    Features in the top-k of EITHER explainer of `pair`, ordered by how far
    apart the two explainers rank them (largest gap first). These are what the
    human reviewer should look at.
    """
    a, b = pair
    ra, rb = importance_ranks(attrs[a]), importance_ranks(attrs[b])
    cand = np.where((ra <= top_k) | (rb <= top_k))[0]
    cand = cand[np.argsort(-np.abs(ra[cand] - rb[cand]), kind="stable")][:max_items]
    names = feature_names if feature_names is not None else [f"f{j}" for j in range(len(ra))]
    return [(names[j], {a: int(round(ra[j])), b: int(round(rb[j]))}) for j in cand]


def cli(attributions: dict, thresholds: GateThresholds = None, feature_names=None,
        exponent: float = 1.0, top_k: int = 5) -> CLIResult:
    """
    attributions : {explainer_name: (p,) attribution vector}, all for the SAME model and
        the SAME single instance, K >= 2 entries.
    thresholds   : calibrated GateThresholds; None = score only, no flag.
    Returns the CLI score (mean pairwise weighted tau), the pairwise matrix, and — when
    thresholds are given — the GREEN/AMBER/RED flag and, for non-GREEN cases, a reviewer
    note naming the weakest pair and the features it disagrees about.
    """
    names = list(attributions)
    if len(names) < 2:
        raise ValueError("CLI needs at least two explainers")
    pw = {(a, b): weighted_kendall_tau(attributions[a], attributions[b], exponent=exponent)
          for a, b in itertools.combinations(names, 2)}
    vals = np.array(list(pw.values()))
    score = float(np.mean(vals)) if np.all(np.isfinite(vals)) else np.nan
    res = CLIResult(score=score, pairwise=pw)
    if thresholds is None:
        return res
    res.flag = thresholds.flag(score)
    finite = {k: v for k, v in pw.items() if np.isfinite(v)}
    res.weakest_pair = min(finite, key=finite.get) if finite else next(iter(pw))
    if res.flag != "GREEN":
        res.disputed = disputed_features(attributions, res.weakest_pair, feature_names, top_k=top_k)
        a, b = res.weakest_pair
        feats = "; ".join(f"{f} (#{r[a]} by {a}, #{r[b]} by {b})" for f, r in res.disputed)
        pref = "REVIEW" if res.flag == "RED" else "CAUTION"
        res.note = (f"{pref}: CLI={score:.3f}. Weakest pair {a} vs {b} (tau_w={pw[res.weakest_pair]:.3f}). "
                    f"Most disputed: {feats}")
    return res


def calibrate_thresholds(cli_scores, red_quantile: float = 0.10, amber_quantile: float = 0.25,
                         how: str = "") -> GateThresholds:
    """Quantile calibration (roadmap 4.4 method 1): flag the lowest red_quantile by construction."""
    s = np.asarray(cli_scores, dtype=float)
    s = s[np.isfinite(s)]
    return GateThresholds(red=float(np.quantile(s, red_quantile)),
                          amber=float(np.quantile(s, amber_quantile)), how=how)


def consensus_attribution(attributions: dict):
    """
    A single consensus ranking: the mean over explainers of each feature's importance
    rank (lower = more important). Returned as a score where larger = more important,
    so it can be fed back into the same ranking utilities.
    """
    R = np.vstack([importance_ranks(a) for a in attributions.values()])
    return -R.mean(axis=0)


def sign_agreement_topk(attributions: dict, top_k: int = 5, unsigned=()):
    """
    Among the consensus top-k features, the fraction where every SIGNED explainer agrees on
    direction. `unsigned` names explainers whose output carries no direction (LRD: a share of
    an error; TabNet's attention mask) — they are skipped. Returns NaN if fewer than two
    signed explainers.
    """
    signed = {k: np.asarray(v) for k, v in attributions.items() if k not in unsigned}
    if len(signed) < 2:
        return np.nan
    top = np.argsort(-consensus_attribution(attributions), kind="stable")[:top_k]
    S = np.sign(np.vstack([v[top] for v in signed.values()]))
    return float(np.mean(np.all(S == S[0], axis=0)))


# --------------------------------------------------------------------------- not the CLI
def model_rashomon_index(attr_by_model: dict, exponent: float = 1.0) -> dict:
    """
    Do DIFFERENT MODELS rely on the same features?  Same explainer, same instance (or the
    same global-importance definition), different models: mean pairwise weighted tau.
    This answers a different question from the CLI and must never be reported as one.
    Returns {"index": mean, "pairwise": {(model_a, model_b): tau}}.
    """
    pw = {(a, b): weighted_kendall_tau(attr_by_model[a], attr_by_model[b], exponent=exponent)
          for a, b in itertools.combinations(list(attr_by_model), 2)}
    vals = np.array(list(pw.values()))
    return {"index": float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan, "pairwise": pw}


def kendalls_w(importance_matrix):
    """Kendall's coefficient of concordance W for m raters (rows) ranking n items (cols), tie-corrected."""
    M = np.asarray(importance_matrix, dtype=float)
    m, n = M.shape
    R = np.vstack([rankdata(-row, method="average") for row in M])
    S = ((R.sum(0) - m * (n + 1) / 2.0) ** 2).sum()
    T = 0.0
    for row in R:
        _, c = np.unique(row, return_counts=True)
        T += (c ** 3 - c).sum()
    return float(12 * S / (m ** 2 * (n ** 3 - n) - m * T))
