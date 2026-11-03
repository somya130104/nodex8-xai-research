"""
Analysis helpers shared by the Phase 4 scripts (TradeFraud and DemandForecast), so both
datasets get the same CLI tables, the same confidence intervals and the same checks.

  cli_frame          per-instance CLI, pairwise taus, companions, gate flag and reviewer note
  group_table        CLI distribution per group (segment / decile / period) with bootstrap CIs
  faithfulness_link  does low CLI coincide with a less faithful CONSENSUS explanation?
  sanity_cases       the roadmap's "how to know Phase 4 is done" check, printed side by side
"""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.consensus import (
    cli, weighted_kendall_tau, consensus_attribution, sign_agreement_topk, importance_ranks,
)
from src.explainers.validation import replacement_effect


def _jaccard_topk(a, b, k):
    sa, sb = set(np.argsort(-np.abs(a), kind="stable")[:k]), set(np.argsort(-np.abs(b), kind="stable")[:k])
    return len(sa & sb) / len(sa | sb)


def cli_frame(attr_sets: dict, ids, thresholds=None, feature_names=None, unsigned=(), top_k=5):
    """
    attr_sets : {explainer: (n, p)} — the SAME model's explainers on the same n instances.
    Returns one row per instance.
    """
    names = list(attr_sets)
    n = len(ids)
    rows = []
    for i in range(n):
        A = {k: attr_sets[k][i] for k in names}
        r = cli(A, thresholds, feature_names=feature_names, top_k=top_k)
        row = {"Instance_ID": ids[i], "CLI": r.score}
        for (a, b), t in r.pairwise.items():
            row[f"tau_{a}_vs_{b}"] = t
        row["CLI_unweighted"] = float(np.mean([weighted_kendall_tau(A[a], A[b], exponent=0)
                                               for a, b in itertools.combinations(names, 2)]))
        row[f"top{top_k}_jaccard"] = float(np.mean([_jaccard_topk(A[a], A[b], top_k)
                                                    for a, b in itertools.combinations(names, 2)]))
        row[f"top{top_k}_sign_agreement"] = sign_agreement_topk(A, top_k, unsigned=unsigned)
        if thresholds is not None:
            row["Flag"] = r.flag
            row["Weakest_Pair"] = " vs ".join(r.weakest_pair)
            row["Review_Note"] = r.note
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(x, stat=np.mean, n_boot=2000, seed=0, alpha=0.05):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.RandomState(seed)
    b = np.array([stat(x[rng.randint(0, len(x), len(x))]) for _ in range(n_boot)])
    return float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def group_table(df, by, value="CLI"):
    out = []
    for g, d in df.groupby(by, sort=True, observed=True):
        lo, hi = bootstrap_ci(d[value].values)
        row = {by: g, "n": len(d), "mean_CLI": d[value].mean(), "ci95_lo": lo, "ci95_hi": hi,
               "median_CLI": d[value].median(), "q10_CLI": d[value].quantile(0.10)}
        if "Flag" in d:
            row["frac_RED"] = float((d["Flag"] == "RED").mean())
            row["frac_RED_ci95"] = "[{:.3f}, {:.3f}]".format(*bootstrap_ci((d["Flag"] == "RED").values.astype(float)))
        out.append(row)
    return pd.DataFrame(out)


def faithfulness_link(cli_scores, attr_sets, scalar_np, U, donors=None, k=5, seed=0):
    """
    For each instance build the CONSENSUS ranking (mean rank over the explainers), replace its
    top-k features (donors=None -> set to the typical value 0; else a random real donor) and
    measure the change in the explained scalar relative to replacing k random features.
    If the gate is meaningful, low-CLI instances should have a LESS faithful consensus.
    Returns (per-instance ratio array, summary dict). Inherits the operator biases documented
    in ARCHITECTURE.md (Phase 3): indicative, not a benchmark.
    """
    n = len(U)
    C = np.vstack([consensus_attribution({k_: attr_sets[k_][i] for k_ in attr_sets}) for i in range(n)])
    C = C - C.min(axis=1, keepdims=True)              # non-negative importance; ranking unchanged
    res = replacement_effect(scalar_np, U, C, donors, k=k, n_random=10, seed=seed)
    ratio = res["top_effect"] / (res["random_effect"] + 1e-12)
    s = np.asarray(cli_scores, float)
    ok = np.isfinite(s) & np.isfinite(ratio)
    rho, pval = spearmanr(s[ok], ratio[ok])
    lowq = s <= np.nanquantile(s, 0.25)
    return ratio, {"spearman_CLI_vs_consensus_top_over_random": float(rho), "p_value": float(pval),
                   "median_ratio_lowest_CLI_quartile": float(np.median(ratio[lowq & ok])),
                   "median_ratio_upper_three_quartiles": float(np.median(ratio[~lowq & ok])),
                   "frac_top_gt_random_lowest_quartile": float(np.mean(ratio[lowq & ok] > 1)),
                   "frac_top_gt_random_upper": float(np.mean(ratio[~lowq & ok] > 1))}


def sanity_cases(df, attr_sets, feature_names, ids, n_each=3, k=5):
    """Top-k lists of every explainer for the n_each highest- and lowest-CLI instances."""
    pos = {v: i for i, v in enumerate(ids)}
    d = df.dropna(subset=["CLI"]).sort_values("CLI")
    lines, rows = [], []
    for title, sel in (("HIGHEST CLI", d.tail(n_each).iloc[::-1]), ("LOWEST CLI", d.head(n_each))):
        lines.append(f"\n  --- {title} ---")
        for _, r in sel.iterrows():
            i = pos[r["Instance_ID"]]
            tops = {e: [feature_names[j] for j in np.argsort(importance_ranks(attr_sets[e][i]), kind="stable")[:k]]
                    for e in attr_sets}
            shared = set.intersection(*[set(t) for t in tops.values()])
            pairs = [len(set(tops[a]) & set(tops[b])) for a, b in itertools.combinations(tops, 2)]
            lines.append(f"  {r['Instance_ID']}  CLI={r['CLI']:.3f}  shared by all={len(shared)}/{k}  "
                         f"pairwise overlap={pairs}")
            for e, t in tops.items():
                lines.append(f"      {e:13s}: " + ", ".join(t))
            rows.append({"group": title, "Instance_ID": r["Instance_ID"], "CLI": r["CLI"],
                         "shared_by_all": len(shared), "mean_pairwise_overlap": float(np.mean(pairs))})
    return "\n".join(lines), pd.DataFrame(rows)
