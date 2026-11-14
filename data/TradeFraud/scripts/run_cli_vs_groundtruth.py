"""
Does a low CLI mean a wrong explanation?  (closes ARCHITECTURE.md Phase 4 finding 7 / limitation 24)

On the Phase 5 corruption test (autoencoder and VAE, 500 unseen normal clients, 3 features corrupted 10x), compute for every
client the CLI of the three primary explainers (Adapted SHAP, Manifold-LIME, LRD lambda 0.5) and check it against what we
KNOW: did the explainers' consensus top-3 contain the corrupted features?

Hypothesis fixed before running: Spearman(CLI, precision@3 of the consensus top-3) > 0, and clients flagged RED (using the
Phase 4 gate thresholds, unchanged) have lower consensus precision than GREEN clients.
Caveat built into the design: corrupted rows are very anomalous, so CLI is higher here than on ordinary clients (Phase 4
found CLI rises with the model's score); this is a test of whether CLI orders explanation quality, not of the gate's flag rate.

Usage: python3 data/TradeFraud/scripts/run_cli_vs_groundtruth.py   (outputs next to the other benchmark outputs)
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append("/Users/somyabhadada/nodex8")
import run_benchmarks as RB
from src.benchmarks.synthetic import corrupt_rows, precision_at_k, bootstrap_ci
from src.consensus import cli, GateThresholds, consensus_attribution

OUT = RB.OUT
P4 = json.load(open("/Users/somyabhadada/nodex8/data/TradeFraud/XAI_Outputs/consensus/consensus_cli_summary.json"))["models"]
S = RB.setup()
rows, summary = [], {"note": "additional analysis; hypothesis fixed before running", "models": {}}
for model in ("Model4_Autoencoder", "Model9_VAE"):
    b = RB.build_bundle(model, S, False)
    th = GateThresholds(red=P4[model]["thresholds"]["red"], amber=P4[model]["thresholds"]["amber"])
    rs = np.random.RandomState(123)
    X = np.asarray(b.gt_rows)[rs.choice(len(b.gt_rows), 500, replace=False)].astype(np.float64)
    Xc, truth = corrupt_rows(X, 10, seed=123 + 10)
    Uc = b.gt_transform(Xc)
    A = {n: b.suite.explain(n, Uc, seed=0) for n in ("AdaptedSHAP", "ManifoldLIME", "LRD_lam0.5")}
    cl = np.array([cli({k: v[i] for k, v in A.items()}, th).score for i in range(len(Uc))])
    flag = np.array([th.flag(c) for c in cl])
    cons = np.array([consensus_attribution({k: v[i] for k, v in A.items()}) for i in range(len(Uc))])
    cons = cons - cons.min(axis=1, keepdims=True)      # consensus_attribution is negative-mean-rank; precision_at_k ranks by |value|, so shift to >= 0
    p_cons = precision_at_k(cons, truth, 3, seed=0)[1]
    p_each = {k: precision_at_k(v, truth, 3, seed=0)[1] for k, v in A.items()}
    rho, pv = spearmanr(cl, p_cons)
    q = pd.qcut(pd.Series(cl).rank(method="first"), 4, labels=["Q1 lowest CLI", "Q2", "Q3", "Q4 highest CLI"])
    by_q = pd.Series(p_cons).groupby(q.values).mean()
    res = {"n": len(cl), "spearman_CLI_vs_consensus_precision": float(rho), "p_value": float(pv),
           "consensus_precision_overall": float(p_cons.mean()),
           "consensus_precision_by_CLI_quartile": {k: float(v) for k, v in by_q.items()},
           "mean_CLI": float(cl.mean()), "median_CLI": float(np.median(cl)),
           "flag_counts": {f: int((flag == f).sum()) for f in ("RED", "AMBER", "GREEN")},
           "consensus_precision_by_flag": {f: (float(p_cons[flag == f].mean()) if (flag == f).any() else None) for f in ("RED", "AMBER", "GREEN")},
           "individual_precision": {k: float(v.mean()) for k, v in p_each.items()},
           "spearman_CLI_vs_best_individual_pair_tau": None}
    lo, hi = bootstrap_ci(p_cons)
    res["consensus_precision_ci95"] = [lo, hi]
    summary["models"][model] = res
    rows.append({"model": model, **{k: v for k, v in res.items() if not isinstance(v, (dict, list))}})
    print(f"\n[{model}] consensus top-3 precision {p_cons.mean():.3f} [{lo:.3f},{hi:.3f}]   (individual: " +
          ", ".join(f"{k} {v.mean():.3f}" for k, v in p_each.items()) + ")")
    print(f"   Spearman(CLI, consensus precision) = {rho:+.3f} (p={pv:.2g}); CLI median {np.median(cl):.3f}")
    print("   consensus precision by CLI quartile:", {k: round(v, 3) for k, v in by_q.items()})
    print("   flags:", res["flag_counts"], "| consensus precision by flag:", {k: (round(v, 3) if v is not None else None) for k, v in res["consensus_precision_by_flag"].items()})
pd.DataFrame(rows).to_csv(OUT + "cli_vs_groundtruth.csv", index=False)
json.dump(summary, open(OUT + "benchmark_cli_vs_groundtruth_summary.json", "w"), indent=1)
