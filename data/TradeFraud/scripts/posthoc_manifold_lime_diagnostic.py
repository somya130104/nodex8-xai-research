"""
POST-HOC diagnostic (NOT pre-registered; run after seeing the Phase 5 results).

Phase 5 found Manifold-LIME below vanilla LIME on the corruption ground truth (autoencoder). Two candidate causes:
  (a) integer rounding of perturbed counts, (b) the covariance-shaped (correlated) perturbations / Mahalanobis kernel.
Same 500 corrupted clients as the benchmark (10x), autoencoder, precision@3 over ALL 500 rows (no tuning/evaluation split,
because nothing is being chosen here), varying one thing at a time.

Usage: python3 data/TradeFraud/scripts/posthoc_manifold_lime_diagnostic.py
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append("/Users/somyabhadada/nodex8")
import run_benchmarks as RB
from src.benchmarks.synthetic import corrupt_rows, precision_at_k, bootstrap_ci

S = RB.setup()
b = RB.build_bundle("Model4_Autoencoder", S, False)
rs = np.random.RandomState(123)
X = np.asarray(b.gt_rows)[rs.choice(len(b.gt_rows), 500, replace=False)].astype(np.float64)
Xc, truth = corrupt_rows(X, 10, seed=123 + 10)
Uc = b.gt_transform(Xc)
ml = b.suite.mlime
rows = []
for mode, scale in [("independent", 0.3), ("manifold", 0.3), ("manifold_noround", 0.3), ("manifold_stochround", 0.3),
                    ("independent", 0.1), ("manifold", 0.1), ("manifold_noround", 0.1), ("independent", 1.0), ("manifold_noround", 1.0)]:
    A = np.array([ml.explain(Uc[i], mode=mode, seed=i, scale=scale).attribution for i in range(len(Uc))])
    m, per = precision_at_k(A, truth, 3, seed=0)
    lo, hi = bootstrap_ci(per)
    rows.append({"mode": mode, "scale": scale, "precision_at_3": m, "ci_lo": lo, "ci_hi": hi})
    print(f"{mode:22s} scale={scale:<4}  precision@3 = {m:.3f}  [{lo:.3f}, {hi:.3f}]", flush=True)
out = "/Users/somyabhadada/nodex8/data/TradeFraud/XAI_Outputs/benchmarks/"
pd.DataFrame(rows).to_csv(out + "posthoc_manifold_lime_groundtruth.csv", index=False)
json.dump({"note": "post-hoc, not pre-registered", "rows": rows}, open(out + "benchmark_posthoc_manifold_lime_summary.json", "w"), indent=1)
