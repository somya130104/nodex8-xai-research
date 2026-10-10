"""Writes the high-level fidelity score CSV: one row per validation run
(model x dataset — the same unit each H_data fingerprint covers), every
sub-metric as its own column, and one composite_fidelity_score column."""
from __future__ import annotations

import csv
import os

FIELDS = [
    "run_id", "model", "model_family", "problem_type", "dataset",
    "n_rows", "n_valid", "completeness",
    "atol", "rtol", "decision_threshold", "decision_threshold_source",
    "tolerance_pass_rate", "decision_agreement_rate", "rank_fidelity",
    "top_decile_overlap", "distributional_similarity", "calibration_score",
    "mean_signed_bias",
    "composite_fidelity_score", "composite_outcome",
    "fidelity_pct", "outcome",
    "max_abs_delta", "mean_abs_delta", "p95_abs_delta", "p99_abs_delta",
    "H_data", "H_model_version", "H_output", "H_decision",
    "validated_csv",
]


def write_csv(results: list[dict], out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            if "composite_fidelity_score" not in r:
                continue   # SKIPPED runs never reached fidelity scoring
            writer.writerow({k: r.get(k, "") for k in FIELDS})
    return out_path
