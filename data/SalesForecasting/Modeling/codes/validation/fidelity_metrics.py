"""
High-level, multi-metric fidelity score for one validation run (one
model x dataset combination — the same unit H_data/H_model_version/H_output
are already fingerprinted for in engine.py). Works for both classification
(predict_proba, score bounded [0,1]) and regression (predict, unbounded
target) — see `problem_type` below.

A single "% of rows within tolerance" number hides a lot: a model could pass
that check while its ranking silently degrades, its decisions flip at the
threshold, or a systematic bias creeps in. This module scores six
independent dimensions and combines them into one composite, so a single
number is still reportable but backed by an auditable breakdown.

Metrics (each 0-100, higher = better fidelity):
  tolerance_pass_rate       — % rows with |computed-stored| <= atol + rtol*|stored|
  decision_agreement_rate   — % rows where (score >= threshold) agrees
                               (classification: probability cutoff, e.g. 0.5;
                                regression: "any demand" cutoff, e.g. > 0 —
                                same formula, different threshold value)
  rank_fidelity             — max(0, Spearman rho(stored, computed)) * 100
  top_decile_overlap        — Jaccard overlap of the top-10%-by-score sets
  distributional_similarity — (1 - KS 2-sample statistic) * 100
  calibration_score         — max(0, 1 - |mean(computed-stored)| / tolerance_scale) * 100
                               where tolerance_scale = atol + rtol*mean(|stored|)

completeness (0-100, gating, not weighted in) — % of rows with a non-NaN
computed score. Applied as a multiplier so missing/failed predictions can't
be diluted away by good performance on the rows that did compute.

composite = (0.30*DAR + 0.20*TPR + 0.15*Rank + 0.15*TopDecile
             + 0.10*Dist + 0.10*Calib) * (completeness / 100)

rtol=0 (default) reproduces the original pure-absolute-tolerance behavior
exactly — a bounded [0,1] probability score never needed a relative
tolerance. An unbounded regression target (e.g. weekly unit sales from 0 to
18,000) does: a fixed absolute tolerance is either too loose near zero or
impossibly tight at the high end, so rtol lets it scale with magnitude,
same as numpy.isclose's atol + rtol*|b| rule.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

WEIGHTS = {
    "decision_agreement_rate":   0.30,
    "tolerance_pass_rate":       0.20,
    "rank_fidelity":             0.15,
    "top_decile_overlap":        0.15,
    "distributional_similarity": 0.10,
    "calibration_score":         0.10,
}

PASS_THRESHOLD_PCT = 95.0
WARN_THRESHOLD_PCT = 85.0

DEFAULT_CLASSIFICATION_THRESHOLD = 0.5   # probability cutoff
DEFAULT_REGRESSION_THRESHOLD     = 0.5   # "any demand" cutoff for a count-like target
_TOP_DECILE_FRACTION = 0.10


def resolve_decision_threshold(
    model_pkl_path: str, problem_type: str = "classification"
) -> tuple[float, str]:
    """Best-effort: for classification, read Best_Threshold from
    performance_scorecard.csv in the same directory as the model pkl
    (preferring the Train/Dev row — the operating threshold should be
    calibrated once and applied consistently, not re-optimized per
    evaluation set). Regression has no such column in this project's
    performance.csv, so it always falls back to DEFAULT_REGRESSION_THRESHOLD.
    Always says which it used so that's auditable, not silent."""
    default = (
        DEFAULT_CLASSIFICATION_THRESHOLD if problem_type == "classification"
        else DEFAULT_REGRESSION_THRESHOLD
    )
    scorecard_path = os.path.join(os.path.dirname(model_pkl_path), "performance_scorecard.csv")
    if problem_type != "classification" or not os.path.exists(scorecard_path):
        return default, f"default ({default}) — no performance_scorecard.csv (Best_Threshold) at {scorecard_path}"
    try:
        df = pd.read_csv(scorecard_path)
        if "Best_Threshold" not in df.columns or "Dataset" not in df.columns:
            return default, "default — performance_scorecard.csv missing Best_Threshold/Dataset columns"
        train_row = df[df["Dataset"].str.contains("Train", case=False, na=False)]
        row = train_row.iloc[0] if not train_row.empty else df.iloc[0]
        return float(row["Best_Threshold"]), f"Best_Threshold from {os.path.basename(scorecard_path)} (row: {row['Dataset']})"
    except Exception as e:
        return default, f"default — could not read performance_scorecard.csv: {e}"


def _band(composite: float) -> str:
    if composite >= PASS_THRESHOLD_PCT:
        return "PASS"
    if composite >= WARN_THRESHOLD_PCT:
        return "WARN"
    return "FAIL"


def compute(
    df: pd.DataFrame,
    score_col: str,
    atol: float,
    rtol: float = 0.0,
    decision_threshold: float | None = None,
    problem_type: str = "classification",
) -> dict:
    """df must already carry a `computed_{score_col}` column (see
    inferencer.rerun_and_enrich). Rows where the computed score is NaN are
    excluded from every metric except `completeness`, which they define."""
    computed_col = f"computed_{score_col}"
    default_threshold = (
        DEFAULT_CLASSIFICATION_THRESHOLD if problem_type == "classification"
        else DEFAULT_REGRESSION_THRESHOLD
    )
    threshold = default_threshold if decision_threshold is None else decision_threshold

    n_total = len(df)
    valid = df[computed_col].notna()
    n_valid = int(valid.sum())
    completeness = round(100.0 * n_valid / n_total, 4) if n_total else 0.0

    # inferencer.rerun_and_enrich() rounds computed_* to 6dp before it's ever
    # written to the dataframe, but `stored` (the original score_col) keeps
    # full float precision. Left unrounded, that alone shows up as spurious
    # "distributional drift" under the KS test at large N (rounding shifts
    # the empirical CDF by ~1e-6, and KS gets more sensitive as N grows) even
    # though it's pure rounding noise, not a real discrepancy. Round both to
    # the same precision so every metric compares like-for-like.
    stored = np.round(df.loc[valid, score_col].to_numpy(dtype=float), 6)
    computed = np.round(df.loc[valid, computed_col].to_numpy(dtype=float), 6)
    n = len(stored)

    if n == 0:
        metrics = {k: 0.0 for k in WEIGHTS}
        composite = 0.0
        return {
            "n_rows": n_total, "n_valid": n_valid, "completeness": completeness,
            "decision_threshold": threshold, **metrics,
            "composite_fidelity_score": composite, "outcome": _band(composite),
        }

    delta = np.abs(computed - stored)
    tol_band = atol + rtol * np.abs(stored)

    # 1. Tolerance Pass Rate
    tolerance_pass_rate = round(100.0 * float((delta <= tol_band).sum()) / n, 4)

    # 2. Decision Agreement Rate
    stored_decision = stored >= threshold
    computed_decision = computed >= threshold
    decision_agreement_rate = round(100.0 * float((stored_decision == computed_decision).sum()) / n, 4)

    # 3. Rank Fidelity (Spearman)
    if n > 1 and np.std(stored) > 0 and np.std(computed) > 0:
        rho, _ = spearmanr(stored, computed)
        rho = 0.0 if np.isnan(rho) else rho
    else:
        rho = 1.0 if n <= 1 else 0.0   # a single row (or constant scores) has no ranking to break
    rank_fidelity = round(max(0.0, rho) * 100.0, 4)

    # 4. Top-Decile Overlap (Jaccard)
    k = max(1, int(np.ceil(n * _TOP_DECILE_FRACTION)))
    top_stored = set(np.argsort(stored)[::-1][:k].tolist())
    top_computed = set(np.argsort(computed)[::-1][:k].tolist())
    union = top_stored | top_computed
    top_decile_overlap = round(100.0 * len(top_stored & top_computed) / len(union), 4) if union else 100.0

    # 5. Distributional Similarity (1 - KS statistic)
    ks_stat, _ = ks_2samp(stored, computed)
    distributional_similarity = round((1.0 - float(ks_stat)) * 100.0, 4)

    # 6. Calibration / Bias — scaled the same way the tolerance band is, so
    # a systematic offset is judged relative to what "close enough" means
    # for this run's magnitude, not a fixed absolute number.
    bias = float(np.mean(computed - stored))
    tolerance_scale = atol + rtol * float(np.mean(np.abs(stored)))
    calibration_score = round(max(0.0, 1.0 - abs(bias) / tolerance_scale) * 100.0, 4) if tolerance_scale else 0.0

    metrics = {
        "tolerance_pass_rate":       tolerance_pass_rate,
        "decision_agreement_rate":   decision_agreement_rate,
        "rank_fidelity":             rank_fidelity,
        "top_decile_overlap":        top_decile_overlap,
        "distributional_similarity": distributional_similarity,
        "calibration_score":         calibration_score,
    }

    raw = sum(WEIGHTS[k] * metrics[k] for k in WEIGHTS)
    composite = round(raw * (completeness / 100.0), 4)

    return {
        "n_rows": n_total,
        "n_valid": n_valid,
        "completeness": completeness,
        "decision_threshold": threshold,
        "mean_signed_bias": round(bias, 6),
        **metrics,
        "composite_fidelity_score": composite,
        "outcome": _band(composite),
    }
