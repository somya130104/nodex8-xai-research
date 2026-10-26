"""
Shared evaluation utilities for Phase 0 fixes (see ARCHITECTURE.md, Section 9).

These three problems were found identically across every TradeFraud training
script, so the fix is centralised here instead of copy-pasted six times:

1. Duplicated dispute rows (dedupe_rows) — the supplied data repeats every
   dispute row 10x. Left alone, this inflates apparent sample size, lets tree
   models memorise fraud rows verbatim, and makes every AUC look more
   confident than the real ~15-50 unique disputes per split support.

2. Per-dataset threshold re-optimisation (metrics_at_threshold /
   select_f1_threshold) — the original scripts picked the F1-maximising
   threshold independently on every evaluation set, including each
   out-of-time month. That silently cheats: a threshold chosen while looking
   at the answers makes precision/recall/F1 look far better than a model
   deployed with one fixed cutoff ever would. The fix: pick ONE threshold on
   validation only, then apply that same number everywhere else.

3. No uncertainty reporting (bootstrap_auc_ci) — with 15-18 real disputes in
   a typical out-of-time month, a point-estimate AUC on its own invites
   over-reading small differences between models as real. A bootstrap CI
   makes that uncertainty visible.
"""
import numpy as np
import pandas as pd
from scipy.stats import weightedtau
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)


def dedupe_rows(df: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """
    Drop exact duplicate rows (identical values across `feature_cols`),
    keeping the first occurrence.

    Safe to call on any TradeFraud split: non-dispute clients do not
    naturally share identical values across 271 columns, so this only
    removes the known 10x-repeated dispute rows (and any accidental true
    duplicates), never genuinely distinct clients.
    """
    before = len(df)
    out = df.drop_duplicates(subset=feature_cols, keep="first").reset_index(drop=True)
    removed = before - len(out)
    if removed > 0:
        print(f"  [dedupe_rows] removed {removed} duplicate rows "
              f"({before} -> {len(out)})")
    return out


def select_f1_threshold(y_true, y_score) -> float:
    """
    Choose the F1-maximising decision threshold on ONE dataset (validation
    only). This is the single place in the whole pipeline where a threshold
    is allowed to be chosen by looking at the labels.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    thresholds = np.linspace(0.01, 0.99, 200)
    best_f1, best_thr = 0.0, 0.5
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        if pred.sum() == 0:
            continue
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return float(best_thr)


def metrics_at_threshold(y_true, y_score, threshold: float) -> dict:
    """
    Precision / Recall / F1 / confusion-matrix counts at a GIVEN threshold.
    Never re-optimises — the threshold must come from select_f1_threshold()
    run once on validation, then passed in unchanged for every other split.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pred = (y_score >= threshold).astype(int)
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Threshold_Used": round(float(threshold), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1": round(float(f1), 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def bootstrap_auc_ci(y_true, y_score, n_boot: int = 1000, seed: int = 42,
                      alpha: float = 0.05):
    """
    Bootstrap confidence interval for AUC-ROC.

    Returns (point_estimate, ci_low, ci_high). If there are fewer than 2
    positive or 2 negative cases (or a resample lacks both classes), that
    resample is skipped; if too few valid resamples remain the CI is
    reported as NaN rather than silently misleading.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)

    if y_true.sum() < 2 or (len(y_true) - y_true.sum()) < 2:
        point = roc_auc_score(y_true, y_score) if 0 < y_true.sum() < n else float("nan")
        return point, float("nan"), float("nan")

    point = roc_auc_score(y_true, y_score)
    rng = np.random.RandomState(seed)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        boot_aucs.append(roc_auc_score(yt, ys))

    if len(boot_aucs) < n_boot // 2:
        return point, float("nan"), float("nan")

    lo = float(np.percentile(boot_aucs, 100 * alpha / 2))
    hi = float(np.percentile(boot_aucs, 100 * (1 - alpha / 2)))
    return float(point), round(lo, 4), round(hi, 4)


def baseline_instance_tau(attr_a, attr_b):
    """
    Placeholder weighted-τ agreement between two explainers' attribution
    vectors for ONE instance (Phase 1's rehearsal of a same-model,
    same-instance "baseline CLI" between SHAP and LIME).

    This is NOT the final CLI implementation. Phase 4 builds and
    unit-tests a proper weighted_kendall_tau() exactly per the proposal's
    Equation 1 (hyperbolic top-k weighting, validated against known
    swap-at-the-top vs swap-at-the-bottom behaviour). This wrapper exists
    only so Phase 1's local-explanation outputs can report an approximate
    same-model SHAP-vs-LIME agreement number using scipy's built-in
    weighted tau, without duplicating that unwritten Phase 4 code early
    and incorrectly. Ranks are derived from absolute attribution (which
    feature matters more), not from signed attribution (push up vs down).
    """
    a = np.asarray(attr_a, dtype=float)
    b = np.asarray(attr_b, dtype=float)
    if a.shape != b.shape or len(a) < 2:
        return None
    tau, _ = weightedtau(np.abs(a), np.abs(b))
    if tau is None or np.isnan(tau):
        return None
    return float(tau)


def bootstrap_auc_pr_ci(y_true, y_score, n_boot: int = 1000, seed: int = 42,
                         alpha: float = 0.05):
    """Same as bootstrap_auc_ci but for Average Precision (AUC-PR)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)

    if y_true.sum() < 2:
        point = average_precision_score(y_true, y_score) if y_true.sum() > 0 else float("nan")
        return point, float("nan"), float("nan")

    point = average_precision_score(y_true, y_score)
    rng = np.random.RandomState(seed)
    boot_scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0:
            continue
        boot_scores.append(average_precision_score(yt, ys))

    if len(boot_scores) < n_boot // 2:
        return point, float("nan"), float("nan")

    lo = float(np.percentile(boot_scores, 100 * alpha / 2))
    hi = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))
    return float(point), round(lo, 4), round(hi, 4)
