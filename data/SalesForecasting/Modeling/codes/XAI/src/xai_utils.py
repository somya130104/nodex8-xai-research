"""
Performance metric utilities for the XAI pipeline.
Supports both Regression and Classification model types.
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
)


def calculate_regression_metrics(actual, predicted, n_features=None, tolerance=0.2):
    """Regression metrics: MAE, RMSE, MAPE, R², Adj-R², confidence buckets."""
    actual    = np.array(actual,    dtype=float)
    predicted = np.array(predicted, dtype=float)

    mask = actual != 0
    actual_m, predicted_m = actual[mask], predicted[mask]

    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual_m - predicted_m) / actual_m)) if mask.any() else np.nan
    acc  = max(0.0, 1.0 - mape) if not np.isnan(mape) else 0.0

    r2    = r2_score(actual, predicted)
    adj_r2 = None
    if n_features and len(actual) > n_features + 1:
        n = len(actual)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    hits      = np.abs(actual - predicted) <= (tolerance * np.abs(actual))
    precision = np.mean(hits)
    recall    = precision
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    rel_err       = np.abs((actual_m - predicted_m) / actual_m) if mask.any() else np.array([])
    high_pct      = np.sum(rel_err <= 0.10) / len(actual)
    med_pct       = np.sum((rel_err > 0.10) & (rel_err <= 0.25)) / len(actual)
    low_pct       = 1.0 - high_pct - med_pct

    return {
        "MAE"                 : round(mae, 6),
        "RMSE"                : round(rmse, 6),
        "MAPE"                : round(float(mape), 6) if not np.isnan(mape) else None,
        "Accuracy %"          : round(acc * 100, 4),
        "Precision %"         : round(precision * 100, 4),
        "Recall %"            : round(recall * 100, 4),
        "F1 Score %"          : round(f1 * 100, 4),
        "R-Squared"           : round(r2, 6),
        "Adjusted R-Squared"  : round(adj_r2, 6) if adj_r2 is not None else None,
        "High Confidence %"   : round(high_pct * 100, 4),
        "Medium Confidence %" : round(med_pct * 100, 4),
        "Low Confidence %"    : round(low_pct * 100, 4),
    }


def calculate_classification_metrics(actual, score, pred, n_features=None, tolerance=None):
    """
    Classification metrics for binary outcomes.

    Parameters
    ----------
    actual : array-like  — true binary labels (0/1)
    score  : array-like  — predicted probability for the positive class
    pred   : array-like  — predicted binary label (0/1, threshold usually 0.5)
    n_features : int     — unused, kept for API symmetry with regression version
    tolerance  : float   — unused, kept for API symmetry

    Returns
    -------
    dict with: AUC, Gini, KS, Accuracy %, Precision %, Recall %, F1 Score %,
               TDR@D1 %, TDR@D2 %, Fraud Capture @10 %
    """
    actual = np.array(actual, dtype=int)
    score  = np.array(score,  dtype=float)
    pred   = np.array(pred,   dtype=int)

    n_pos  = actual.sum()

    # AUC / Gini
    try:
        auc  = roc_auc_score(actual, score) if n_pos > 0 and n_pos < len(actual) else np.nan
    except Exception:
        auc  = np.nan
    gini = 2 * auc - 1 if not np.isnan(auc) else np.nan

    # KS statistic
    try:
        order    = np.argsort(-score)
        cum_pos  = np.cumsum(actual[order]) / max(n_pos, 1)
        cum_neg  = np.cumsum(1 - actual[order]) / max(len(actual) - n_pos, 1)
        ks       = float(np.max(np.abs(cum_pos - cum_neg)))
    except Exception:
        ks = np.nan

    # Standard classification metrics at pred threshold
    try:
        acc  = accuracy_score(actual, pred)
        prec = precision_score(actual, pred, zero_division=0)
        rec  = recall_score(actual, pred, zero_division=0)
        f1   = f1_score(actual, pred, zero_division=0)
    except Exception:
        acc = prec = rec = f1 = np.nan

    # TDR (True Dispute Rate = % frauds captured) at top deciles
    def _tdr_at_top_pct(pct):
        if n_pos == 0:
            return np.nan
        n_top = max(1, int(np.ceil(len(actual) * pct)))
        top_idx = np.argsort(-score)[:n_top]
        return float(actual[top_idx].sum()) / n_pos

    tdr_d1 = _tdr_at_top_pct(0.10)   # top 10% → D1
    tdr_d2 = _tdr_at_top_pct(0.20)   # top 20% → D1+D2

    return {
        "AUC"                 : round(float(auc), 6) if not np.isnan(auc) else None,
        "Gini"                : round(float(gini), 6) if not np.isnan(gini) else None,
        "KS"                  : round(float(ks), 6) if not np.isnan(ks) else None,
        "Accuracy %"          : round(acc * 100, 4),
        "Precision %"         : round(prec * 100, 4),
        "Recall %"            : round(rec * 100, 4),
        "F1 Score %"          : round(f1 * 100, 4),
        "TDR@Top10 %"         : round(tdr_d1 * 100, 4) if not np.isnan(tdr_d1) else None,
        "TDR@Top20 %"         : round(tdr_d2 * 100, 4) if not np.isnan(tdr_d2) else None,
        # Regression-only columns — fill None so CSV schema is uniform
        "MAE"                 : None,
        "RMSE"                : None,
        "MAPE"                : None,
        "R-Squared"           : None,
        "Adjusted R-Squared"  : None,
        "High Confidence %"   : None,
        "Medium Confidence %" : None,
        "Low Confidence %"    : None,
    }
