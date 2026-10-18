"""
Model 7 — TabNet Attentive Interpretable Tabular Classifier
===========================================================
Deep neural network architecture with sequential attention mechanism (Sparsemax)
for tabular dispute/fraud classification.

Outputs:
  - Model weights / checkpoint
  - Performance scorecard (AUC, Gini, KS, Top-Decile Recall, F1)
  - Native feature importances & mask attributions

Phase 0 fixes applied (see ARCHITECTURE.md Section 9):
  - Duplicate dispute rows are dropped before training or evaluation.
  - Early stopping previously used Validation OOS directly as eval_set, then
    scored the model on that same Validation OOS in the final scorecard —
    a leak that makes the reported validation AUC optimistic. Early stopping
    now uses an internal split carved from training data only (matching the
    XGBoost/LightGBM scripts); Validation OOS is held out for evaluation and
    threshold selection only.
  - The decision threshold is chosen once on Validation OOS and reused
    unchanged on every other split. AUC-ROC is reported with a bootstrap 95%
    confidence interval.
"""
import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

sys.path.append("/Users/somyabhadada/nodex8")
from src.eval_utils import (
    dedupe_rows, select_f1_threshold, metrics_at_threshold, bootstrap_auc_ci,
)

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────
DATA_DIR   = "/Users/somyabhadada/nodex8/data/TradeFraud/Modeling_Data_Enriched/"
OUTPUT_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/Model_Outputs/Model7_TabNet/"

EXCLUDE_COLS = ["Client", "Client_ID", "Date", "Dispute", "Segment", "Activeness_Segment"]
EVAL_SETS = [
    ("train.csv",        "Train (all)"),
    ("val_oos.csv",      "Validation OOS"),
    ("internal_oot.csv", "Internal OOT (Aug–Sep)"),
    ("oot_202510.csv",   "OOT Oct 2025"),
    ("oot_202511.csv",   "OOT Nov 2025"),
    ("oot_202512.csv",   "OOT Dec 2025"),
]

def load_data(fname, feat_cols=None):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    if feat_cols is None:
        feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    df = dedupe_rows(df, feat_cols)
    y = (df["Dispute"] == "D").astype(int).values
    X = df[feat_cols].fillna(0).values
    return X, y, feat_cols

def evaluate(y_true, y_score, label, threshold):
    """threshold must be chosen once on Validation OOS and reused unchanged elsewhere."""
    n_pos = int(y_true.sum())
    auc, auc_lo, auc_hi = bootstrap_auc_ci(y_true, y_score)
    gini = 2 * auc - 1 if not np.isnan(auc) else np.nan

    try:
        auc_pr = average_precision_score(y_true, y_score) if 0 < n_pos < len(y_true) else np.nan
    except Exception:
        auc_pr = np.nan

    # KS
    try:
        order = np.argsort(-y_score)
        cum_pos = np.cumsum(y_true[order]) / max(n_pos, 1)
        cum_neg = np.cumsum(1 - y_true[order]) / max(len(y_true) - n_pos, 1)
        ks = float(np.max(np.abs(cum_pos - cum_neg)))
    except Exception:
        ks = np.nan

    # Top decile recall
    n_top = max(1, int(len(y_true) * 0.10))
    top_idx = np.argsort(-y_score)[:n_top]
    top_decile_recall = float(y_true[top_idx].sum() / max(n_pos, 1))

    thr_metrics = metrics_at_threshold(y_true, y_score, threshold)

    return {
        "Dataset": label,
        "Rows": len(y_true),
        "Disputes": n_pos,
        "AUC_ROC": round(auc, 4) if not np.isnan(auc) else None,
        "AUC_ROC_CI_Low": auc_lo,
        "AUC_ROC_CI_High": auc_hi,
        "AUC_PR": round(auc_pr, 4) if not np.isnan(auc_pr) else None,
        "Gini": round(gini, 4) if not np.isnan(gini) else None,
        "KS": round(ks, 4) if not np.isnan(ks) else None,
        "Top_Decile_Recall": round(top_decile_recall, 4),
        **thr_metrics,
    }

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading data for TabNet...")
    X_train_full, y_train_full, feat_cols = load_data("train.csv")

    # Internal split for early stopping, carved from training data ONLY.
    # Validation OOS must never be used for early stopping — it is the set
    # we later report the model's performance on.
    X_train, X_train_val, y_train, y_train_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1,
        stratify=y_train_full, random_state=42,
    )

    print(f"Features: {len(feat_cols)}, Train rows: {len(y_train)} (Disputes: {y_train.sum()}) "
          f"| internal early-stopping rows: {len(y_train_val)} (Disputes: {y_train_val.sum()})")

    # Calculate pos_weight for class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    weights = {0: 1.0, 1: float(n_neg / max(n_pos, 1))}
    
    print("\nInitialising TabNetClassifier...")
    clf = TabNetClassifier(
        n_d=32,
        n_a=32,
        n_steps=5,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        momentum=0.02,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='sparsemax',
        verbose=10
    )
    
    print("Training TabNet (early stopping on an internal split of train.csv only)...")
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train_val, y_train_val)],
        eval_name=['internal_val'],
        eval_metric=['auc'],
        max_epochs=40,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        weights=weights
    )

    # Save model
    clf.save_model(os.path.join(OUTPUT_DIR, "tabnet_model.zip"))
    with open(os.path.join(OUTPUT_DIR, "features.json"), "w") as f:
        json.dump(feat_cols, f, indent=4)

    # Choose ONE decision threshold on Validation OOS (never seen during
    # training or early stopping), then reuse it unchanged on every other
    # split, including train and OOT.
    X_val, y_val, _ = load_data("val_oos.csv", feat_cols)
    val_preds = clf.predict_proba(X_val)[:, 1]
    FIXED_THRESHOLD = select_f1_threshold(y_val, val_preds)
    print(f"\nThreshold fixed at {FIXED_THRESHOLD:.4f} (F1-maximising on Validation OOS only)")

    print("\nEvaluating across sets...")
    scorecards = []
    for fname, label in EVAL_SETS:
        X_eval, y_eval, _ = load_data(fname, feat_cols)
        preds = clf.predict_proba(X_eval)[:, 1]
        metrics = evaluate(y_eval, preds, label, FIXED_THRESHOLD)
        scorecards.append(metrics)
        print(f"  {label:25s}: AUC={metrics['AUC_ROC']}, Gini={metrics['Gini']}, Top10_Recall={metrics['Top_Decile_Recall']}")
        
    df_score = pd.DataFrame(scorecards)
    df_score.to_csv(os.path.join(OUTPUT_DIR, "performance_scorecard.csv"), index=False)
    
    # Feature Importance
    importances = clf.feature_importances_
    df_fi = pd.DataFrame({
        "Feature": feat_cols,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    df_fi.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    print("\nTop 10 features by TabNet attention:")
    print(df_fi.head(10).to_string(index=False))
    
    print(f"\n✅ Model 7 TabNet complete! Outputs saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    run()
