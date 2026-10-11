"""
Model 7 — TabNet Attentive Interpretable Tabular Classifier
===========================================================
Deep neural network architecture with sequential attention mechanism (Sparsemax)
for tabular dispute/fraud classification.

Outputs:
  - Model weights / checkpoint
  - Performance scorecard (AUC, Gini, KS, Top-Decile Recall, F1)
  - Native feature importances & mask attributions
"""
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score, confusion_matrix)

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
    y = (df["Dispute"] == "D").astype(int).values
    if feat_cols is None:
        feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feat_cols].fillna(0).values
    return X, y, feat_cols

def evaluate(y_true, y_score, label):
    n_pos = int(y_true.sum())
    try:
        auc = roc_auc_score(y_true, y_score) if 0 < n_pos < len(y_true) else np.nan
    except Exception:
        auc = np.nan
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
    
    # Best threshold by F1
    thresholds = np.linspace(0.01, 0.99, 99)
    best_th, best_f1, best_prec, best_rec = 0.5, 0.0, 0.0, 0.0
    for th in thresholds:
        pred = (y_score >= th).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_th = th
            best_prec = precision_score(y_true, pred, zero_division=0)
            best_rec = recall_score(y_true, pred, zero_division=0)
            
    pred_best = (y_score >= best_th).astype(int)
    cm = confusion_matrix(y_true, pred_best)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    return {
        "Dataset": label,
        "Rows": len(y_true),
        "Disputes": n_pos,
        "AUC_ROC": round(auc, 4) if not np.isnan(auc) else None,
        "AUC_PR": round(auc_pr, 4) if not np.isnan(auc_pr) else None,
        "Gini": round(gini, 4) if not np.isnan(gini) else None,
        "KS": round(ks, 4) if not np.isnan(ks) else None,
        "Top_Decile_Recall": round(top_decile_recall, 4),
        "Best_Threshold": round(best_th, 4),
        "Precision": round(best_prec, 4),
        "Recall": round(best_rec, 4),
        "F1": round(best_f1, 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
    }

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading data for TabNet...")
    X_train, y_train, feat_cols = load_data("train.csv")
    X_val, y_val, _ = load_data("val_oos.csv", feat_cols)
    
    print(f"Features: {len(feat_cols)}, Train rows: {len(y_train)} (Disputes: {y_train.sum()})")
    
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
    
    print("Training TabNet...")
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=['val'],
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
        
    print("\nEvaluating across sets...")
    scorecards = []
    for fname, label in EVAL_SETS:
        X_eval, y_eval, _ = load_data(fname, feat_cols)
        preds = clf.predict_proba(X_eval)[:, 1]
        metrics = evaluate(y_eval, preds, label)
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
