"""
Model 2 — XGBoost + scale_pos_weight
Features: All 277 (186 original + 91 engineered)
Evaluated on: Train, OOS, Internal OOT, OOT Oct/Nov/Dec 2025

Phase 0 fixes applied (see ARCHITECTURE.md Section 9):
  - Duplicate dispute rows (10x repeated in the supplied data) are dropped
    before training or evaluating on any split.
  - The decision threshold is chosen ONCE on Validation OOS, then reused
    unchanged on every other split (including OOT). It is no longer
    re-optimised per dataset, which would otherwise flatter every set.
  - Each AUC-ROC is reported with a bootstrap 95% confidence interval, since
    the out-of-time months contain as few as 15-18 real disputes.
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import pickle
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score

sys.path.append("/Users/somyabhadada/nodex8")
from src.eval_utils import (
    dedupe_rows, select_f1_threshold, metrics_at_threshold, bootstrap_auc_ci,
)

# ── Paths ────────────────────────────────────────────────────────────
DATA_DIR   = "/Users/somyabhadada/nodex8/data/TradeFraud/Modeling_Data_Enriched/"
OUTPUT_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/Model_Outputs/Model2_XGBoost/"

# Metadata / non-feature columns
EXCLUDE_COLS = ["Client", "Client_ID", "Date", "Dispute",
                "Segment", "Activeness_Segment"]

SCALE_POS_WEIGHT = 44.60   # n_neg / n_pos in oversampled train

# ── Helpers ──────────────────────────────────────────────────────────
def load_dataset(fname):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df["target"] = (df["Dispute"] == "D").astype(int)
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != "target"]
    # Cast everything to numeric where possible
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = dedupe_rows(df, feat_cols)
    X = df[feat_cols].fillna(0)
    y = df["target"]
    return X, y, feat_cols

def ks_statistic(y_true, y_prob):
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False).reset_index(drop=True)
    df["cum_pos"] = df["y"].cumsum() / df["y"].sum()
    df["cum_neg"] = (1 - df["y"]).cumsum() / (1 - df["y"]).sum()
    return (df["cum_pos"] - df["cum_neg"]).abs().max()

def top_decile_recall(y_true, y_prob):
    n_decile = max(1, len(y_true) // 10)
    idx = np.argsort(y_prob)[::-1][:n_decile]
    return y_true.iloc[idx].sum() / max(1, y_true.sum())

def evaluate(model, X, y, dataset_name, threshold):
    """
    threshold must be chosen once on Validation OOS (see select_f1_threshold
    in the main script below) and passed in unchanged for every other split.
    """
    proba = model.predict_proba(X)[:, 1]
    auc_roc, auc_lo, auc_hi = bootstrap_auc_ci(y, proba)
    auc_pr  = average_precision_score(y, proba)
    gini    = 2 * auc_roc - 1
    ks      = ks_statistic(y, proba)
    tdr     = top_decile_recall(y, proba)
    thr_metrics = metrics_at_threshold(y, proba, threshold)

    return {
        "Dataset":           dataset_name,
        "Rows":              len(y),
        "Disputes":          int(y.sum()),
        "AUC_ROC":           round(auc_roc, 4),
        "AUC_ROC_CI_Low":    auc_lo,
        "AUC_ROC_CI_High":   auc_hi,
        "AUC_PR":            round(auc_pr, 4),
        "Gini":              round(gini, 4),
        "KS":                round(ks, 4),
        "Top_Decile_Recall": round(tdr, 4),
        **thr_metrics,
    }

# ── 1. Load training data ────────────────────────────────────────────
print("Loading enriched training data...")
X_train, y_train, feat_cols = load_dataset("train.csv")
print(f"  Train: {X_train.shape}  | Disputes: {y_train.sum()}")

# ── 2. Configure & train XGBoost ─────────────────────────────────────
print("\nTraining XGBoost Model 2...")
params = dict(
    n_estimators       = 500,
    max_depth          = 6,
    learning_rate      = 0.05,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    min_child_weight   = 5,
    reg_alpha          = 0.1,
    reg_lambda         = 1.0,
    scale_pos_weight   = SCALE_POS_WEIGHT,
    eval_metric        = "aucpr",
    early_stopping_rounds = 30,
    random_state       = 42,
    n_jobs             = -1,
    tree_method        = "hist",
)

from sklearn.model_selection import train_test_split

# Create an internal validation set strictly from the training data for early stopping
X_train_sub, X_val_internal, y_train_sub, y_val_internal = train_test_split(
    X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
)

model = XGBClassifier(**params)
model.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val_internal, y_val_internal)],
    verbose=50
)
print(f"\nBest iteration: {model.best_iteration}")

# ── 3. Save model & config ───────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
model_path = os.path.join(OUTPUT_DIR, "model2_xgboost.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

config = {k: (v if not callable(v) else str(v)) for k, v in params.items()}
config["best_iteration"] = model.best_iteration
config["n_features"]     = len(feat_cols)
config["feature_set"]    = "Enriched (186 original + 91 engineered = 277 total)"
with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w") as f:
    json.dump(config, f, indent=2)
print(f"Model saved → {model_path}")

# ── 4. Evaluate across all datasets ──────────────────────────────────
# Choose ONE decision threshold on Validation OOS, then reuse it, unchanged,
# on every other split (train and every OOT month). Re-optimising a
# threshold per dataset — as the original version of this script did — looks
# at each set's own labels before deciding, which makes precision/recall/F1
# look better than a single deployed threshold ever would.
print("\nSelecting decision threshold on Validation OOS...")
X_val, y_val, _ = load_dataset("val_oos.csv")
val_proba = model.predict_proba(X_val)[:, 1]
FIXED_THRESHOLD = select_f1_threshold(y_val, val_proba)
print(f"  Threshold fixed at {FIXED_THRESHOLD:.4f} (F1-maximising on Validation OOS only)")

print("\nEvaluating across all datasets...")
eval_sets = [
    ("train.csv",        "Train (oversampled)"),
    ("val_oos.csv",      "Validation OOS"),
    ("internal_oot.csv", "Internal OOT (Aug–Sep 2025)"),
    ("oot_202510.csv",   "OOT Oct 2025"),
    ("oot_202511.csv",   "OOT Nov 2025"),
    ("oot_202512.csv",   "OOT Dec 2025"),
]

results = []
for fname, label in eval_sets:
    X, y, _ = load_dataset(fname)
    res = evaluate(model, X, y, label, FIXED_THRESHOLD)
    results.append(res)
    ci_str = f"[{res['AUC_ROC_CI_Low']},{res['AUC_ROC_CI_High']}]"
    print(f"  {label:35s} | AUC={res['AUC_ROC']:.4f} {ci_str:18s} | KS={res['KS']:.4f} | "
          f"Gini={res['Gini']:.4f} | TopDecile={res['Top_Decile_Recall']:.4f} | F1={res['F1']:.4f}")

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(OUTPUT_DIR, "performance_scorecard.csv"), index=False)
print(f"\nScorecard saved → {OUTPUT_DIR}performance_scorecard.csv")

# ── 5. Feature importance (built-in + SHAP top-50) ───────────────────
print("\nComputing feature importance...")
fi = pd.DataFrame({
    "Feature":    feat_cols,
    "Importance_Gain":  model.feature_importances_,
}).sort_values("Importance_Gain", ascending=False).reset_index(drop=True)
fi["Rank"] = fi.index + 1
fi.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

print("\nTop 20 features by Gain:")
print(fi[["Rank", "Feature", "Importance_Gain"]].head(20).to_string(index=False))
print(f"\n✅ Model 2 complete. All outputs in {OUTPUT_DIR}")
