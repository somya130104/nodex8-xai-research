"""
Model 6 — Logistic Regression (Elastic Net)
Features: All 277 (186 original + 91 engineered)
Evaluated on: Train, OOS, Internal OOT, OOT Oct/Nov/Dec 2025

Phase 0 fixes applied (see ARCHITECTURE.md Section 9): duplicate dispute rows
are dropped before training or evaluation; the decision threshold is chosen
once on Validation OOS and reused unchanged everywhere else; AUC-ROC is
reported with a bootstrap 95% confidence interval.
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import pickle
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

sys.path.append("/Users/somyabhadada/nodex8")
from src.eval_utils import (
    dedupe_rows, select_f1_threshold, metrics_at_threshold, bootstrap_auc_ci,
)

# ── Paths ────────────────────────────────────────────────────────────
DATA_DIR   = "/Users/somyabhadada/nodex8/data/TradeFraud/Modeling_Data_Enriched/"
OUTPUT_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/Model_Outputs/Model6_LogReg/"

# Metadata / non-feature columns
EXCLUDE_COLS = ["Client", "Client_ID", "Date", "Dispute",
                "Segment", "Activeness_Segment"]

# ── Helpers ──────────────────────────────────────────────────────────
def load_dataset(fname):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df["target"] = (df["Dispute"] == "D").astype(int)
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != "target"]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = dedupe_rows(df, feat_cols)
    X = df[feat_cols].fillna(0)
    y = df["target"]
    return X, y, feat_cols

def ks_statistic(y_true, y_prob):
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False).reset_index(drop=True)
    df["cum_pos"] = df["y"].cumsum() / max(1, df["y"].sum())
    df["cum_neg"] = (1 - df["y"]).cumsum() / max(1, (1 - df["y"]).sum())
    return (df["cum_pos"] - df["cum_neg"]).abs().max()

def top_decile_recall(y_true, y_prob):
    n_decile = max(1, len(y_true) // 10)
    idx = np.argsort(y_prob)[::-1][:n_decile]
    return y_true.iloc[idx].sum() / max(1, y_true.sum())

def evaluate(model, scaler, X, y, dataset_name, threshold):
    """threshold must be chosen once on Validation OOS and reused unchanged elsewhere."""
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, 1]
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

print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ── 2. Configure & train Logistic Regression ─────────────────────────
print("\nTraining Logistic Regression (Elastic Net) Model 6...")
# saga solver supports elasticnet penalty
params = dict(
    penalty='elasticnet',
    l1_ratio=0.5,
    C=0.1,
    class_weight='balanced',
    solver='saga',
    max_iter=500,
    random_state=42,
    n_jobs=-1
)

model = LogisticRegression(**params)
model.fit(X_train_scaled, y_train)

# ── 3. Save model & config ───────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
model_path = os.path.join(OUTPUT_DIR, "model6_logreg.pkl")
with open(model_path, "wb") as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)

config = {k: str(v) for k, v in params.items()}
config["n_features"]     = len(feat_cols)
config["feature_set"]    = "Enriched (186 original + 91 engineered = 277 total)"
with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w") as f:
    json.dump(config, f, indent=2)
print(f"Model saved → {model_path}")

# ── 4. Evaluate across all datasets ──────────────────────────────────
# Choose ONE decision threshold on Validation OOS, then reuse it unchanged on
# every other split — see module docstring.
print("\nSelecting decision threshold on Validation OOS...")
X_val, y_val, _ = load_dataset("val_oos.csv")
val_proba = model.predict_proba(scaler.transform(X_val))[:, 1]
FIXED_THRESHOLD = select_f1_threshold(y_val, val_proba)
print(f"  Threshold fixed at {FIXED_THRESHOLD:.4f} (F1-maximising on Validation OOS only)")

print("\nEvaluating across all datasets...")
eval_sets = [
    ("train.csv",        "Train (balanced)"),
    ("val_oos.csv",      "Validation OOS"),
    ("internal_oot.csv", "Internal OOT (Aug–Sep 2025)"),
    ("oot_202510.csv",   "OOT Oct 2025"),
    ("oot_202511.csv",   "OOT Nov 2025"),
    ("oot_202512.csv",   "OOT Dec 2025"),
]

results = []
for fname, label in eval_sets:
    X, y, _ = load_dataset(fname)
    res = evaluate(model, scaler, X, y, label, FIXED_THRESHOLD)
    results.append(res)
    print(f"  {label:35s} | AUC={res['AUC_ROC']:.4f} | KS={res['KS']:.4f} | "
          f"Gini={res['Gini']:.4f} | TopDecile={res['Top_Decile_Recall']:.4f} | F1={res['F1']:.4f}")

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(OUTPUT_DIR, "performance_scorecard.csv"), index=False)
print(f"\nScorecard saved → {OUTPUT_DIR}performance_scorecard.csv")

# ── 5. Feature importance (Coefficients) ─────────────────────────────
print("\nComputing feature importance (absolute coefficients)...")
fi = pd.DataFrame({
    "Feature":    feat_cols,
    "Coefficient": model.coef_[0],
    "Abs_Coefficient": np.abs(model.coef_[0])
}).sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)
fi["Rank"] = fi.index + 1
fi.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

print("\nTop 20 features by Absolute Coefficient:")
print(fi[["Rank", "Feature", "Coefficient", "Abs_Coefficient"]].head(20).to_string(index=False))
print(f"\n✅ Model 6 complete. All outputs in {OUTPUT_DIR}")
