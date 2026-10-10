"""
Model 3 — Random Forest + SMOTE
Features: All 271 enriched numeric features (same as Model 2)
Oversampling: SMOTE applied ONLY to training data (val/OOT never seen)
Outputs: Performance scorecard, ROC curve, Hosmer-Lemeshow chart + data
"""
import os, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, f1_score, precision_score,
                             recall_score, confusion_matrix)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────
DATA_DIR   = "/Users/somyabhadada/nodex8/data/TradeFraud/Modeling_Data_Enriched/"
OUTPUT_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/Model_Outputs/Model3_RandomForest/"
ART_DIR    = "/Users/somyabhadada/.gemini/antigravity-ide/brain/1d5e1c27-d371-4ffb-a6a7-9cb908b0122b/"

EXCLUDE_COLS = ["Client", "Client_ID", "Date", "Dispute", "Segment", "Activeness_Segment"]

EVAL_SETS = [
    ("train.csv",        "Train (SMOTE)"),
    ("val_oos.csv",      "Validation OOS"),
    ("internal_oot.csv", "Internal OOT (Aug–Sep)"),
    ("oot_202510.csv",   "OOT Oct 2025"),
    ("oot_202511.csv",   "OOT Nov 2025"),
    ("oot_202512.csv",   "OOT Dec 2025"),
]

# ── Helpers ──────────────────────────────────────────────────────────
def load_dataset(fname, feat_cols=None):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df["target"] = (df["Dispute"] == "D").astype(int)
    if feat_cols is None:
        feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != "target"]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    X = df[feat_cols].fillna(0)
    y = df["target"]
    return X, y, feat_cols

def ks_statistic(y_true, y_prob):
    df = pd.DataFrame({"y": y_true.values, "p": y_prob})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["cum_pos"] = df["y"].cumsum() / df["y"].sum()
    df["cum_neg"] = (1 - df["y"]).cumsum() / (1 - df["y"]).sum()
    return (df["cum_pos"] - df["cum_neg"]).abs().max()

def top_decile_recall(y_true, y_prob):
    n = max(1, len(y_true) // 10)
    idx = np.argsort(y_prob)[::-1][:n]
    return y_true.iloc[idx].sum() / max(1, y_true.sum())

def evaluate(model, X, y, label, feat_cols):
    proba = model.predict_proba(X)[:, 1]
    auc   = roc_auc_score(y, proba)
    aupr  = average_precision_score(y, proba)
    ks    = ks_statistic(y, proba)
    tdr   = top_decile_recall(y, proba)
    gini  = 2 * auc - 1
    thresholds = np.linspace(0.01, 0.99, 200)
    best_f1, best_thr = 0, 0.5
    for t in thresholds:
        pred = (proba >= t).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1: best_f1, best_thr = f, t
    pred_opt = (proba >= best_thr).astype(int)
    prec = precision_score(y, pred_opt, zero_division=0)
    rec  = recall_score(y, pred_opt, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred_opt).ravel()
    return {
        "Dataset": label, "Rows": len(y), "Disputes": int(y.sum()),
        "AUC_ROC": round(auc,4), "AUC_PR": round(aupr,4),
        "Gini": round(gini,4), "KS": round(ks,4),
        "Top_Decile_Recall": round(tdr,4), "Best_Threshold": round(best_thr,4),
        "Precision": round(prec,4), "Recall": round(rec,4), "F1": round(best_f1,4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
    }, proba

def hosmer_lemeshow(y_true, y_prob, bins=10):
    df = pd.DataFrame({"y": y_true.values, "p": y_prob})
    try:
        df["decile"] = pd.qcut(df["p"].rank(method="first"), q=bins, labels=False)
    except Exception:
        df["decile"] = pd.cut(df["p"], bins=bins, labels=False)
    hl = df.groupby("decile").agg(Total=("y","count"),
                                   Obs_Event=("y","sum"),
                                   Exp_Event=("p","sum")).reset_index()
    hl["Obs_NonEvent"] = hl["Total"] - hl["Obs_Event"]
    hl["Exp_NonEvent"] = hl["Total"] - hl["Exp_Event"]
    eps = 1e-10
    stat = np.sum(
        ((hl["Obs_Event"] - hl["Exp_Event"])**2 / (hl["Exp_Event"] + eps)) +
        ((hl["Obs_NonEvent"] - hl["Exp_NonEvent"])**2 / (hl["Exp_NonEvent"] + eps))
    )
    dof = max(1, len(hl) - 2)
    pval = 1 - chi2.cdf(stat, dof)
    hl["Decile_Rank"] = hl["decile"] + 1
    return stat, pval, hl

# ── 1. SMOTE on training data only ───────────────────────────────────
print("Step 1: Loading training data...")
X_train_raw, y_train_raw, feat_cols = load_dataset("train.csv")
print(f"  Before SMOTE → Shape: {X_train_raw.shape} | "
      f"Disputes: {y_train_raw.sum()} | Non-disputes: {(y_train_raw==0).sum()}")

print("\nStep 2: Applying SMOTE (sampling_strategy=0.2)...")
smote = SMOTE(sampling_strategy=0.2, k_neighbors=5, random_state=42)
X_smote, y_smote = smote.fit_resample(X_train_raw, y_train_raw)
print(f"  After SMOTE  → Shape: {X_smote.shape} | "
      f"Disputes: {y_smote.sum()} | Non-disputes: {(y_smote==0).sum()}")
print(f"  New class ratio: 1:{(y_smote==0).sum() / y_smote.sum():.1f}")

# ── 2. Train Random Forest ────────────────────────────────────────────
print("\nStep 3: Training Random Forest...")
rf_params = dict(
    n_estimators   = 400,
    max_depth      = 20,
    min_samples_leaf = 10,
    min_samples_split = 20,
    max_features   = "sqrt",
    oob_score      = True,
    class_weight   = None,   # SMOTE already handles imbalance
    random_state   = 42,
    n_jobs         = -1,
)

model = RandomForestClassifier(**rf_params)
model.fit(X_smote, y_smote)
print(f"  OOB Score (internal estimate): {model.oob_score_:.4f}")

# ── 3. Save model & config ───────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)

model_path = os.path.join(OUTPUT_DIR, "model3_rf.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

config = {k: (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v)
          for k, v in rf_params.items()}
config["oob_score_achieved"] = round(model.oob_score_, 4)
config["n_features"]         = len(feat_cols)
config["smote_strategy"]     = 0.2
config["train_post_smote_rows"] = len(X_smote)
config["train_disputes_post_smote"] = int(y_smote.sum())
with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w") as f:
    json.dump(config, f, indent=2)
print(f"  Model saved → {model_path}")

# ── 4. Evaluate & collect probabilities ──────────────────────────────
print("\nStep 4: Evaluating on all datasets...")
results, probas_dict, y_dict = [], {}, {}
for fname, label in EVAL_SETS:
    X, y, _ = load_dataset(fname, feat_cols)
    res, proba = evaluate(model, X, y, label, feat_cols)
    results.append(res)
    probas_dict[label] = proba
    y_dict[label]      = y
    print(f"  {label:35s} | AUC={res['AUC_ROC']:.4f} | KS={res['KS']:.4f} | "
          f"Gini={res['Gini']:.4f} | TopDecile={res['Top_Decile_Recall']:.4f} | F1={res['F1']:.4f}")

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(OUTPUT_DIR, "performance_scorecard.csv"), index=False)

# ── 5. Feature Importance ────────────────────────────────────────────
print("\nStep 5: Feature importance...")
fi = pd.DataFrame({"Feature": feat_cols,
                   "Importance_MDA": model.feature_importances_}
                  ).sort_values("Importance_MDA", ascending=False).reset_index(drop=True)
fi["Rank"] = fi.index + 1
fi.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
print("Top 20 features:")
print(fi[["Rank","Feature","Importance_MDA"]].head(20).to_string(index=False))

# ── 6. ROC Curves ───────────────────────────────────────────────────
print("\nStep 6: Plotting ROC curves...")
COLORS = ["#2c3e50","#3498db","#e74c3c","#27ae60","#f39c12","#9b59b6"]
STYLES = ["-","--","-.",":","--","-."]

fig, ax = plt.subplots(figsize=(10, 8))
for (fname, label), col, ls in zip(EVAL_SETS, COLORS, STYLES):
    fpr, tpr, _ = roc_curve(y_dict[label], probas_dict[label])
    auc = roc_auc_score(y_dict[label], probas_dict[label])
    lw = 2.5 if fname == "train.csv" else 2.0
    ax.plot(fpr, tpr, lw=lw, color=col, linestyle=ls,
            label=f"{label} (AUC = {auc:.3f})")

ax.plot([0,1],[0,1], "k--", lw=1.2, label="Random Classifier")
ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Model 3 (Random Forest + SMOTE)", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(ART_DIR, "roc_chart_model3.png")
roc_local = os.path.join(OUTPUT_DIR, "roc_chart_model3.png")
plt.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.savefig(roc_local, dpi=300, bbox_inches="tight")
plt.close()
print(f"  ROC chart saved.")

# ── 7. Hosmer-Lemeshow ───────────────────────────────────────────────
print("\nStep 7: Hosmer-Lemeshow test...")
fig, axes = plt.subplots(3, 2, figsize=(15, 13))
axes = axes.flatten()
all_hl_data, hl_summary = [], []

for i, (fname, label) in enumerate(EVAL_SETS):
    stat, pval, hl_df = hosmer_lemeshow(y_dict[label], probas_dict[label])
    hl_df["Dataset"] = label
    all_hl_data.append(hl_df)
    hl_summary.append({"Dataset": label, "HL_Statistic": round(stat,2),
                        "P_Value": round(pval,6), "Bins": len(hl_df)})

    ax = axes[i]
    x = hl_df["Decile_Rank"]
    w = 0.35
    ax.bar(x - w/2, hl_df["Exp_Event"], w, label="Expected (Model)", color="#3498db", alpha=0.85)
    ax.bar(x + w/2, hl_df["Obs_Event"], w, label="Observed (Actual)", color="#e74c3c", alpha=0.85)
    ax.set_title(f"{label}\nHL={stat:.1f}  p={pval:.4f}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Decile (Low → High Risk)", fontsize=9)
    ax.set_ylabel("# Disputes", fontsize=9)
    ax.set_xticks(x)
    if i == 0: ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Hosmer-Lemeshow Calibration — Model 3 (Random Forest + SMOTE)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
hl_path  = os.path.join(ART_DIR, "hosmer_lemeshow_chart_model3.png")
hl_local = os.path.join(OUTPUT_DIR, "hosmer_lemeshow_chart_model3.png")
plt.savefig(hl_path, dpi=300, bbox_inches="tight")
plt.savefig(hl_local, dpi=300, bbox_inches="tight")
plt.close()

# Save HL data
hl_data_df = pd.concat(all_hl_data, ignore_index=True)[
    ["Dataset","Decile_Rank","Total","Obs_Event","Exp_Event","Obs_NonEvent","Exp_NonEvent"]]
hl_data_df.to_csv(os.path.join(OUTPUT_DIR, "hosmer_lemeshow_data.csv"), index=False)

# Print HL summary
print("\nHosmer-Lemeshow Summary:")
print(pd.DataFrame(hl_summary).to_string(index=False))
print(f"\n✅ Model 3 complete. All outputs in {OUTPUT_DIR}")
