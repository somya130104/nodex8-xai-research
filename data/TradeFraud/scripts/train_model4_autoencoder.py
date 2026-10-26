"""
Model 4 — Autoencoder Anomaly Detection
=========================================
Core idea:   Train ONLY on non-dispute (normal) transactions.
             Disputes → high reconstruction error → anomaly score.

Advantages:  Sidesteps oversampling entirely; learns what "normal" looks like
             and flags deviations — ideal for 0.23% dispute rate.

Outputs:     Performance scorecard, ROC curve, HL calibration chart,
             per-feature reconstruction error importance, model artefacts.

Phase 0 fixes applied (see ARCHITECTURE.md Section 9): duplicate dispute rows
are dropped before training or evaluation; the decision threshold is chosen
once on Validation OOS and reused unchanged everywhere else; AUC-ROC is
reported with a bootstrap 95% confidence interval. This model's own
early-stopping split was already drawn strictly from normal training rows
(never from Validation OOS), so no leakage fix was needed there.

Phase 2.1 fixes applied (see ARCHITECTURE.md Section 7 item 9 / ROADMAP.md
Phase 2.1):
  1. Preprocessing: heavy-tailed amount/count columns (221 of 271 had
     |skew| > 2 on normal rows) are now signed-log1p-transformed before
     scaling, and scaled values are clipped to [-5, 5] — see
     src/autoencoder_preprocessing.py. Previously, a handful of large-valued
     raw columns dominated the mean-squared reconstruction error, which is
     very likely why the old AUC sat at ~0.53.
  2. Architecture: the bottleneck layer is now a plain nn.Linear with no
     BatchNorm/Dropout/activation on it — Phase 3's LRD method needs a
     well-defined, deterministic latent vector z to back-project a Jacobian
     through; Dropout on the bottleneck would make z different on every
     forward pass and break that.
  3. encode(x) and decode(z) are now exposed as separate callable methods
     (not just a combined forward()), again for Phase 3's LRD.

Honest framing (also in ARCHITECTURE.md): this model's job in this research
is to be explained, not to maximise fraud-catching AUC. An explainer is
judged on whether it faithfully describes what the model actually did, not
on whether the model is a good fraud detector. Whatever AUC this version
achieves is reported as-is, with no target to hit.
"""
import os, sys, json, pickle, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("/Users/somyabhadada/nodex8")
from src.eval_utils import (
    dedupe_rows, select_f1_threshold, metrics_at_threshold, bootstrap_auc_ci,
)
from src.autoencoder_preprocessing import AutoencoderPreprocessor
from src.autoencoder_models import Autoencoder

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────
DATA_DIR   = "/Users/somyabhadada/nodex8/data/TradeFraud/Modeling_Data_Enriched/"
OUTPUT_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/Model_Outputs/Model4_Autoencoder/"

EXCLUDE_COLS = ["Client", "Client_ID", "Date", "Dispute", "Segment", "Activeness_Segment"]
EVAL_SETS = [
    ("train.csv",        "Train (all)"),
    ("val_oos.csv",      "Validation OOS"),
    ("internal_oot.csv", "Internal OOT (Aug–Sep)"),
    ("oot_202510.csv",   "OOT Oct 2025"),
    ("oot_202511.csv",   "OOT Nov 2025"),
    ("oot_202512.csv",   "OOT Dec 2025"),
]
HIDDEN  = [128, 64, 32]    # Encoder dims; decoder mirrors. 32 is the bottleneck.
EPOCHS  = 60
BATCH   = 512
LR      = 1e-3
PATIENCE = 10              # Early stopping patience

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Autoencoder class now lives in src/autoencoder_models.py (importable by Phase 3's explainers)

# ── Helpers ───────────────────────────────────────────────────────────
def load_features(fname, feat_cols=None):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df["target"] = (df["Dispute"] == "D").astype(int)
    if feat_cols is None:
        feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != "target"]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = dedupe_rows(df, feat_cols)
    X = df[feat_cols].fillna(0)
    y = df["target"].values
    return X, y, feat_cols

def get_recon_error(model, X_tensor):
    model.eval()
    with torch.no_grad():
        recon = model(X_tensor)
        errors = torch.mean((X_tensor - recon) ** 2, dim=1).numpy()
    return errors

def normalize_to_prob(errors, ref_min, ref_max):
    """Normalize reconstruction errors → [0,1] pseudo-probabilities."""
    clipped = np.clip(errors, ref_min, ref_max)
    return (clipped - ref_min) / (ref_max - ref_min + 1e-10)

def ks_stat(y, scores):
    df = pd.DataFrame({"y": y, "s": scores}).sort_values("s", ascending=False).reset_index(drop=True)
    df["cp"] = df["y"].cumsum() / df["y"].sum()
    df["cn"] = (1 - df["y"]).cumsum() / (1 - df["y"]).sum()
    return (df["cp"] - df["cn"]).abs().max()

def top_decile_recall(y, scores):
    n = max(1, len(y) // 10)
    idx = np.argsort(scores)[::-1][:n]
    return y[idx].sum() / max(1, y.sum())

def evaluate(y, proba, threshold):
    """threshold must be chosen once on Validation OOS and reused unchanged elsewhere."""
    auc, auc_lo, auc_hi = bootstrap_auc_ci(y, proba)
    aupr = average_precision_score(y, proba)
    ks   = ks_stat(y, proba)
    tdr  = top_decile_recall(y, proba)
    gini = 2 * auc - 1
    thr_metrics = metrics_at_threshold(y, proba, threshold)
    return {"AUC_ROC": round(auc,4), "AUC_ROC_CI_Low": auc_lo, "AUC_ROC_CI_High": auc_hi,
            "AUC_PR": round(aupr,4), "Gini": round(gini,4), "KS": round(ks,4),
            "Top_Decile_Recall": round(tdr,4), **thr_metrics}, proba

def hosmer_lemeshow(y, proba, bins=10):
    df = pd.DataFrame({"y": y, "p": proba})
    try:
        df["decile"] = pd.qcut(df["p"].rank(method="first"), q=bins, labels=False)
    except Exception:
        df["decile"] = pd.cut(df["p"], bins=bins, labels=False)
    hl = df.groupby("decile").agg(
        Total=("y","count"), Obs_E=("y","sum"), Exp_E=("p","sum")).reset_index()
    hl["Obs_NE"] = hl["Total"] - hl["Obs_E"]
    hl["Exp_NE"] = hl["Total"] - hl["Exp_E"]
    eps = 1e-10
    stat = float(np.sum(
        ((hl["Obs_E"] - hl["Exp_E"])**2 / (hl["Exp_E"] + eps)) +
        ((hl["Obs_NE"] - hl["Exp_NE"])**2 / (hl["Exp_NE"] + eps))
    ))
    pval = 1 - chi2.cdf(stat, max(1, len(hl) - 2))
    hl["Decile_Rank"] = hl["decile"] + 1
    return stat, pval, hl

# ─────────────────────────────────────────────────────────────────────
# STEP 1 — Load training data, extract Normal-only subset
# ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Loading training data...")
X_train_all, y_train_all, feat_cols = load_features("train.csv")
X_normal = X_train_all[y_train_all == 0]   # Train AE only on non-dispute rows
print(f"  Total train rows  : {len(X_train_all):,}")
print(f"  Normal (N) rows   : {len(X_normal):,}")
print(f"  Dispute (D) rows  : {(y_train_all==1).sum():,}")
print(f"  Feature count     : {X_normal.shape[1]}")

# ─────────────────────────────────────────────────────────────────────
# STEP 2 — Preprocess: signed-log1p skewed columns, scale, clip
#          (fit on normal train only — see src/autoencoder_preprocessing.py)
# ─────────────────────────────────────────────────────────────────────
print("\nStep 2: Fitting AutoencoderPreprocessor (signed-log1p + scale + clip) on normal train data...")
preprocessor = AutoencoderPreprocessor(skew_threshold=2.0, clip_value=5.0).fit(X_normal)
X_normal_scaled = preprocessor.transform(X_normal)
print(f"  {len(preprocessor.log_cols)} / {len(feat_cols)} features were signed-log1p-transformed "
      f"(|skew| > {preprocessor.skew_threshold} on normal rows)")

# ─────────────────────────────────────────────────────────────────────
# STEP 3 — Build & train Autoencoder
# ─────────────────────────────────────────────────────────────────────
input_dim = X_normal_scaled.shape[1]
model = Autoencoder(input_dim, HIDDEN)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.MSELoss()

# 90/10 internal split for early stopping (within normal rows only)
n_val = int(0.1 * len(X_normal_scaled))
X_ae_train = torch.tensor(X_normal_scaled[:-n_val])
X_ae_val   = torch.tensor(X_normal_scaled[-n_val:])

train_loader = DataLoader(TensorDataset(X_ae_train, X_ae_train),
                          batch_size=BATCH, shuffle=True)

print(f"\nStep 3: Training Autoencoder ({EPOCHS} epochs, arch={HIDDEN})...")
best_val_loss, patience_ctr, best_state = np.inf, 0, None
history = []
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0
    for xb, _ in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), xb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(X_ae_train)

    # Validate
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_ae_val), X_ae_val).item()
    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_ctr = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

model.load_state_dict(best_state)
print(f"  Training complete in {time.time()-t0:.1f}s | Best Val MSE: {best_val_loss:.6f}")

# Save training loss curve
hist_df = pd.DataFrame(history)
hist_df.to_csv(os.path.join(OUTPUT_DIR, "training_loss.csv"), index=False)

# Plot loss curve
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(hist_df["epoch"], hist_df["train_loss"], label="Train MSE", color="#3498db")
ax.plot(hist_df["epoch"], hist_df["val_loss"],   label="Val MSE",   color="#e74c3c", linestyle="--")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.set_title("Autoencoder Training Loss Curve — Model 4", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ae_loss_curve.png"), dpi=200, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────
# STEP 4 — Compute reconstruction errors across all normal train data
#          Use to set reference min/max for pseudo-probability scaling
# ─────────────────────────────────────────────────────────────────────
print("\nStep 4: Computing reference reconstruction error range...")
X_train_scaled = preprocessor.transform(X_train_all)
train_errors = get_recon_error(model, torch.tensor(X_train_scaled))
ref_min = np.percentile(train_errors[y_train_all == 0], 1)
ref_max = np.percentile(train_errors, 99)
print(f"  Normal train errors → mean: {train_errors[y_train_all==0].mean():.4f} | "
      f"std: {train_errors[y_train_all==0].std():.4f}")
print(f"  Dispute train errors → mean: {train_errors[y_train_all==1].mean():.4f} | "
      f"std: {train_errors[y_train_all==1].std():.4f}")
print(f"  Reference range for scaling: [{ref_min:.4f}, {ref_max:.4f}]")

# ─────────────────────────────────────────────────────────────────────
# STEP 5 — Evaluate on all datasets
# ─────────────────────────────────────────────────────────────────────
print("\nStep 5: Evaluating on all datasets...")

# Choose ONE decision threshold on Validation OOS, then reuse it unchanged on
# every other split — see module docstring.
X_val_raw, y_val, _ = load_features("val_oos.csv", feat_cols)
X_val_sc = preprocessor.transform(X_val_raw)
val_errors = get_recon_error(model, torch.tensor(X_val_sc))
val_proba  = normalize_to_prob(val_errors, ref_min, ref_max)
FIXED_THRESHOLD = select_f1_threshold(y_val, val_proba)
print(f"  Threshold fixed at {FIXED_THRESHOLD:.4f} (F1-maximising on Validation OOS only)")

results, scores_dict, y_dict = [], {}, {}

for fname, label in EVAL_SETS:
    X_raw, y, _ = load_features(fname, feat_cols)
    X_sc = preprocessor.transform(X_raw)
    errors = get_recon_error(model, torch.tensor(X_sc))
    proba  = normalize_to_prob(errors, ref_min, ref_max)

    res, _ = evaluate(y, proba, FIXED_THRESHOLD)
    res.update({"Dataset": label, "Rows": len(y), "Disputes": int(y.sum())})
    # reorder columns
    res = {k: res[k] for k in ["Dataset","Rows","Disputes","AUC_ROC",
                                 "AUC_ROC_CI_Low","AUC_ROC_CI_High","AUC_PR",
                                 "Gini","KS","Top_Decile_Recall","Threshold_Used",
                                 "Precision","Recall","F1","TP","FP","TN","FN"]}
    results.append(res)
    scores_dict[label] = proba
    y_dict[label]      = y
    print(f"  {label:35s} | AUC={res['AUC_ROC']:.4f} | KS={res['KS']:.4f} | "
          f"Gini={res['Gini']:.4f} | TopDecile={res['Top_Decile_Recall']:.4f} | F1={res['F1']:.4f}")

pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "performance_scorecard.csv"), index=False)

# ─────────────────────────────────────────────────────────────────────
# STEP 6 — Feature Importance: per-feature reconstruction error gap
#          (Dispute MSE - Normal MSE per feature, on train data)
#          This is a simple, label-derived, GLOBAL diagnostic only — it is
#          NOT a per-instance explanation and is NOT one of the proposal's
#          Tri-Explainer methods (those are built in Phase 3).
# ─────────────────────────────────────────────────────────────────────
print("\nStep 6: Computing per-feature reconstruction importance (diagnostic only)...")
model.eval()
with torch.no_grad():
    X_t = torch.tensor(X_train_scaled)
    recon = model(X_t).numpy()
feat_errors = (X_train_scaled - recon) ** 2   # shape: (n_samples, n_features)

fi_normal = feat_errors[y_train_all == 0].mean(axis=0)
fi_dispute = feat_errors[y_train_all == 1].mean(axis=0)
fi_gap = fi_dispute - fi_normal   # High gap = feature much worse reconstructed for disputes

fi_df = pd.DataFrame({
    "Feature": feat_cols,
    "Normal_Recon_Error":  fi_normal,
    "Dispute_Recon_Error": fi_dispute,
    "Gap (Dispute-Normal)": fi_gap
}).sort_values("Gap (Dispute-Normal)", ascending=False).reset_index(drop=True)
fi_df["Rank"] = fi_df.index + 1
fi_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

print("Top 20 features by Dispute Reconstruction Gap:")
print(fi_df[["Rank","Feature","Normal_Recon_Error","Dispute_Recon_Error",
             "Gap (Dispute-Normal)"]].head(20).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────
# STEP 7 — ROC Curves
# ─────────────────────────────────────────────────────────────────────
print("\nStep 7: Plotting ROC curves...")
COLORS = ["#2c3e50","#3498db","#e74c3c","#27ae60","#f39c12","#9b59b6"]
STYLES = ["-","--","-.",":","-.",":"]

fig, ax = plt.subplots(figsize=(10, 8))
for (_, label), col, ls in zip(EVAL_SETS, COLORS, STYLES):
    fpr, tpr, _ = roc_curve(y_dict[label], scores_dict[label])
    auc = roc_auc_score(y_dict[label], scores_dict[label])
    lw = 2.5 if "Train" in label else 2.0
    ax.plot(fpr, tpr, lw=lw, color=col, linestyle=ls,
            label=f"{label} (AUC = {auc:.3f})")

ax.plot([0,1],[0,1], "k--", lw=1.2, label="Random Classifier")
ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Model 4 (Autoencoder Anomaly Detection)", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_chart_model4.png"), dpi=300, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────
# STEP 8 — Hosmer-Lemeshow
#          (Caveat, see ARCHITECTURE.md limitation 10: this model's
#          "probability" is a min-max-scaled reconstruction error, not a
#          real probability, so a calibration test on it is not meaningful.
#          Kept only for visual/diagnostic continuity with the other models.)
# ─────────────────────────────────────────────────────────────────────
print("\nStep 8: Hosmer-Lemeshow calibration test (diagnostic only — see caveat above)...")
fig, axes = plt.subplots(3, 2, figsize=(15, 13))
axes = axes.flatten()
all_hl, hl_summary = [], []

for i, (_, label) in enumerate(EVAL_SETS):
    stat, pval, hl_df = hosmer_lemeshow(y_dict[label], scores_dict[label])
    hl_df["Dataset"] = label
    all_hl.append(hl_df)
    hl_summary.append({"Dataset": label, "HL_Statistic": round(stat,2),
                        "P_Value": round(pval,6), "Bins": len(hl_df)})
    ax = axes[i]
    x = hl_df["Decile_Rank"]
    ax.bar(x - 0.2, hl_df["Exp_E"], 0.4, label="Expected (Model)", color="#3498db", alpha=0.85)
    ax.bar(x + 0.2, hl_df["Obs_E"], 0.4, label="Observed (Actual)", color="#e74c3c", alpha=0.85)
    ax.set_title(f"{label}\nHL={stat:.1f}  p={pval:.4f}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Decile (Low → High Risk)", fontsize=9)
    ax.set_ylabel("# Disputes", fontsize=9)
    ax.set_xticks(x)
    if i == 0: ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Hosmer-Lemeshow Calibration — Model 4 (Autoencoder)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hosmer_lemeshow_chart_model4.png"), dpi=300, bbox_inches="tight")
plt.close()

pd.concat(all_hl)[["Dataset","Decile_Rank","Total","Obs_E","Exp_E","Obs_NE","Exp_NE"]].to_csv(
    os.path.join(OUTPUT_DIR, "hosmer_lemeshow_data.csv"), index=False)

print("\nHosmer-Lemeshow Summary:")
print(pd.DataFrame(hl_summary).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────
# STEP 9 — Save model, preprocessor & config
# ─────────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model4_ae_state.pt"))
with open(os.path.join(OUTPUT_DIR, "preprocessor.pkl"), "wb") as f:
    pickle.dump(preprocessor, f)
config = {"architecture": f"AE {input_dim}->{HIDDEN}->{input_dim} "
                           "(bottleneck is a plain Linear layer, no BatchNorm/Dropout)",
          "hidden_dims":  HIDDEN, "epochs": EPOCHS, "batch_size": BATCH,
          "learning_rate": LR, "early_stopping_patience": PATIENCE,
          "best_val_mse": round(best_val_loss, 6),
          "n_features":   len(feat_cols),
          "preprocessing": preprocessor.summary(),
          "training_note": "Trained ONLY on non-dispute (N) rows from train.csv",
          "normal_train_rows": int((y_train_all==0).sum()),
          "dispute_train_rows": int((y_train_all==1).sum())}
with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"\n✅ Model 4 complete. All outputs → {OUTPUT_DIR}")
