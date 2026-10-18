"""
Model 4 — Autoencoder Anomaly Detection
=========================================
Core idea:   Train ONLY on non-dispute (normal) transactions.
             Disputes → high reconstruction error → anomaly score.

Advantages:  Sidesteps oversampling entirely; learns what "normal" looks like
             and flags deviations — ideal for 0.23% dispute rate.

Outputs:     Performance scorecard, ROC curve, HL calibration chart,
             per-feature reconstruction error importance, model artefacts.
"""
import os, json, pickle, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             f1_score, precision_score, recall_score, confusion_matrix)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
HIDDEN  = [128, 64, 32]    # Encoder dims; decoder mirrors
EPOCHS  = 60
BATCH   = 512
LR      = 1e-3
PATIENCE = 10              # Early stopping patience

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Autoencoder Model ─────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        # Encoder
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        self.encoder = nn.Sequential(*enc)
        # Decoder (mirrors encoder, skip bottleneck layer itself)
        dec = []
        for h in reversed(hidden_dims[:-1]):
            dec += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ── Helpers ───────────────────────────────────────────────────────────
def load_features(fname, feat_cols=None):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df["target"] = (df["Dispute"] == "D").astype(int)
    if feat_cols is None:
        feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != "target"]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    y = df["target"].values
    return X, y, feat_cols

def get_recon_error(model, X_scaled_tensor):
    model.eval()
    with torch.no_grad():
        recon = model(X_scaled_tensor)
        errors = torch.mean((X_scaled_tensor - recon) ** 2, dim=1).numpy()
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

def evaluate(y, proba):
    auc  = roc_auc_score(y, proba)
    aupr = average_precision_score(y, proba)
    ks   = ks_stat(y, proba)
    tdr  = top_decile_recall(y, proba)
    gini = 2 * auc - 1
    thrs = np.linspace(0.001, 0.999, 300)
    best_f1, best_thr = 0, 0.5
    for t in thrs:
        pred = (proba >= t).astype(int)
        if pred.sum() == 0: continue
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1: best_f1, best_thr = f, t
    pred_opt = (proba >= best_thr).astype(int)
    prec = precision_score(y, pred_opt, zero_division=0)
    rec  = recall_score(y, pred_opt, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred_opt).ravel()
    return {"AUC_ROC": round(auc,4), "AUC_PR": round(aupr,4),
            "Gini": round(gini,4), "KS": round(ks,4),
            "Top_Decile_Recall": round(tdr,4), "Best_Threshold": round(best_thr,4),
            "Precision": round(prec,4), "Recall": round(rec,4), "F1": round(best_f1,4),
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}, proba

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
# STEP 2 — Normalise features (fit on normal train only)
# ─────────────────────────────────────────────────────────────────────
print("\nStep 2: Fitting StandardScaler on normal train data...")
scaler = StandardScaler()
X_normal_scaled = scaler.fit_transform(X_normal).astype(np.float32)

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
X_train_scaled = scaler.transform(X_train_all).astype(np.float32)
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
results, scores_dict, y_dict = [], {}, {}

for fname, label in EVAL_SETS:
    X_raw, y, _ = load_features(fname, feat_cols)
    X_sc = scaler.transform(X_raw).astype(np.float32)
    errors = get_recon_error(model, torch.tensor(X_sc))
    proba  = normalize_to_prob(errors, ref_min, ref_max)

    res, _ = evaluate(y, proba)
    res.update({"Dataset": label, "Rows": len(y), "Disputes": int(y.sum())})
    # reorder columns
    res = {k: res[k] for k in ["Dataset","Rows","Disputes","AUC_ROC","AUC_PR",
                                 "Gini","KS","Top_Decile_Recall","Best_Threshold",
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
# ─────────────────────────────────────────────────────────────────────
print("\nStep 6: Computing per-feature reconstruction importance...")
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
# ─────────────────────────────────────────────────────────────────────
print("\nStep 8: Hosmer-Lemeshow calibration test...")
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
# STEP 9 — Save model & config
# ─────────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model4_ae_state.pt"))
with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
config = {"architecture": f"AE {input_dim}→{HIDDEN}→{input_dim}",
          "hidden_dims":  HIDDEN, "epochs": EPOCHS, "batch_size": BATCH,
          "learning_rate": LR, "early_stopping_patience": PATIENCE,
          "best_val_mse": round(best_val_loss, 6),
          "n_features":   len(feat_cols),
          "training_note": "Trained ONLY on non-dispute (N) rows from train.csv",
          "normal_train_rows": int((y_train_all==0).sum()),
          "dispute_train_rows": int((y_train_all==1).sum())}
with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"\n✅ Model 4 complete. All outputs → {OUTPUT_DIR}")
