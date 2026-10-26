"""
Model 9 — Variational Autoencoder (VAE) Anomaly Detection
===========================================================
Phase 2.2 (see ROADMAP.md). Same anomaly-detection idea as Model 4's plain
Autoencoder (train on normal clients only, score by reconstruction error),
but the bottleneck is now a PROBABILITY DISTRIBUTION — a mean mu and a
variance sigma^2 — instead of one fixed point. This means the model can
express "I'm not sure what this looks like" (wide sigma^2), which is a
richer signal than the plain AE's single number, and gives Phase 3's LRD
method a more informative latent space to decompose (optionally weighting
each latent unit's contribution by how confident the model is in it).

There is no Model 1-8 gap issue here either — the numbering simply
continues from the existing Model 2,3,4,5,6,7,8 (no Model 1 exists; see
ARCHITECTURE.md Section 7, item 13 — a pre-existing quirk, left as-is).

Preprocessing is IDENTICAL to Model 4 (src/autoencoder_preprocessing.py):
signed-log1p the skewed columns, StandardScaler, clip to [-5, 5] — fit on
normal training rows only, same as the plain AE, so the two models' inputs
are directly comparable.

The one real risk with any VAE, and how this script avoids it: "posterior
collapse" — if the KL term in the loss dominates the reconstruction term
too early, the model learns to ignore the input entirely and just outputs
mu=0, sigma^2=1 for everything. This is avoided with KL ANNEALING: the KL
term's weight (beta) ramps linearly from 0 to 1 over the first 20% of
training, so the model first learns to reconstruct well, then is gradually
asked to also keep its latent space well-behaved.
"""
import os, json, pickle, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys

sys.path.append("/Users/somyabhadada/nodex8")
from src.eval_utils import (
    dedupe_rows, select_f1_threshold, metrics_at_threshold, bootstrap_auc_ci,
)
from src.autoencoder_preprocessing import AutoencoderPreprocessor
from src.autoencoder_models import VAE
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

DATA_DIR   = "/Users/somyabhadada/nodex8/data/TradeFraud/Modeling_Data_Enriched/"
OUTPUT_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/Model_Outputs/Model9_VAE/"

EXCLUDE_COLS = ["Client", "Client_ID", "Date", "Dispute", "Segment", "Activeness_Segment"]
EVAL_SETS = [
    ("train.csv",        "Train (all)"),
    ("val_oos.csv",      "Validation OOS"),
    ("internal_oot.csv", "Internal OOT (Aug–Sep)"),
    ("oot_202510.csv",   "OOT Oct 2025"),
    ("oot_202511.csv",   "OOT Nov 2025"),
    ("oot_202512.csv",   "OOT Dec 2025"),
]
LATENT_DIM = 32           # same size as Model 4's bottleneck, for comparability
HIDDEN = [256, 128]       # encoder/decoder hidden layers (mu/logvar heads sit after these)
EPOCHS = 60
BATCH = 512
LR = 1e-3
PATIENCE = 10
KL_ANNEAL_FRACTION = 0.2  # beta ramps 0 -> 1 over the first 20% of training epochs

os.makedirs(OUTPUT_DIR, exist_ok=True)


# VAE class now lives in src/autoencoder_models.py (importable by Phase 3's explainers)


def vae_loss(x, x_hat, mu, logvar, beta):
    recon = torch.mean(torch.sum((x - x_hat) ** 2, dim=1))
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon + beta * kl, recon, kl


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


def normalize_to_prob(errors, ref_min, ref_max):
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
    auc, auc_lo, auc_hi = bootstrap_auc_ci(y, proba)
    aupr = average_precision_score(y, proba)
    ks = ks_stat(y, proba)
    tdr = top_decile_recall(y, proba)
    gini = 2 * auc - 1
    thr_metrics = metrics_at_threshold(y, proba, threshold)
    return {"AUC_ROC": round(auc, 4), "AUC_ROC_CI_Low": auc_lo, "AUC_ROC_CI_High": auc_hi,
            "AUC_PR": round(aupr, 4), "Gini": round(gini, 4), "KS": round(ks, 4),
            "Top_Decile_Recall": round(tdr, 4), **thr_metrics}


# ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Loading training data...")
X_train_all, y_train_all, feat_cols = load_features("train.csv")
X_normal = X_train_all[y_train_all == 0]
print(f"  Total train rows: {len(X_train_all):,} | Normal: {len(X_normal):,} | "
      f"Dispute: {(y_train_all==1).sum():,} | Features: {X_normal.shape[1]}")

print("\nStep 2: Fitting AutoencoderPreprocessor (same as Model 4) on normal train data...")
preprocessor = AutoencoderPreprocessor(skew_threshold=2.0, clip_value=5.0).fit(X_normal)
X_normal_scaled = preprocessor.transform(X_normal)
print(f"  {len(preprocessor.log_cols)} / {len(feat_cols)} features signed-log1p-transformed")

input_dim = X_normal_scaled.shape[1]
model = VAE(input_dim, HIDDEN, LATENT_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

n_val = int(0.1 * len(X_normal_scaled))
X_train_t = torch.tensor(X_normal_scaled[:-n_val])
X_val_t = torch.tensor(X_normal_scaled[-n_val:])
train_loader = DataLoader(TensorDataset(X_train_t, X_train_t), batch_size=BATCH, shuffle=True)

anneal_epochs = max(1, int(EPOCHS * KL_ANNEAL_FRACTION))
print(f"\nStep 3: Training VAE ({EPOCHS} epochs, latent_dim={LATENT_DIM}, "
      f"KL annealed over first {anneal_epochs} epochs)...")

best_val_loss, patience_ctr, best_state = np.inf, 0, None
history = []
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    beta = min(1.0, epoch / anneal_epochs)
    model.train()
    train_recon, train_kl = 0.0, 0.0
    for xb, _ in train_loader:
        optimizer.zero_grad()
        x_hat, mu, logvar = model(xb)
        loss, recon, kl = vae_loss(xb, x_hat, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        train_recon += recon.item() * len(xb)
        train_kl += kl.item() * len(xb)
    train_recon /= len(X_train_t)
    train_kl /= len(X_train_t)

    model.eval()
    with torch.no_grad():
        x_hat_v, mu_v, logvar_v = model(X_val_t)
        val_loss, val_recon, val_kl = vae_loss(X_val_t, x_hat_v, mu_v, logvar_v, beta)
        val_loss, val_recon, val_kl = val_loss.item(), val_recon.item(), val_kl.item()

    history.append({"epoch": epoch, "beta": beta, "train_recon": train_recon, "train_kl": train_kl,
                     "val_recon": val_recon, "val_kl": val_kl, "val_loss": val_loss})

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | beta={beta:.2f} | "
              f"Train Recon: {train_recon:.4f} KL: {train_kl:.4f} | "
              f"Val Recon: {val_recon:.4f} KL: {val_kl:.4f}")

    # Early stopping on the fully-annealed (beta=1) validation loss only,
    # to avoid stopping early just because beta was still small
    if beta >= 1.0:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

if best_state is not None:
    model.load_state_dict(best_state)
print(f"  Training complete in {time.time()-t0:.1f}s | Best (post-anneal) Val Loss: {best_val_loss:.4f}")

# Posterior-collapse check: if KL is ~0 the whole time while reconstruction
# never improves, the model learned to ignore the input entirely.
final_kl = history[-1]["val_kl"]
final_recon = history[-1]["val_recon"]
first_recon = history[0]["val_recon"]
collapsed = final_kl < 0.01 and final_recon >= first_recon * 0.95
print(f"  Posterior collapse check: final val KL={final_kl:.4f}, "
      f"val recon {first_recon:.4f} -> {final_recon:.4f}  "
      f"=> {'COLLAPSED (see warning)' if collapsed else 'OK, model is using its latent space'}")
if collapsed:
    print("  WARNING: posterior collapse detected — the VAE may be ignoring its input. "
          "Consider lowering KL_ANNEAL_FRACTION or beta further.")

hist_df = pd.DataFrame(history)
hist_df.to_csv(os.path.join(OUTPUT_DIR, "training_loss.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(hist_df["epoch"], hist_df["train_recon"], label="Train Recon", color="#3498db")
axes[0].plot(hist_df["epoch"], hist_df["val_recon"], label="Val Recon", color="#e74c3c", linestyle="--")
axes[0].set_title("Reconstruction Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(hist_df["epoch"], hist_df["train_kl"], label="Train KL", color="#27ae60")
axes[1].plot(hist_df["epoch"], hist_df["val_kl"], label="Val KL", color="#f39c12", linestyle="--")
ax2 = axes[1].twinx()
ax2.plot(hist_df["epoch"], hist_df["beta"], label="beta (KL weight)", color="gray", linestyle=":")
axes[1].set_title("KL Divergence (and annealing schedule)"); axes[1].set_xlabel("Epoch")
axes[1].legend(loc="upper left"); ax2.legend(loc="upper right"); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "vae_loss_curve.png"), dpi=200, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────
print("\nStep 4: Computing reference reconstruction error range (z = mu, deterministic)...")
X_train_scaled = preprocessor.transform(X_train_all)
model.eval()
with torch.no_grad():
    train_errors = model.score(torch.tensor(X_train_scaled)).numpy()
ref_min = np.percentile(train_errors[y_train_all == 0], 1)
ref_max = np.percentile(train_errors, 99)
print(f"  Normal train errors -> mean: {train_errors[y_train_all==0].mean():.4f} | "
      f"std: {train_errors[y_train_all==0].std():.4f}")
print(f"  Dispute train errors -> mean: {train_errors[y_train_all==1].mean():.4f} | "
      f"std: {train_errors[y_train_all==1].std():.4f}")

print("\nStep 5: Evaluating on all datasets...")
X_val_raw, y_val, _ = load_features("val_oos.csv", feat_cols)
X_val_sc = preprocessor.transform(X_val_raw)
with torch.no_grad():
    val_errors = model.score(torch.tensor(X_val_sc)).numpy()
val_proba = normalize_to_prob(val_errors, ref_min, ref_max)
FIXED_THRESHOLD = select_f1_threshold(y_val, val_proba)
print(f"  Threshold fixed at {FIXED_THRESHOLD:.4f} (F1-maximising on Validation OOS only)")

results = []
for fname, label in EVAL_SETS:
    X_raw, y, _ = load_features(fname, feat_cols)
    X_sc = preprocessor.transform(X_raw)
    with torch.no_grad():
        errors = model.score(torch.tensor(X_sc)).numpy()
    proba = normalize_to_prob(errors, ref_min, ref_max)
    res = evaluate(y, proba, FIXED_THRESHOLD)
    res.update({"Dataset": label, "Rows": len(y), "Disputes": int(y.sum())})
    res = {k: res[k] for k in ["Dataset", "Rows", "Disputes", "AUC_ROC",
                                 "AUC_ROC_CI_Low", "AUC_ROC_CI_High", "AUC_PR",
                                 "Gini", "KS", "Top_Decile_Recall", "Threshold_Used",
                                 "Precision", "Recall", "F1", "TP", "FP", "TN", "FN"]}
    results.append(res)
    print(f"  {label:35s} | AUC={res['AUC_ROC']:.4f} | KS={res['KS']:.4f} | "
          f"Gini={res['Gini']:.4f} | TopDecile={res['Top_Decile_Recall']:.4f} | F1={res['F1']:.4f}")

pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "performance_scorecard.csv"), index=False)

# ─────────────────────────────────────────────────────────────────────
print("\nStep 6: Comparing VAE vs plain Autoencoder (Model 4) latent agreement...")
ae_config_path = "/Users/somyabhadada/nodex8/data/TradeFraud/Model_Outputs/Model4_Autoencoder/performance_scorecard.csv"
if os.path.exists(ae_config_path):
    ae_scores = pd.read_csv(ae_config_path)
    vae_scores = pd.DataFrame(results)
    merged = ae_scores[["Dataset", "AUC_ROC"]].merge(
        vae_scores[["Dataset", "AUC_ROC"]], on="Dataset", suffixes=("_AE", "_VAE"))
    print(merged.to_string(index=False))
    merged.to_csv(os.path.join(OUTPUT_DIR, "vae_vs_ae_comparison.csv"), index=False)

# ─────────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model9_vae_state.pt"))
with open(os.path.join(OUTPUT_DIR, "preprocessor.pkl"), "wb") as f:
    pickle.dump(preprocessor, f)
config = {
    "architecture": f"VAE {input_dim}->{HIDDEN}->mu/logvar({LATENT_DIM})->{HIDDEN[::-1]}->{input_dim}",
    "hidden_dims": HIDDEN, "latent_dim": LATENT_DIM,
    "epochs_run": len(history), "batch_size": BATCH, "learning_rate": LR,
    "early_stopping_patience": PATIENCE, "kl_anneal_fraction": KL_ANNEAL_FRACTION,
    "best_post_anneal_val_loss": round(best_val_loss, 6) if best_state is not None else None,
    "posterior_collapse_detected": bool(collapsed),
    "n_features": len(feat_cols),
    "preprocessing": preprocessor.summary(),
    "training_note": "Trained ONLY on non-dispute (N) rows from train.csv; scoring uses z=mu "
                      "(deterministic, no sampling) so every instance gets a reproducible score.",
    "normal_train_rows": int((y_train_all == 0).sum()),
    "dispute_train_rows": int((y_train_all == 1).sum()),
}
with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"\n✅ Model 9 (VAE) complete. All outputs → {OUTPUT_DIR}")
