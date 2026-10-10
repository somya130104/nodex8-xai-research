"""
LSTM Model — sequence-based, Segment as learned embedding.
Each customer's weekly observations form a time series; the LSTM
learns temporal patterns from 13-week lookback windows.
Predictions at CustID level with all original columns retained.
"""

import os, warnings, pickle, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")
torch.manual_seed(42)

MODEL_NAME = "LSTM"
OUT_DIR    = f"../outputs/{MODEL_NAME}"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── 0. Config ─────────────────────────────────────────────────────────────────
TARGET      = "UnitSales"
LOG_TARGET  = "log1p_sales"
SEQ_LEN     = 13          # weeks of lookback per sequence window

# Instantaneous features only — LSTM learns temporal structure from the sequence
SEQ_FEATURES = [
    "week_sin", "week_cos", "quarter", "is_year_start", "is_year_end",
    "is_promo", "promo_intensity",
    "pre_promo_1", "pre_promo_2", "post_promo_1", "post_promo_2",
    "weeks_since_promo", "weeks_to_next_promo", "promo_density_13",
    "August", "Christmas", "Spring", "NewYear",
    "September", "Thanksgiving", "VetreansDay", "any_holiday",
    "promo_x_holiday", "promo_x_windex", "holiday_x_windex",
    "Weekly_Index",
    "cust_hist_mean", "zero_streak", "zero_frac_13",
    # Include lag_1 so the network has a sales-level signal at each step
    "lag_1",
]

FILL_VALUES = {
    "lag_1": 0,
    "promo_density_13": 0, "cust_hist_mean": 0,
    "zero_frac_13": 0, "zero_streak": 0,
    "weeks_since_promo": 52, "weeks_to_next_promo": 52,
}

LSTM_CFG = dict(hidden_size=64, num_layers=2, dropout=0.2)
TRAIN_CFG = dict(
    lr=1e-3, batch_size=512, max_epochs=100,
    patience=10, val_weeks=20,
)

# ── 1. Model definition ───────────────────────────────────────────────────────
class SalesLSTM(nn.Module):
    def __init__(self, n_features, n_segments, embed_dim=4, **lstm_cfg):
        super().__init__()
        self.seg_embed = nn.Embedding(n_segments, embed_dim)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_cfg["hidden_size"],
            num_layers=lstm_cfg["num_layers"],
            dropout=lstm_cfg["dropout"] if lstm_cfg["num_layers"] > 1 else 0,
            batch_first=True,
        )
        fc_in = lstm_cfg["hidden_size"] + embed_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 32),
            nn.ReLU(),
            nn.Dropout(lstm_cfg["dropout"]),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq, x_seg):
        _, (h_n, _) = self.lstm(x_seq)        # h_n: [layers, B, hidden]
        h = h_n[-1]                            # last layer: [B, hidden]
        seg = self.seg_embed(x_seg)            # [B, embed_dim]
        out = self.fc(torch.cat([h, seg], 1))  # [B, 1]
        return out.squeeze(1)


# ── 2. Dataset ────────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, X_seq, X_seg, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_seg = torch.tensor(X_seg, dtype=torch.long)
        self.y     = torch.tensor(y,     dtype=torch.float32)

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X_seq[i], self.X_seg[i], self.y[i]


# ── 3. Helpers ────────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[LOG_TARGET] = np.log1p(df[TARGET].clip(0))
    for col, val in FILL_VALUES.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    for col in ["weeks_since_promo", "weeks_to_next_promo"]:
        if col in df.columns:
            df[col] = df[col].clip(upper=52)
    return df


def build_sequences(full_scaled: dict, query_df: pd.DataFrame,
                    le: LabelEncoder, scaler: StandardScaler) -> tuple:
    """
    For each row in query_df, look up SEQ_LEN timesteps of history from
    full_scaled (keyed by (Segment, Cust_ID, global_t)) and stack into arrays.
    Zero-padding used for cold-start rows with insufficient history.
    """
    N, F = len(query_df), len(SEQ_FEATURES)
    X_seq = np.zeros((N, SEQ_LEN, F), dtype=np.float32)
    X_seg = np.zeros(N, dtype=np.int64)
    y_arr = np.zeros(N, dtype=np.float32)

    for i, (_, row) in enumerate(query_df.iterrows()):
        seg, cid, t = row["Segment"], row["Cust_ID"], int(row["global_t"])
        for j, tj in enumerate(range(t - SEQ_LEN + 1, t + 1)):
            vec = full_scaled.get((seg, cid, tj))
            if vec is not None:
                X_seq[i, j] = vec
        X_seg[i] = le.transform([seg])[0]
        y_arr[i]  = row[LOG_TARGET]

    return X_seq, X_seg, y_arr


def metrics(y_true: np.ndarray, y_pred_log: np.ndarray) -> dict:
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred_log).clip(0)
    rmse = np.sqrt(np.mean((y_true_orig - y_pred_orig) ** 2))
    mae  = np.mean(np.abs(y_true_orig - y_pred_orig))
    mask = y_true_orig > 0
    mape = np.mean(np.abs((y_true_orig[mask] - y_pred_orig[mask]) / y_true_orig[mask])) * 100
    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "MAPE%": round(mape, 2)}


@torch.no_grad()
def predict_log(model, X_seq, X_seg, batch_size=1024):
    model.eval()
    preds = []
    ds = SequenceDataset(X_seq, X_seg, np.zeros(len(X_seq)))
    for xb, sb, _ in DataLoader(ds, batch_size=batch_size):
        preds.append(model(xb.to(DEVICE), sb.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds)


# ── 4. Load data ──────────────────────────────────────────────────────────────
full_df = preprocess(pd.read_csv("../data/SalesForecasting_features.csv"))
dev     = preprocess(pd.read_csv("../data/samples/dev.csv"))
oos     = preprocess(pd.read_csv("../data/samples/oos.csv"))
oot     = preprocess(pd.read_csv("../data/samples/oot.csv"))

print(f"Full : {len(full_df):>7,} rows  |  DEV: {len(dev):,}  OOS: {len(oos):,}  OOT: {len(oot):,}")

# ── 5. Fit scaler & label encoder on DEV ──────────────────────────────────────
le = LabelEncoder().fit(dev["Segment"])
scaler = StandardScaler().fit(dev[SEQ_FEATURES].fillna(0))

# Scale full dataset and build fast lookup dict
full_scaled_arr = scaler.transform(full_df[SEQ_FEATURES].fillna(0))
full_lookup = {
    (row.Segment, row.Cust_ID, int(row.global_t)): full_scaled_arr[i]
    for i, row in full_df.iterrows()
}
print(f"Lookup built: {len(full_lookup):,} (Segment, CustID, week) entries")

# ── 6. Build sequence arrays ──────────────────────────────────────────────────
print("Building sequences …")
X_dev_seq, X_dev_seg, y_dev = build_sequences(full_lookup, dev, le, scaler)
X_oos_seq, X_oos_seg, y_oos = build_sequences(full_lookup, oos, le, scaler)
X_oot_seq, X_oot_seg, y_oot = build_sequences(full_lookup, oot, le, scaler)
print(f"DEV sequences : {X_dev_seq.shape}  |  OOS: {X_oos_seq.shape}  |  OOT: {X_oot_seq.shape}")

# ── 7. Time-based train / val split within DEV ───────────────────────────────
val_cutoff = dev["global_t"].max() - TRAIN_CFG["val_weeks"]
val_mask   = (dev["global_t"] > val_cutoff).values
trn_mask   = ~val_mask

X_trn_seq, X_trn_seg, y_trn = X_dev_seq[trn_mask], X_dev_seg[trn_mask], y_dev[trn_mask]
X_val_seq, X_val_seg, y_val = X_dev_seq[val_mask],  X_dev_seg[val_mask],  y_dev[val_mask]

print(f"\nTrain: {trn_mask.sum():,} rows  |  Val: {val_mask.sum():,} rows")

# ── 8. Train ──────────────────────────────────────────────────────────────────
n_features  = len(SEQ_FEATURES)
n_segments  = len(le.classes_)

model     = SalesLSTM(n_features, n_segments, **LSTM_CFG).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CFG["lr"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=4, factor=0.5
)
criterion = nn.MSELoss()

train_ds  = SequenceDataset(X_trn_seq, X_trn_seg, y_trn)
train_ldr = DataLoader(train_ds, batch_size=TRAIN_CFG["batch_size"],
                       shuffle=True, num_workers=0)

X_val_t = torch.tensor(X_val_seq, dtype=torch.float32).to(DEVICE)
y_val_t = torch.tensor(y_val,     dtype=torch.float32).to(DEVICE)
S_val_t = torch.tensor(X_val_seg, dtype=torch.long).to(DEVICE)

best_val, best_state, patience_cnt = np.inf, None, 0
loss_log = []

print(f"\n── Training LSTM ───────────────────────────────────────────")
for epoch in range(1, TRAIN_CFG["max_epochs"] + 1):
    model.train()
    trn_loss = 0.0
    for xb, sb, yb in train_ldr:
        xb, sb, yb = xb.to(DEVICE), sb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb, sb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        trn_loss += loss.item() * len(yb)
    trn_loss /= len(y_trn)

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_t, S_val_t), y_val_t).item()

    scheduler.step(val_loss)
    loss_log.append({"epoch": epoch, "train_mse": trn_loss, "val_mse": val_loss})

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3}  train_mse={trn_loss:.4f}  val_mse={val_loss:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

    if val_loss < best_val:
        best_val, best_state, patience_cnt = val_loss, copy.deepcopy(model.state_dict()), 0
    else:
        patience_cnt += 1
        if patience_cnt >= TRAIN_CFG["patience"]:
            print(f"\nEarly stopping at epoch {epoch}  (best val_mse={best_val:.4f})")
            break

model.load_state_dict(best_state)

# ── 9. Save model ─────────────────────────────────────────────────────────────
pkl_path = f"{OUT_DIR}/{MODEL_NAME}.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump({
        "model_state": best_state,
        "model_cfg"  : {"n_features": n_features, "n_segments": n_segments, **LSTM_CFG},
        "label_encoder": le,
        "scaler"     : scaler,
        "seq_features": SEQ_FEATURES,
        "seq_len"    : SEQ_LEN,
        "best_val_mse": best_val,
    }, f)
print(f"\nModel saved : {pkl_path}")

# Save loss curve
pd.DataFrame(loss_log).to_csv(f"{OUT_DIR}/loss_curve.csv", index=False)

# ── 10. Predictions ───────────────────────────────────────────────────────────
def save_full_predictions(df, seq_arr, seg_arr, log_y, label):
    pred_log = predict_log(model, seq_arr, seg_arr)
    out = df.copy()
    out["pred_lstm"] = np.expm1(pred_log).clip(0).round(2)
    out.to_csv(f"{OUT_DIR}/pred_{label}.csv", index=False)
    m = metrics(log_y, pred_log)
    print(f"  pred_{label}.csv  |  rows={len(out):>7,}  |  "
          f"RMSE={m['RMSE']:>8.2f}  MAE={m['MAE']:>7.2f}  MAPE={m['MAPE%']:>6.2f}%")
    return {"sample": label, **m}

print("\n── Predictions ─────────────────────────────────────────────")
perf = []
perf.append(save_full_predictions(dev, X_dev_seq, X_dev_seg, y_dev, "dev"))
perf.append(save_full_predictions(oos, X_oos_seq, X_oos_seg, y_oos, "oos"))

for bucket_id in sorted(oot["oot_bucket"].unique()):
    mask = (oot["oot_bucket"] == bucket_id).values
    perf.append(save_full_predictions(
        oot[mask], X_oot_seq[mask], X_oot_seg[mask], y_oot[mask],
        f"oot_{int(bucket_id):02d}"
    ))

# ── 11. Save performance ──────────────────────────────────────────────────────
perf_df = pd.DataFrame(perf)
perf_df.to_csv(f"{OUT_DIR}/performance.csv", index=False)

print(f"\n── Summary ─────────────────────────────────────────────────")
print(perf_df.to_string(index=False))
print(f"\nAll outputs in: {OUT_DIR}/")
