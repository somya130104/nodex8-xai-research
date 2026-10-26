"""
Phase 2.4 — DemandLSTM: a genuine sequence model for DemandForecast.
=====================================================================
Replaces the previous train_lstm.py, which fed sequences of length 1 (see
ARCHITECTURE.md Section 7, item 8 — that made it mathematically identical to
a plain feed-forward network, with no actual temporal memory being used).

This version builds a real 28-calendar-day lookback window per product
(UPC), adapting the windowing convention from SalesForecasting's reference
LSTM (data/SalesForecasting/Modeling/codes/lstm_model.py): for a row on day
t, the sequence covers days [t-27 .. t] inclusive (so the model sees the
current day's own price/promo/calendar features at the last timestep, plus
27 days of history before it), and the target is that same day's QUANTITY.
Days with no data for that UPC (gaps, or simply too early in a product's
history) are zero-padded — this is a real cold-start case, not an error.

Why calendar days and not "the last 28 rows": checked first (see commit
history / ARCHITECTURE.md) — almost every UPC has exactly one row per
calendar day with no gaps, so this distinction barely matters in practice,
but calendar-day lookback is the honest definition and handles the few
UPCs that do have gaps or short histories correctly.

Why this needs the FULL features.csv, not just one split file: a sequence
for a day in OOT needs the 27 days before it, which live in DEV or OOS (the
UPC's own split, from the brand-based DEV/OOS split — never split in time).
features.csv has the full unsplit history, so it is used to build the
lookback lookup table; each split file (dev/oos/oot_*) then only supplies
which (UPC, day) rows to EVALUATE on and what a day's target value is.
"""
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

FULL_DATA_PATH = "/Users/somyabhadada/nodex8/data/DemandForecast/features.csv"
DATA_DIR       = "/Users/somyabhadada/nodex8/data/DemandForecast/samples/"
OUTPUT_DIR     = "/Users/somyabhadada/nodex8/data/DemandForecast/Model_Outputs/LSTM/"

EXCLUDE_COLS = ["RowNum", "Date", "UPC", "COMP_STR_NO", "QUANTITY", "BRAND", "RET_LIR_ID"]
CATEGORICAL_COLS = ["PROMOTION_FLG", "C_PROMOTION_FLG"]
SEQ_LEN = 28  # calendar days, inclusive of the target day itself

EVAL_SETS = [
    ("dev.csv", "DEV"),
    ("oos.csv", "OOS"),
    ("oot_01_201212.csv", "OOT_201212"),
    ("oot_02_201301.csv", "OOT_201301"),
    ("oot_03_201302.csv", "OOT_201302"),
]


class DemandLSTM(nn.Module):
    """A real sequence model: the LSTM's hidden state carries information
    forward across the 28-day window, instead of seeing one timestep at a
    time with nothing to remember."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq):
        # x_seq: (batch, SEQ_LEN, n_features)
        out, _ = self.lstm(x_seq)
        y = self.fc(out[:, -1, :])  # last timestep's hidden state
        return torch.relu(y).squeeze(-1)


def clean_categoricals(df):
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].map({"Y": 1, "N": 0, "1": 1, "0": 0, 1: 1, 0: 0}).fillna(0)
    return df


def build_day_lookup(full_df, feat_cols, scaler):
    """
    Returns a dict {(UPC, day_index): scaled_feature_vector} covering the
    ENTIRE dataset (all time periods), so a sequence for any split (DEV,
    OOS, or an OOT month) can look back into history outside that split's
    own rows.
    """
    scaled = scaler.transform(full_df[feat_cols].fillna(0).values.astype(np.float32))
    lookup = {}
    for i, (upc, day_idx) in enumerate(zip(full_df["UPC"].values, full_df["_day_idx"].values)):
        lookup[(upc, int(day_idx))] = scaled[i]
    return lookup


def build_sequences(query_df, lookup, n_features):
    """For each row in query_df, assemble its [t-SEQ_LEN+1 .. t] window."""
    n = len(query_df)
    X_seq = np.zeros((n, SEQ_LEN, n_features), dtype=np.float32)
    y = query_df["QUANTITY"].fillna(0).values.astype(np.float32)
    upcs = query_df["UPC"].values
    day_idxs = query_df["_day_idx"].values
    for i in range(n):
        upc, t = upcs[i], int(day_idxs[i])
        for j, tj in enumerate(range(t - SEQ_LEN + 1, t + 1)):
            vec = lookup.get((upc, tj))
            if vec is not None:
                X_seq[i, j] = vec
    return X_seq, y


def evaluate(model, X_eval, y_true, device, label):
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X_eval)), batch_size=1024, shuffle=False)
    preds = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            preds.extend(model(bx).cpu().numpy())
    pred = np.maximum(0, np.array(preds))

    mae = mean_absolute_error(y_true, pred)
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    r2 = r2_score(y_true, pred)

    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - pred[mask]) / y_true[mask])) if mask.sum() > 0 else np.nan

    hits = np.abs(y_true - pred) <= (0.2 * np.abs(y_true))
    acc_20 = hits.mean()

    return {
        "Dataset": label, "Rows": len(y_true),
        "MAE": round(mae, 4), "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4) if not np.isnan(mape) else None,
        "R2": round(r2, 4), "Accuracy_20pct": round(acc_20, 4),
    }


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading full history (features.csv) to build the day-level lookback table...")
    full_df = pd.read_csv(FULL_DATA_PATH, parse_dates=["Date"])
    full_df = clean_categoricals(full_df)
    min_date = full_df["Date"].min()
    full_df["_day_idx"] = (full_df["Date"] - min_date).dt.days

    feat_cols = [c for c in full_df.columns if c not in EXCLUDE_COLS and c != "_day_idx"]
    print(f"  {len(full_df):,} rows total, {full_df['UPC'].nunique()} products, {len(feat_cols)} features")

    print("\nLoading DEV to fit the scaler (never fit on OOS/OOT)...")
    dev_df = clean_categoricals(pd.read_csv(os.path.join(DATA_DIR, "dev.csv"), parse_dates=["Date"]))
    scaler = StandardScaler().fit(dev_df[feat_cols].fillna(0))

    print("Building the full day-level lookup table (used for every split's lookback)...")
    lookup = build_day_lookup(full_df, feat_cols, scaler)

    print("\nBuilding 28-day sequences for DEV...")
    dev_df["_day_idx"] = (dev_df["Date"] - min_date).dt.days
    X_dev, y_dev = build_sequences(dev_df, lookup, len(feat_cols))

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.15, random_state=42,
    )
    # Early stopping uses an internal split of DEV only — OOS/OOT are never
    # touched until final evaluation (same rule as every other model here).

    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUTPUT_DIR, "features.json"), "w") as f:
        json.dump(feat_cols, f, indent=4)

    train_dataset = TensorDataset(torch.tensor(X_train_raw), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val_raw), torch.tensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    model = DemandLSTM(input_dim=len(feat_cols), hidden_dim=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_rmse = float("inf")
    patience, patience_cnt = 6, 0

    print(f"\nTraining DemandLSTM on real {SEQ_LEN}-day sequences...")
    for epoch in range(1, 30):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(by)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item() * len(by)
        val_loss /= len(val_dataset)
        val_rmse = np.sqrt(val_loss)
        scheduler.step(val_rmse)

        print(f"Epoch {epoch:2d} | Train MSE: {train_loss:.4f} | Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_lstm.pt"))
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_lstm.pt")))

    print("\nEvaluating across sets (building sequences for each)...")
    scorecards = []
    sanity_pred_before = None
    for fname, lbl in EVAL_SETS:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        q_df = clean_categoricals(pd.read_csv(path, parse_dates=["Date"]))
        q_df["_day_idx"] = (q_df["Date"] - min_date).dt.days
        X_eval, y_eval = build_sequences(q_df, lookup, len(feat_cols))
        metrics = evaluate(model, X_eval, y_eval, device, lbl)
        scorecards.append(metrics)
        print(f"  {lbl:12s}: MAE={metrics['MAE']}, RMSE={metrics['RMSE']}, "
              f"R2={metrics['R2']}, Acc_20%={metrics['Accuracy_20pct']}")
        if lbl == "OOS":
            sanity_x, sanity_y = X_eval, y_eval

    df_score = pd.DataFrame(scorecards)
    df_score.to_csv(os.path.join(OUTPUT_DIR, "performance_scorecard.csv"), index=False)

    # ── Sanity check: is the sequence actually being used? ──────────────
    # Shuffle the order of timesteps within each sequence (keeping the final
    # timestep — the current day — fixed, so each row still "knows" today's
    # own price/promo) and confirm predictions change. If they don't, the
    # LSTM isn't using its memory of the past and we're back to the old bug.
    print("\nSanity check: does shuffling sequence order change predictions?")
    rng = np.random.RandomState(0)
    sample_idx = rng.choice(len(sanity_x), min(200, len(sanity_x)), replace=False)
    x_sample = sanity_x[sample_idx]
    x_shuffled = x_sample.copy()
    for i in range(len(x_shuffled)):
        perm = rng.permutation(SEQ_LEN - 1)  # shuffle all but the last timestep
        x_shuffled[i, :-1] = x_shuffled[i, perm]
    with torch.no_grad():
        pred_orig = model(torch.tensor(x_sample).to(device)).cpu().numpy()
        pred_shuf = model(torch.tensor(x_shuffled).to(device)).cpu().numpy()
    mean_abs_diff = np.mean(np.abs(pred_orig - pred_shuf))
    print(f"  Mean |prediction change| after shuffling history order: {mean_abs_diff:.4f}")
    if mean_abs_diff < 1e-4:
        print("  WARNING: predictions barely changed — the sequence may not be used meaningfully.")
    else:
        print("  OK: the model's prediction depends on the ORDER of past days, "
              "confirming real temporal memory is being used (not just a feed-forward net).")

    # ── Feature Saliency ──────────────────────────────────────────────
    print("\nComputing Gradient Saliency Importance...")
    sample_x = torch.tensor(X_train_raw[:1000], dtype=torch.float32).to(device)
    sample_x.requires_grad_(True)
    model.zero_grad()
    out = model(sample_x).sum()
    out.backward()
    saliency = torch.mean(torch.abs(sample_x.grad), dim=[0, 1]).detach().cpu().numpy()

    df_fi = pd.DataFrame({"Feature": feat_cols, "Importance": saliency}).sort_values(
        "Importance", ascending=False)
    df_fi.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    print("\nTop 10 features by Gradient Saliency:")
    print(df_fi.head(10).to_string(index=False))

    config = {
        "architecture": "LSTM(2 layers, hidden=64) -> FC(32) -> 1",
        "sequence_length_days": SEQ_LEN,
        "sequence_note": "Window [t-27..t] inclusive; current day's own exogenous "
                          "features (price/promo/calendar) are included at the last "
                          "timestep, target is that day's QUANTITY. Missing days "
                          "(gaps or insufficient history) are zero-padded.",
        "n_features": len(feat_cols),
        "shuffle_sanity_check_mean_abs_pred_change": round(float(mean_abs_diff), 6),
    }
    with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ DemandLSTM (real sequence model) complete! Outputs saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
