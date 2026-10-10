"""
Predict-only regeneration of pred_dev.csv, pred_oot_01.csv, pred_oot_02.csv.

Why: validation/run_validation.py's re-inference check found these three
files don't reproduce from the currently-saved XGBoost.pkl (large deltas /
a dtype crash), while pred_oos.csv, pred_oot_03/04/05.csv, and
performance.csv all reproduce exactly. File mtimes confirm it: the model
pkl, the code, and the raw sample data are all from 2026-04-27 16:03-16:18
(the original training run); these three prediction files were each
individually overwritten later (2026-05-31, 2026-05-31, 2026-07-27) without
a full pipeline re-run, so they no longer match the current model/data.

This script does NOT retrain — it loads the existing XGBoost.pkl (model +
label_encoder + features, unchanged) and re-derives predictions for
dev/oot_01/oot_02 from the untouched raw samples, using the exact same
preprocess() logic as xgboost_model.py (FILL_VALUES, weeks_since/to_promo
clipping, Segment label-encoding via the SAME saved encoder — never refit).
Also refreshes the corresponding rows in performance.csv, which were
computed from the same stale files.
"""
import pickle
import numpy as np
import pandas as pd

MODEL_NAME = "XGBoost"
OUT_DIR    = f"../outputs/{MODEL_NAME}"
TARGET     = "UnitSales"

# Must match xgboost_model.py exactly — this is a predict-only reproduction
# of that pipeline's preprocessing, not a redesign of it.
FILL_VALUES = {
    "lag_1": 0, "lag_2": 0, "lag_4": 0,
    "roll_mean_4": 0, "roll_mean_13": 0, "roll_std_4": 0,
    "wow_growth": 0, "momentum_4_13": 1,
    "promo_density_13": 0, "cust_hist_mean": 0,
    "zero_frac_13": 0, "zero_streak": 0,
    "weeks_since_promo": 52, "weeks_to_next_promo": 52,
}


def preprocess(df: pd.DataFrame, le, features: list) -> pd.DataFrame:
    df = df.copy()
    for col, val in FILL_VALUES.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    for col in ["weeks_since_promo", "weeks_to_next_promo"]:
        if col in df.columns:
            df[col] = df[col].clip(upper=52)
    df["Segment_enc"] = le.transform(df["Segment"])   # saved encoder — never refit
    feat_cols = [f for f in features if f in df.columns]
    df = df.dropna(subset=feat_cols + [TARGET])
    return df


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.clip(y_pred, 0, None)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae  = np.mean(np.abs(y_true - y_pred))
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "MAPE%": round(mape, 2)}


def save_full_predictions(model, features, df: pd.DataFrame, label: str) -> dict:
    out = df.copy()
    out["pred_xgb"] = model.predict(df[features]).clip(0).round(2)
    path = f"{OUT_DIR}/pred_{label}.csv"
    out.to_csv(path, index=False)
    m = metrics(df[TARGET].values, out["pred_xgb"].values)
    print(f"  pred_{label}.csv  |  rows={len(out):>7,}  |  "
          f"RMSE={m['RMSE']:>8.2f}  MAE={m['MAE']:>7.2f}  MAPE={m['MAPE%']:>6.2f}%")
    return {"sample": label, **m}


def main():
    with open(f"{OUT_DIR}/{MODEL_NAME}.pkl", "rb") as f:
        pkg = pickle.load(f)
    model, le, features = pkg["model"], pkg["label_encoder"], pkg["features"]
    print(f"Loaded {MODEL_NAME}.pkl — {type(model).__name__}, {len(features)} features (not retraining)")

    raw_dev = pd.read_csv("../data/samples/dev.csv")
    raw_oot = pd.read_csv("../data/samples/oot.csv")

    dev = preprocess(raw_dev, le, features)
    oot = preprocess(raw_oot, le, features)

    print(f"\nDEV : {len(dev):>7,} rows")
    print(f"OOT : {len(oot):>7,} rows (buckets 1-2 only will be regenerated)")

    print("\n── Regenerating stale predictions (predict-only) ─────────────")
    refreshed = {}
    refreshed["dev"] = save_full_predictions(model, features, dev, "dev")

    for bucket_id in (1, 2):
        bucket = oot[oot["oot_bucket"] == bucket_id]
        label = f"oot_{bucket_id:02d}"
        refreshed[label] = save_full_predictions(model, features, bucket, label)

    # Refresh the matching rows in performance.csv (computed from the same
    # stale files) — leave oos/oot_03/04/05 rows exactly as they were.
    perf_path = f"{OUT_DIR}/performance.csv"
    perf = pd.read_csv(perf_path)
    for label, m in refreshed.items():
        mask = perf["sample"] == label
        if mask.any():
            perf.loc[mask, ["RMSE", "MAE", "MAPE%"]] = [m["RMSE"], m["MAE"], m["MAPE%"]]
        else:
            perf = pd.concat([perf, pd.DataFrame([m])], ignore_index=True)
    perf.to_csv(perf_path, index=False)
    print(f"\nperformance.csv refreshed for: {', '.join(refreshed)}")
    print(perf.to_string(index=False))


if __name__ == "__main__":
    main()
