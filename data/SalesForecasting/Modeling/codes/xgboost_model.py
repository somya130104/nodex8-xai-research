"""
XGBoost Model — Tweedie objective (zero-inflated, right-skewed sales).
Segment label-encoded as a feature.
Time-based train/val split within DEV for early stopping.
Predictions at CustID level with all original columns retained.
"""

import os, warnings, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

MODEL_NAME = "XGBoost"
OUT_DIR    = f"../outputs/{MODEL_NAME}"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 0. Config ─────────────────────────────────────────────────────────────────
TARGET  = "UnitSales"
EARLY_STOP_WEEKS = 20        # last N weeks of DEV held out for early stopping

FEATURES = [
    # Identifiers used as features
    "Segment_enc",
    # Time
    "global_t", "week_sin", "week_cos", "quarter", "is_year_start", "is_year_end",
    # Autoregressive
    "lag_1", "lag_2", "lag_4",
    # Rolling level & volatility
    "roll_mean_4", "roll_mean_13", "roll_std_4",
    # Growth / momentum
    "wow_growth", "momentum_4_13",
    # Promotion
    "is_promo", "promo_intensity",
    "pre_promo_1", "pre_promo_2", "post_promo_1", "post_promo_2",
    "weeks_since_promo", "weeks_to_next_promo", "promo_density_13",
    # Holidays
    "August", "Christmas", "Spring", "NewYear",
    "September", "Thanksgiving", "VetreansDay", "any_holiday",
    # Interactions
    "promo_x_holiday", "promo_x_windex", "holiday_x_windex",
    # Market index
    "Weekly_Index",
    # Customer history
    "cust_hist_mean", "zero_streak", "zero_frac_13",
]

FILL_VALUES = {
    "lag_1": 0, "lag_2": 0, "lag_4": 0,
    "roll_mean_4": 0, "roll_mean_13": 0, "roll_std_4": 0,
    "wow_growth": 0, "momentum_4_13": 1,
    "promo_density_13": 0, "cust_hist_mean": 0,
    "zero_frac_13": 0, "zero_streak": 0,
    "weeks_since_promo": 52, "weeks_to_next_promo": 52,
}

XGB_PARAMS = {
    "objective"             : "reg:tweedie",
    "tweedie_variance_power": 1.5,          # 1=Poisson, 2=Gamma; 1.5 balances both
    "n_estimators"          : 1000,
    "learning_rate"         : 0.05,
    "max_depth"             : 6,
    "min_child_weight"      : 10,
    "subsample"             : 0.8,
    "colsample_bytree"      : 0.8,
    "reg_alpha"             : 0.1,          # L1
    "reg_lambda"            : 1.0,          # L2
    "random_state"          : 42,
    "n_jobs"                : -1,
    "verbosity"             : 0,
    "early_stopping_rounds" : 30,
    "eval_metric"           : "tweedie-nloglik@1.5",
}

# ── 1. Helpers ────────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame, le: LabelEncoder = None):
    df = df.copy()
    for col, val in FILL_VALUES.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    for col in ["weeks_since_promo", "weeks_to_next_promo"]:
        if col in df.columns:
            df[col] = df[col].clip(upper=52)
    # Label-encode Segment
    if le is None:
        le = LabelEncoder()
        df["Segment_enc"] = le.fit_transform(df["Segment"])
    else:
        df["Segment_enc"] = le.transform(df["Segment"])
    feat_cols = [f for f in FEATURES if f in df.columns]
    df = df.dropna(subset=feat_cols + [TARGET])
    return df, le


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.clip(y_pred, 0, None)
    rmse   = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae    = np.mean(np.abs(y_true - y_pred))
    mask   = y_true > 0
    mape   = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "MAPE%": round(mape, 2)}


# ── 2. Load & preprocess ──────────────────────────────────────────────────────
raw_dev = pd.read_csv("../data/samples/dev.csv")
raw_oos = pd.read_csv("../data/samples/oos.csv")
raw_oot = pd.read_csv("../data/samples/oot.csv")

dev, le = preprocess(raw_dev)
oos, _  = preprocess(raw_oos, le)
oot, _  = preprocess(raw_oot, le)

print(f"DEV : {len(dev):>7,} rows")
print(f"OOS : {len(oos):>7,} rows")
print(f"OOT : {len(oot):>7,} rows")
print(f"Segment encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── 3. Time-based train / val split within DEV ───────────────────────────────
val_cutoff = dev["global_t"].max() - EARLY_STOP_WEEKS
train = dev[dev["global_t"] <= val_cutoff]
val   = dev[dev["global_t"] >  val_cutoff]

X_train, y_train = train[FEATURES], train[TARGET]
X_val,   y_val   = val[FEATURES],   val[TARGET]

print(f"\nTrain : {len(train):,} rows  (global_t ≤ {int(val_cutoff)})")
print(f"Val   : {len(val):,} rows  (global_t >  {int(val_cutoff)})")

# ── 4. Fit XGBoost ────────────────────────────────────────────────────────────
print("\n── Training XGBoost (Tweedie) ──────────────────────────────")
model = xgb.XGBRegressor(**XGB_PARAMS)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)
print(f"\nBest iteration : {model.best_iteration}")
print(f"Best val score : {model.best_score:.6f}")

# ── 5. Save model pickle ──────────────────────────────────────────────────────
pkl_path = f"{OUT_DIR}/{MODEL_NAME}.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump({"model": model, "label_encoder": le, "features": FEATURES}, f)
print(f"Model saved    : {pkl_path}")

# ── 6. Feature importance ─────────────────────────────────────────────────────
imp = pd.DataFrame({
    "feature"    : FEATURES,
    "importance_gain"  : model.get_booster().get_score(importance_type="gain").get(f, 0)
                         if False else [
                             model.get_booster().get_score(importance_type="gain").get(f, 0)
                             for f in FEATURES
                         ],
    "importance_weight": [
        model.get_booster().get_score(importance_type="weight").get(f, 0)
        for f in FEATURES
    ],
    "importance_cover" : [
        model.get_booster().get_score(importance_type="cover").get(f, 0)
        for f in FEATURES
    ],
}).sort_values("importance_gain", ascending=False).reset_index(drop=True)

imp_path = f"{OUT_DIR}/feature_importance.csv"
imp.to_csv(imp_path, index=False)

print(f"\nTop 15 features by gain:")
print(imp.head(15)[["feature", "importance_gain"]].to_string(index=False))

# ── 7. Prediction files — full data + prediction column ──────────────────────
def save_full_predictions(df: pd.DataFrame, label: str):
    out           = df.copy()
    out["pred_xgb"] = model.predict(df[FEATURES]).clip(0).round(2)
    path          = f"{OUT_DIR}/pred_{label}.csv"
    out.to_csv(path, index=False)
    m = metrics(df[TARGET].values, out["pred_xgb"].values)
    print(f"  pred_{label}.csv  |  rows={len(out):>7,}  |  "
          f"RMSE={m['RMSE']:>8.2f}  MAE={m['MAE']:>7.2f}  MAPE={m['MAPE%']:>6.2f}%")
    return {"sample": label, **m}

print("\n── Predictions ─────────────────────────────────────────────")
perf = []
perf.append(save_full_predictions(dev, "dev"))
perf.append(save_full_predictions(oos, "oos"))

for bucket_id in sorted(oot["oot_bucket"].unique()):
    bucket = oot[oot["oot_bucket"] == bucket_id]
    perf.append(save_full_predictions(bucket, f"oot_{int(bucket_id):02d}"))

# ── 8. Performance summary ────────────────────────────────────────────────────
perf_df = pd.DataFrame(perf)
perf_path = f"{OUT_DIR}/performance.csv"
perf_df.to_csv(perf_path, index=False)

print(f"\nPerformance saved : {perf_path}")
print("\n── Summary ─────────────────────────────────────────────────")
print(perf_df.to_string(index=False))
print(f"\nAll outputs in: {OUT_DIR}/")
