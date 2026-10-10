"""
Panel Data Model — Fixed Effects LSDV (Segment as panel unit).
Predictions saved at CustID level with all original columns retained.
"""

import os, warnings, pickle
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

MODEL_NAME = "FE_LSDV"
OUT_DIR    = f"../outputs/{MODEL_NAME}"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 0. Config ─────────────────────────────────────────────────────────────────
TARGET     = "UnitSales"
LOG_TARGET = "log1p_sales"

FEATURES = [
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

# ── 1. Helpers ────────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[LOG_TARGET] = np.log1p(df[TARGET])
    for col, val in FILL_VALUES.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    for col in ["weeks_since_promo", "weeks_to_next_promo"]:
        if col in df.columns:
            df[col] = df[col].clip(upper=52)
    return df.dropna(subset=FEATURES + [LOG_TARGET])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred).clip(0)
    rmse   = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae    = np.mean(np.abs(y_true - y_pred))
    mask   = y_true > 0
    mape   = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "MAPE%": round(mape, 2)}


def predict(model, df: pd.DataFrame) -> np.ndarray:
    return np.expm1(model.predict(df)).clip(0)


# ── 2. Load & preprocess ──────────────────────────────────────────────────────
dev = preprocess(pd.read_csv("../data/samples/dev.csv"))
oos = preprocess(pd.read_csv("../data/samples/oos.csv"))
oot = preprocess(pd.read_csv("../data/samples/oot.csv"))

print(f"DEV : {len(dev):>7,} rows")
print(f"OOS : {len(oos):>7,} rows")
print(f"OOT : {len(oot):>7,} rows")

# ── 3. Fit Fixed Effects (LSDV) on DEV ───────────────────────────────────────
fe_formula = f"{LOG_TARGET} ~ C(Segment) + {' + '.join(FEATURES)}"
fe = smf.ols(fe_formula, data=dev).fit(cov_type="HC3")

print("\n── Fixed Effects Model Summary ─────────────────────────────")
print(fe.summary2().tables[0])

print("\nSegment fixed effects (Bronze = reference):")
seg_fx = fe.params.filter(like="C(Segment)").rename(
    lambda x: x.replace("C(Segment)[T.", "").replace("]", "")
)
print(seg_fx.round(4).to_string())

print(f"\nAdj. R²  : {fe.rsquared_adj:.4f}")
print(f"AIC      : {fe.aic:.1f}")
print(f"N (DEV)  : {int(fe.nobs):,}")

# ── 4. Save model pickle + DataFrame metadata ─────────────────────────────────
# Wrap statsmodels result in sklearn interface so SHAP explainers accept it.
# .params is preserved so shap_utils.py can still read result.params.index.
_skmodel = LinearRegression()
_skmodel.coef_      = np.array([fe.params.get(f, 0.0) for f in FEATURES])
_skmodel.intercept_ = float(fe.params.get("Intercept", 0.0))
_skmodel.params     = fe.params

pkl_path = f"{OUT_DIR}/{MODEL_NAME}.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump({
        "model": _skmodel,
        "features": FEATURES,
        "target": TARGET,
        "log_target": LOG_TARGET,
        "training_data": {
            "shape": dev.shape,
            "columns": dev.columns.tolist(),
            "dtypes": dev.dtypes.astype(str).to_dict(),
            "observations": len(dev),
        },
    }, f)
print(f"\nModel saved : {pkl_path}")

# ── 5. Save coefficient table ─────────────────────────────────────────────────
coef_df = pd.DataFrame({
    "variable"   : fe.params.index,
    "coefficient": fe.params.values,
    "std_err"    : fe.bse.values,
    "t_stat"     : fe.tvalues.values,
    "p_value"    : fe.pvalues.values,
    "ci_lower"   : fe.conf_int()[0].values,
    "ci_upper"   : fe.conf_int()[1].values,
}).round(6)
coef_path = f"{OUT_DIR}/coefficients.csv"
coef_df.to_csv(coef_path, index=False)
print(f"Coefficients: {coef_path}  ({len(coef_df)} parameters)")

# ── 6. Prediction files — full data + prediction column ──────────────────────
def save_full_predictions(df: pd.DataFrame, label: str):
    out = df.copy()
    out["pred_fe"] = predict(fe, df).round(2)
    path = f"{OUT_DIR}/pred_{label}.csv"
    out.to_csv(path, index=False)
    m = metrics(df[LOG_TARGET].values, np.log1p(out["pred_fe"].values))
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

# ── 7. Performance summary ────────────────────────────────────────────────────
perf_df = pd.DataFrame(perf)
perf_path = f"{OUT_DIR}/performance.csv"
perf_df.to_csv(perf_path, index=False)

print(f"\nPerformance saved : {perf_path}")
print("\n── Summary ─────────────────────────────────────────────────")
print(perf_df.to_string(index=False))
print(f"\nAll outputs in: {OUT_DIR}/")
