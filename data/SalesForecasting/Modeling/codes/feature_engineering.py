import pandas as pd
import numpy as np
from scipy import stats

# ── 0. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv("SalesForecasting.csv")

# ── 1. Fix WeekNum > 52 ───────────────────────────────────────────────────────
rng = np.random.default_rng(42)
mask = df["WeekNum"] > 52
df.loc[mask, "WeekNum"] = rng.integers(1, 53, size=mask.sum()).astype(float)
df["WeekNum"] = df["WeekNum"].round().astype(int)

# ── 2. Global time index (continuous across years) ───────────────────────────
df["global_t"] = (df["Year"] - df["Year"].min()) * 52 + df["WeekNum"]

# Sort chronologically within each Segment+Customer series
df = df.sort_values(["Segment", "Cust_ID", "global_t"]).reset_index(drop=True)

GRAIN = ["Segment", "Cust_ID"]

# ── 3. Time-based / calendar features ────────────────────────────────────────
df["week_sin"] = np.sin(2 * np.pi * df["WeekNum"] / 52)
df["week_cos"] = np.cos(2 * np.pi * df["WeekNum"] / 52)
df["quarter"] = np.ceil(df["WeekNum"] / 13).astype(int).clip(1, 4)
df["is_year_start"] = (df["WeekNum"] <= 4).astype(int)
df["is_year_end"]   = (df["WeekNum"] >= 49).astype(int)

# ── 4. Lag features ───────────────────────────────────────────────────────────
grp = df.groupby(GRAIN)["UnitSales"]

for lag in [1, 2, 3, 4, 8, 13, 26, 52]:
    df[f"lag_{lag}"] = grp.shift(lag)

# ── 5. Rolling window features ────────────────────────────────────────────────
def roll(series_grp, window, func, min_p=1):
    return series_grp.transform(
        lambda x: x.shift(1).rolling(window, min_periods=min_p).agg(func)
    )

sales_grp = df.groupby(GRAIN)["UnitSales"]

for w in [4, 13, 26]:
    df[f"roll_mean_{w}"]   = roll(sales_grp, w, "mean")
    df[f"roll_std_{w}"]    = roll(sales_grp, w, "std")
    df[f"roll_max_{w}"]    = roll(sales_grp, w, "max")

df["roll_min_4"]  = roll(sales_grp, 4,  "min")
df["roll_min_13"] = roll(sales_grp, 13, "min")

# ── 6. Growth & trend features ────────────────────────────────────────────────
eps = 1e-6

df["wow_growth"]       = (df["lag_1"] - df["lag_2"]) / (df["lag_2"].abs() + eps)
df["yoy_growth"]       = (df["lag_1"] - df["lag_52"]) / (df["lag_52"].abs() + eps)

# Momentum: short-term mean vs medium-term mean
df["momentum_4_13"]  = df["roll_mean_4"]  / (df["roll_mean_13"]  + eps)
df["momentum_4_26"]  = df["roll_mean_4"]  / (df["roll_mean_26"]  + eps)
df["momentum_13_26"] = df["roll_mean_13"] / (df["roll_mean_26"]  + eps)

# Coefficient of variation (short & medium)
df["cv_4"]  = df["roll_std_4"]  / (df["roll_mean_4"]  + eps)
df["cv_13"] = df["roll_std_13"] / (df["roll_mean_13"] + eps)

# Rolling linear trend slope over 4 and 13 weeks
def rolling_slope(x, window):
    result = np.full(len(x), np.nan)
    arr = x.values
    for i in range(window - 1, len(arr)):
        y = arr[i - window + 1 : i + 1]
        if np.isnan(y).any():
            continue
        t = np.arange(window)
        slope, *_ = np.polyfit(t, y, 1)
        result[i] = slope
    return pd.Series(result, index=x.index)

# Compute slope on lagged series (no look-ahead)
def rolling_slope_transform(grp, window):
    return grp.transform(lambda x: rolling_slope(x.shift(1), window))

df["trend_slope_4"]  = rolling_slope_transform(df.groupby(GRAIN)["UnitSales"], 4)
df["trend_slope_13"] = rolling_slope_transform(df.groupby(GRAIN)["UnitSales"], 13)

# ── 7. Promotion features ─────────────────────────────────────────────────────
df["is_promo"]        = (df["PromoWeek"] > 0).astype(int)
df["promo_intensity"] = df["PromoWeek"]           # raw value preserved

# Pre-promo windows: will next 1 or 2 weeks be a promo?
promo_grp = df.groupby(GRAIN)["is_promo"]
df["pre_promo_1"] = promo_grp.shift(-1).fillna(0).astype(int)
df["pre_promo_2"] = promo_grp.shift(-2).fillna(0).astype(int)

# Post-promo windows: was the previous 1 or 2 weeks a promo?
df["post_promo_1"] = promo_grp.shift(1).fillna(0).astype(int)
df["post_promo_2"] = promo_grp.shift(2).fillna(0).astype(int)

# Weeks since last promo and weeks to next promo
def weeks_since(series):
    """Distance (in rows) since last 1, forward-looking-safe."""
    out = []
    count = np.nan
    for v in series:
        if v == 1:
            count = 0
        elif not np.isnan(count):
            count += 1
        out.append(count)
    return out

def weeks_to(series):
    return list(reversed(weeks_since(list(reversed(series)))))

df["weeks_since_promo"] = (
    df.groupby(GRAIN)["is_promo"]
    .transform(lambda x: weeks_since(x.tolist()))
)
df["weeks_to_next_promo"] = (
    df.groupby(GRAIN)["is_promo"]
    .transform(lambda x: weeks_to(x.tolist()))
)

# Cumulative promo count per customer (up to but not including current row)
df["cum_promo_count"] = (
    df.groupby(GRAIN)["is_promo"]
    .transform(lambda x: x.shift(1).fillna(0).cumsum())
)

# Rolling promo density (fraction of last 13 weeks that were promos)
df["promo_density_13"] = (
    df.groupby(GRAIN)["is_promo"]
    .transform(lambda x: x.shift(1).rolling(13, min_periods=1).mean())
)

# ── 8. Customer & segment aggregate features ──────────────────────────────────
# Expanding historical mean/std per Segment+CustID (no look-ahead)
df["cust_hist_mean"] = (
    df.groupby(GRAIN)["UnitSales"]
    .transform(lambda x: x.shift(1).expanding().mean())
)
df["cust_hist_std"] = (
    df.groupby(GRAIN)["UnitSales"]
    .transform(lambda x: x.shift(1).expanding().std())
)
df["cust_hist_cv"] = df["cust_hist_std"] / (df["cust_hist_mean"] + eps)

# Segment-level weekly mean (all customers in segment, same week across years)
seg_week_mean = (
    df.groupby(["Segment", "WeekNum"])["UnitSales"].mean().rename("seg_week_mean")
)
df = df.join(seg_week_mean, on=["Segment", "WeekNum"])

# Customer sales relative to segment average
df["cust_vs_seg"] = df["cust_hist_mean"] / (df["seg_week_mean"] + eps)

# Segment-level expanding mean at current time step (peer context)
seg_t_mean = (
    df.groupby(["Segment", "global_t"])["UnitSales"].mean().rename("seg_t_mean")
)
df = df.join(seg_t_mean, on=["Segment", "global_t"])

# ── 9. Zero-sales / intermittency features ────────────────────────────────────
df["is_zero"] = (df["UnitSales"] == 0).astype(int)

def zero_streak(series):
    """Count of consecutive zeros ending at t-1."""
    out = []
    streak = 0
    for v in series:
        out.append(streak)
        streak = streak + 1 if v == 0 else 0
    return out

df["zero_streak"] = (
    df.groupby(GRAIN)["UnitSales"]
    .transform(lambda x: zero_streak(x.tolist()))
)

# Fraction of last 13 weeks that were zero (intermittency)
df["zero_frac_13"] = (
    df.groupby(GRAIN)["is_zero"]
    .transform(lambda x: x.shift(1).rolling(13, min_periods=1).mean())
)

# Fraction of last 26 weeks that were zero (longer memory)
df["zero_frac_26"] = (
    df.groupby(GRAIN)["is_zero"]
    .transform(lambda x: x.shift(1).rolling(26, min_periods=1).mean())
)

# ── 10. Holiday interaction features ─────────────────────────────────────────
holiday_cols = ["August", "Christmas", "Spring", "NewYear",
                "September", "Thanksgiving", "VetreansDay"]

# Any holiday signal active this week
df["any_holiday"] = (df[holiday_cols].abs().sum(axis=1) > 0).astype(int)

# Promo during holiday (uplift interaction)
df["promo_x_holiday"] = df["is_promo"] * df["any_holiday"]

# Weekly_Index interactions
df["promo_x_windex"]   = df["is_promo"]    * df["Weekly_Index"]
df["holiday_x_windex"] = df["any_holiday"] * df["Weekly_Index"]

# ── 11. Save ──────────────────────────────────────────────────────────────────
out_path = "SalesForecasting_features.csv"
df.drop(columns=["is_zero"]).to_csv(out_path, index=False)

print("=" * 60)
print(f"Output shape : {df.shape}")
print(f"Saved to     : {out_path}")
print(f"\nFeatures added ({df.shape[1] - 14}):")
original = {'Segment','Cust_ID','Year','WeekNum','UnitSales','August','Christmas',
            'Spring','NewYear','September','Thanksgiving','VetreansDay',
            'Weekly_Index','PromoWeek'}
new_cols = [c for c in df.columns if c not in original]
for c in new_cols:
    print(f"  {c}")

print(f"\nMissing values in new features:")
miss = df[new_cols].isnull().sum()
print(miss[miss > 0].to_string())
