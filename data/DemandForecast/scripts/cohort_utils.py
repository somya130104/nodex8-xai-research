"""Small shared helper for DemandForecast's cohort definitions (Phase 1.2),
used by both run_baseline_shap.py and run_baseline_lime.py so the two don't
define the day-type bucketing differently by accident."""
import pandas as pd


def assign_day_type(df):
    """One mutually-exclusive bucket per row: Holiday > Promotion > Weekend > Weekday."""
    is_holiday = pd.to_numeric(df["holiday"], errors="coerce").fillna(0) == 1
    is_promo = df["PROMOTION_FLG"].astype(str).str.upper().isin(["Y", "1"])
    is_weekend = pd.to_numeric(df["is_weekend"], errors="coerce").fillna(0) == 1
    day_type = pd.Series("Weekday", index=df.index)
    day_type[is_weekend] = "Weekend"
    day_type[is_promo] = "Promotion"
    day_type[is_holiday] = "Holiday"
    return day_type
