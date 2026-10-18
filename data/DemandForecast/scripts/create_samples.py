"""
Data splitting script for DemandForecast.
Creates DEV, OOS, and OOT splits based on the detailed plan.

Phase 0 fix applied (see ARCHITECTURE.md Section 9): the OOT cutoff used to
be "max_date - 60 days", which does not land on a calendar-month boundary.
With this data's max date of 27-Feb-2013, that cut the resulting "Dec 2012"
OOT bucket down to its last 2 days (30-31 Dec) instead of the full month,
making any metric on it essentially meaningless. The cutoff is now the start
of the last three full calendar months present in the data, so OOT becomes
Dec 2012 (full month), Jan 2013 (full month), and Feb 2013 (partial, through
the 27th — the dataset simply ends there, which is unavoidable).
"""
import os
import pandas as pd

DATA_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast"
OUTPUT_DIR = os.path.join(DATA_DIR, "samples")

# Number of full calendar months (counted back from the month before the
# data's max date) to hold out as OOT buckets.
N_OOT_MONTHS = 3

def create_splits():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading features.csv...")
    df = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 1. Time-based OOT split, on calendar-month boundaries.
    # OOT = the last N_OOT_MONTHS full calendar months that start before the
    # data's max date (so the last, necessarily partial, month the dataset
    # ends in is still included as one OOT bucket — it is simply short).
    max_date = df["Date"].max()
    last_full_month_start = max_date.to_period("M").to_timestamp()
    cutoff_date = (last_full_month_start - pd.DateOffset(months=N_OOT_MONTHS - 1)) - pd.Timedelta(days=1)

    print(f"Max date: {max_date.date()}, OOT cutoff (calendar-month boundary): {cutoff_date.date()}")

    in_time_mask = df["Date"] <= cutoff_date
    df_in_time = df[in_time_mask].copy()
    df_oot = df[~in_time_mask].copy()

    print(f"In-time rows: {len(df_in_time)}, OOT rows: {len(df_oot)}")
    
    # 2. Entity-based DEV/OOS Split
    # Entity is BRAND x COMP_STR_NO
    df_in_time["Entity"] = df_in_time["BRAND"].astype(str) + "_" + df_in_time["COMP_STR_NO"].astype(str)
    unique_entities = df_in_time["Entity"].unique()
    
    # 70% DEV, 30% OOS
    import numpy as np
    np.random.seed(42)
    shuffled_entities = np.random.permutation(unique_entities)
    split_idx = int(len(shuffled_entities) * 0.7)
    
    dev_entities = set(shuffled_entities[:split_idx])
    oos_entities = set(shuffled_entities[split_idx:])
    
    df_dev = df_in_time[df_in_time["Entity"].isin(dev_entities)].drop(columns=["Entity"])
    df_oos = df_in_time[df_in_time["Entity"].isin(oos_entities)].drop(columns=["Entity"])
    
    print(f"DEV rows: {len(df_dev)}, OOS rows: {len(df_oos)}")
    
    # 3. OOT Buckets (monthly)
    df_oot["YearMonth"] = df_oot["Date"].dt.to_period("M")
    oot_months = sorted(df_oot["YearMonth"].unique())
    
    # Save splits
    df_dev.to_csv(os.path.join(OUTPUT_DIR, "dev.csv"), index=False)
    df_oos.to_csv(os.path.join(OUTPUT_DIR, "oos.csv"), index=False)
    
    summary = []
    summary.append({"Sample": "DEV", "Rows": len(df_dev)})
    summary.append({"Sample": "OOS", "Rows": len(df_oos)})
    
    for i, ym in enumerate(oot_months):
        bucket_df = df_oot[df_oot["YearMonth"] == ym].drop(columns=["YearMonth"])
        fname = f"oot_{i+1:02d}_{ym.strftime('%Y%m')}.csv"
        bucket_df.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
        print(f"Saved {fname} ({len(bucket_df)} rows)")
        summary.append({"Sample": fname, "Rows": len(bucket_df)})
        
    pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_DIR, "split_summary.csv"), index=False)
    print(f"All splits saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    create_splits()
