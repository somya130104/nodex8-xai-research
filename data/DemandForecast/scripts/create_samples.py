"""
Data splitting script for DemandForecast.
Creates DEV, OOS, and OOT splits based on the detailed plan.
"""
import os
import pandas as pd

DATA_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast"
OUTPUT_DIR = os.path.join(DATA_DIR, "samples")

def create_splits():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading features.csv...")
    df = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    
    # 1. Time-based OOT Split
    # The data is from Oct 2012 to roughly mid 2014.
    # Let's use the last 60 days as OOT.
    max_date = df["Date"].max()
    cutoff_date = max_date - pd.Timedelta(days=60)
    
    print(f"Max date: {max_date.date()}, Cutoff date (max - 60 days): {cutoff_date.date()}")
    
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
