"""
Feature Engineering Script — FraudPrediction
Applies Groups A–E of engineered features to all Modeling_Data CSVs.
Saves enriched datasets to Modeling_Data_Enriched/
"""
import pandas as pd
import numpy as np
import os

WINDOWS = [3, 7, 15, 21, 30, 45, 60, 75, 90, 120, 150, 180, 365]

def wstr(w):
    return f"{w:03d}" if w < 100 else str(w)

def add_engineered_features(df):
    """Add all 91 new engineered features in-place."""

    # ── Group A: Buy/Sell Imbalance (2 × 13 = 26 features) ──────────
    for w in WINDOWS:
        ws = wstr(w)
        b = f"Trade_bAmt_l_{ws}d"
        s = f"Trade_sAmt_l_{ws}d"
        n = f"Trade_numbers_l_{ws}d"
        df[f"BS_ratio_l_{ws}d"]       = df[b] / (df[s] + 1)
        df[f"net_trade_amt_l_{ws}d"]  = df[b] - df[s]

    # ── Group B: Behavioural Shift / Acceleration (8 features) ───────
    cross_pairs = [(3, 30), (7, 90), (30, 180), (90, 365)]
    for w_short, w_long in cross_pairs:
        ss, sl = wstr(w_short), wstr(w_long)
        df[f"activity_shift_{w_short}v{w_long}d"] = (
            df[f"Trade_numbers_l_{ss}d"] /
            (df[f"Trade_numbers_l_{sl}d"] + 1)
        )
        df[f"amount_shift_{w_short}v{w_long}d"] = (
            df[f"Trade_totalAmt_l_{ss}d"] /
            (df[f"Trade_totalAmt_l_{sl}d"] + 1)
        )

    # ── Group C: Script Concentration Volatility (2 × 13 = 26) ───────
    for w in WINDOWS:
        ws = wstr(w)
        df[f"bscript_range_l_{ws}d"]    = (
            df[f"maxBscript_l_{ws}d"] - df[f"minBscript_l_{ws}d"]
        )
        df[f"bscript_bs_ratio_l_{ws}d"] = (
            df[f"BscriptAvg_l_{ws}d"] / (df[f"SscriptAvg_l_{ws}d"] + 1)
        )

    # ── Group D: Dormancy & Recency Engagement (5 features) ──────────
    df["dormancy_ratio"]          = df["cash_dormancyPeriod"] / (df["cash_membershipLength"] + 1)
    df["trade_intensity"]         = df["cash_daysTraded"] / (df["cash_liveDays"] + 1)
    df["recent_activity_ratio_30d"] = df["Trade_days_l_030d"] / (df["cash_daysTraded"] + 1)
    df["recent_amount_ratio_30d"] = (
        df["Trade_totalAmt_l_030d"] / (df["cash_Td_totalAmt"] + 1)
    )
    df["recent_amount_ratio_90d"] = (
        df["Trade_totalAmt_l_090d"] / (df["cash_Td_totalAmt"] + 1)
    )

    # ── Group E: Per-Trade Size Consistency (2 × 13 = 26 features) ───
    for w in WINDOWS:
        ws = wstr(w)
        n = f"Trade_numbers_l_{ws}d"
        df[f"avg_buy_size_l_{ws}d"]  = df[f"Trade_bAmt_l_{ws}d"]  / (df[n] + 1)
        df[f"avg_sell_size_l_{ws}d"] = df[f"Trade_sAmt_l_{ws}d"] / (df[n] + 1)

    return df


def run():
    src_dir  = "/Users/somyabhadada/nodex8/data/TradeFraud/Modeling_Data/"
    dest_dir = "/Users/somyabhadada/nodex8/data/TradeFraud/Modeling_Data_Enriched/"
    datasets = [
        "train.csv", "val_oos.csv", "internal_oot.csv",
        "oot_202510.csv", "oot_202511.csv", "oot_202512.csv"
    ]

    for fname in datasets:
        src  = os.path.join(src_dir, fname)
        dest = os.path.join(dest_dir, fname)
        print(f"Processing {fname}...", end=" ", flush=True)
        df = pd.read_csv(src)
        original_cols = len(df.columns)
        df = add_engineered_features(df)
        new_cols = len(df.columns) - original_cols
        df.to_csv(dest, index=False)
        n_d = (df["Dispute"] == "D").sum()
        print(f"✓  {len(df):>7,} rows | +{new_cols} features → {len(df.columns)} total | Disputes: {n_d}")

    # Verify new feature count
    df_check = pd.read_csv(os.path.join(dest_dir, "train.csv"), nrows=1)
    print(f"\nEnriched train columns: {len(df_check.columns)}")
    new_feat_names = [c for c in df_check.columns if c not in
                      pd.read_csv(os.path.join(src_dir, "train.csv"), nrows=1).columns]
    print(f"New features added ({len(new_feat_names)}): first 10 → {new_feat_names[:10]}")
    print("Done.")


if __name__ == "__main__":
    run()
