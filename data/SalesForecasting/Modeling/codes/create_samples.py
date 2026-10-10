import pandas as pd
import numpy as np
import os

# ── 0. Load feature-engineered data ──────────────────────────────────────────
df = pd.read_csv("../data/SalesForecasting_features.csv")

os.makedirs("../data/samples", exist_ok=True)

# ── 1. Time split: first 3 years (dev) vs OOT period ────────────────────────
# DEV: global_t 1–156  (Wk 1 2023 → Wk 52 2025, 3 full years)
# OOT: global_t 157+   (2026 onward), split into 12-week buckets
DEV_CUTOFF = 156

in_time = df[df["global_t"] <= DEV_CUTOFF].copy()
oot_all  = df[df["global_t"] >  DEV_CUTOFF].copy()

# ── 2. DEV / OOS split (customer-level, global 80/20) ────────────────────────
# Split at Cust_ID level so each customer's rows land entirely in DEV or OOS
# across all segments — no customer appears in both samples.
rng = np.random.default_rng(42)

all_custs = in_time["Cust_ID"].unique().copy()
rng.shuffle(all_custs)
cut = int(len(all_custs) * 0.70)
dev_custs = set(all_custs[:cut])
oos_custs = set(all_custs[cut:])

dev = in_time[in_time["Cust_ID"].isin(dev_custs)].copy()
oos = in_time[in_time["Cust_ID"].isin(oos_custs)].copy()

# ── 3. OOT buckets — 4-week rolling windows ───────────────────────────────────
oot_weeks = sorted(oot_all["global_t"].unique())          # 77 weeks
bucket_size = 12

oot_buckets = {}
for i, start in enumerate(range(0, len(oot_weeks), bucket_size), start=1):
    weeks = oot_weeks[start : start + bucket_size]
    bucket_df = oot_all[oot_all["global_t"].isin(weeks)].copy()
    bucket_df["oot_bucket"] = i
    oot_buckets[i] = bucket_df

oot_combined = pd.concat(oot_buckets.values(), ignore_index=True)

# ── 4. Save ───────────────────────────────────────────────────────────────────
dev.to_csv("../data/samples/dev.csv",          index=False)
oos.to_csv("../data/samples/oos.csv",          index=False)
oot_combined.to_csv("../data/samples/oot.csv", index=False)

# Also save individual OOT bucket files
for i, bucket_df in oot_buckets.items():
    bucket_df.to_csv(f"../data/samples/oot_{i:02d}.csv", index=False)

# ── 5. Summary report ────────────────────────────────────────────────────────
print("=" * 65)
print(f"{'Sample':<20} {'Rows':>8} {'Customers':>10} {'Weeks':>8}")
print("-" * 65)
print(f"{'DEV':<20} {len(dev):>8,} {dev['Cust_ID'].nunique():>10,} {dev['global_t'].nunique():>8}")
print(f"{'OOS':<20} {len(oos):>8,} {oos['Cust_ID'].nunique():>10,} {oos['global_t'].nunique():>8}")
print(f"{'OOT (all buckets)':<20} {len(oot_combined):>8,} {oot_combined['Cust_ID'].nunique():>10,} {oot_combined['global_t'].nunique():>8}")
print("-" * 65)

print(f"\nTime boundaries:")
print(f"  DEV / OOS  : global_t  1 – {DEV_CUTOFF}  "
      f"(Wk {int(dev['WeekNum'].min())} {int(dev['Year'].min())} → "
      f"Wk {int(dev['WeekNum'].max())} {int(dev['Year'].max())})")
print(f"  OOT period : global_t {DEV_CUTOFF+1} – {oot_weeks[-1]}  "
      f"(Wk {int(oot_all['WeekNum'].min())} {int(oot_all['Year'].min())} → "
      f"Wk {int(oot_all['WeekNum'].max())} {int(oot_all['Year'].max())})")

print(f"\nOOT buckets ({len(oot_buckets)} × {bucket_size}-week windows):")
print(f"{'Bucket':<10} {'global_t range':<20} {'Year/Wk range':<28} {'Rows':>8}")
print("-" * 65)
for i, bucket_df in oot_buckets.items():
    gt_lo, gt_hi = bucket_df["global_t"].min(), bucket_df["global_t"].max()
    yr_lo = int(bucket_df.loc[bucket_df["global_t"]==gt_lo, "Year"].iloc[0])
    wk_lo = int(bucket_df.loc[bucket_df["global_t"]==gt_lo, "WeekNum"].iloc[0])
    yr_hi = int(bucket_df.loc[bucket_df["global_t"]==gt_hi, "Year"].iloc[0])
    wk_hi = int(bucket_df.loc[bucket_df["global_t"]==gt_hi, "WeekNum"].iloc[0])
    print(f"  OOT_{i:02d}   {gt_lo:>5} – {gt_hi:<10}   "
          f"Wk {wk_lo:>2} {yr_lo} → Wk {wk_hi:>2} {yr_hi}   {len(bucket_df):>8,}")

print(f"\nSegment distribution:")
print(f"\n  DEV:\n{dev.groupby('Segment')['Cust_ID'].nunique().to_string()}")
print(f"\n  OOS:\n{oos.groupby('Segment')['Cust_ID'].nunique().to_string()}")
print(f"\n  OOT (all rows by Segment):\n{oot_combined.groupby('Segment').size().to_string()}")
