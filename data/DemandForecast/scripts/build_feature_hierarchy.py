"""
Phase 1 deliverable — feature_hierarchy.csv for DemandForecast.

Unlike TradeFraud (which has a hand-curated variable_dictionary.csv),
DemandForecast has no such reference file, so the Group / Sub_Group mapping
below is hand-built directly from the feature names and what each one means
(see data/DemandForecast/samples/oos.csv for the raw column definitions).
Same output format as TradeFraud's build_feature_hierarchy.py and
SalesForecasting's reference file: Feature, Sub_Group, Group, Feature_Weight,
Sub_Group_Weight, with equal weights within each Sub_Group / Group.

If a feature is added or renamed later (e.g. new engineered features), add
it to FEATURE_GROUPS below — the script will warn about any model feature
left unmapped rather than silently dropping it.
"""
import os
import pandas as pd

DATA_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/samples/"
OUTPUT_PATH = "/Users/somyabhadada/nodex8/data/DemandForecast/XAI_Outputs/feature_hierarchy.csv"

EXCLUDE_COLS = ["RowNum", "Date", "UPC", "COMP_STR_NO", "QUANTITY", "BRAND", "RET_LIR_ID"]

# (Group, Sub_Group) -> [features]
FEATURE_GROUPS = {
    ("1. Pricing", "Own Price"): [
        "MIN_PRICE", "REG_PRICE", "REG_M_PACK", "REG_M_PRICE",
        "SALE_PRICE", "SALE_M_PACK", "SALE_M_PRICE",
    ],
    ("1. Pricing", "Price Derived"): [
        "price_ratio", "price_discount", "discount_pct", "price_per_pack",
    ],
    ("2. Competitor", "Competitor Price"): [
        "C_REG_PRICE", "C_REG_M_PACK", "C_REG_M_PRICE",
        "C_SALE_PRICE", "C_SALE_M_PACK", "C_SALE_M_PRICE",
    ],
    ("2. Competitor", "Competitor Derived"): [
        "competitor_price_gap", "price_vs_competitor", "competitor_discount_pct",
    ],
    ("3. Promotion", "Own Promotion"): [
        "PROMOTION_FLG", "PROMOTION_FLG_NUM", "PROMOTION_FLG_BIN",
    ],
    ("3. Promotion", "Competitor Promotion"): [
        "C_PROMOTION_FLG", "C_PROMOTION_FLG_NUM", "C_PROMOTION_FLG_BIN",
    ],
    ("3. Promotion", "Promotion Interaction"): [
        "price_x_promo",
    ],
    ("4. Calendar & Seasonality", "Holiday"): [
        "holiday", "holiday_pre", "holiday_post",
    ],
    ("4. Calendar & Seasonality", "Calendar Position"): [
        "Day", "FirstTue", "QDSPWeek", "year", "month", "day", "quarter",
    ],
    ("4. Calendar & Seasonality", "Day of Week"): [
        "dayofweek_num", "day_of_week", "is_weekend",
    ],
    ("5. Demand History", "Lags"): [
        "lag_1", "lag_7", "lag_14", "lag_28",
    ],
    ("5. Demand History", "Rolling Statistics"): [
        "rolling_mean_7", "rolling_std_7", "rolling_mean_28", "rolling_std_28",
    ],
    ("5. Demand History", "Momentum & Volume"): [
        "demand_velocity_7", "demand_volatility", "count",
    ],
}


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "oos.csv"), nrows=1)
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    rows = []
    mapped = set()
    for (group, sub_group), features in FEATURE_GROUPS.items():
        for feat in features:
            rows.append({"Feature": feat, "Sub_Group": sub_group, "Group": group})
            mapped.add(feat)

    missing = set(feat_cols) - mapped
    if missing:
        print(f"WARNING: {len(missing)} model features have no entry in FEATURE_GROUPS: "
              f"{sorted(missing)}")
        for feat in sorted(missing):
            rows.append({"Feature": feat, "Sub_Group": "Unmapped", "Group": "Unmapped"})

    extra = mapped - set(feat_cols)
    if extra:
        print(f"NOTE: {len(extra)} entries in FEATURE_GROUPS are not model features "
              f"(dropped, probably stale): {sorted(extra)}")

    hier = pd.DataFrame(rows)
    hier = hier[hier["Feature"].isin(feat_cols)].reset_index(drop=True)

    sub_group_sizes = hier.groupby("Sub_Group")["Feature"].transform("count")
    hier["Feature_Weight"] = (1.0 / sub_group_sizes).round(4)

    sub_groups_per_group = hier.groupby("Group")["Sub_Group"].transform("nunique")
    hier["Sub_Group_Weight"] = (1.0 / sub_groups_per_group).round(4)

    hier = hier[["Feature", "Sub_Group", "Group", "Feature_Weight", "Sub_Group_Weight"]]
    hier = hier.sort_values(["Group", "Sub_Group", "Feature"]).reset_index(drop=True)
    hier.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {len(hier)} features across {hier['Group'].nunique()} groups "
          f"and {hier['Sub_Group'].nunique()} sub-groups to {OUTPUT_PATH}")
    print(hier["Group"].value_counts())


if __name__ == "__main__":
    main()
