"""
Phase 1 deliverable — feature_hierarchy.csv for TradeFraud.

Maps every model feature to a Sub_Group -> Group, in the same format as
SalesForecasting's reference file (data/SalesForecasting/Modeling/codes/XAI/
feature_hierarchy.csv): Feature, Sub_Group, Group, Feature_Weight,
Sub_Group_Weight. This is what lets later work report "Promotional Activity:
31%" instead of 271 individual feature rows (the "rollup" step).

TradeFraud already has this information hand-curated in
Data/variable_dictionary.csv (Group / Sub_Group columns), so this script
just filters that down to the 271 columns actually used as model features
and computes equal weights within each Sub_Group / Group, matching
SalesForecasting's convention (see that file: 3 equally-weighted features
in one Sub_Group each get Feature_Weight = 1/3).
"""
import os
import pandas as pd

DATA_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/Modeling_Data_Enriched/"
VAR_DICT_PATH = "/Users/somyabhadada/nodex8/data/TradeFraud/Data/variable_dictionary.csv"
OUTPUT_PATH = "/Users/somyabhadada/nodex8/data/TradeFraud/XAI_Outputs/feature_hierarchy.csv"

EXCLUDE_COLS = ["Client", "Client_ID", "Date", "Dispute", "Segment", "Activeness_Segment"]


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), nrows=1)
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    vd = pd.read_csv(VAR_DICT_PATH)
    vd = vd[vd["Variable_Name"].isin(feat_cols)][["Variable_Name", "Group", "Sub_Group"]].copy()
    vd = vd.rename(columns={"Variable_Name": "Feature"})

    missing = set(feat_cols) - set(vd["Feature"])
    if missing:
        print(f"WARNING: {len(missing)} model features have no entry in variable_dictionary.csv: "
              f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
        # Fall back to an explicit "Unmapped" bucket rather than silently dropping them
        fallback = pd.DataFrame({"Feature": sorted(missing), "Group": "Unmapped", "Sub_Group": "Unmapped"})
        vd = pd.concat([vd, fallback], ignore_index=True)

    sub_group_sizes = vd.groupby("Sub_Group")["Feature"].transform("count")
    vd["Feature_Weight"] = (1.0 / sub_group_sizes).round(4)

    sub_groups_per_group = vd.groupby("Group")["Sub_Group"].transform("nunique")
    vd["Sub_Group_Weight"] = (1.0 / sub_groups_per_group).round(4)

    vd = vd[["Feature", "Sub_Group", "Group", "Feature_Weight", "Sub_Group_Weight"]]
    vd = vd.sort_values(["Group", "Sub_Group", "Feature"]).reset_index(drop=True)
    vd.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(vd)} features across {vd['Group'].nunique()} groups "
          f"and {vd['Sub_Group'].nunique()} sub-groups to {OUTPUT_PATH}")
    print(vd["Group"].value_counts())


if __name__ == "__main__":
    main()
