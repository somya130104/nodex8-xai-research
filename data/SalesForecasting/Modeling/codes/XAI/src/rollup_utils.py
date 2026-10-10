"""
rollup_utils.py — Roll up feature-level XAI importance to Sub_Group and Group levels.

Hierarchy: Feature → Sub_Group → Group  (three levels, equal weights per level)

Public API
----------
load_hierarchy(path)
    Load and validate feature_hierarchy.csv.

rollup(df, hierarchy, value_col, extra_group_cols=[])
    Aggregate feature-level importance to Sub_Group and Group.
    Returns a combined DataFrame with Level ∈ {Feature, Sub_Group, Group}.

validate_weights(rolled_df, value_col, extra_group_cols=[])
    Assert renormalized weights sum to 1.0 at both levels for every slice.
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_META_COLS = [
    "Dataset", "Model", "Model_Domain", "Model_Type",
    "Group_Variable", "Group_Value", "Samples",
]


# ── Hierarchy loader ───────────────────────────────────────────────────────────

def load_hierarchy(path: str) -> pd.DataFrame:
    """Load feature_hierarchy.csv and assert weight integrity."""
    h = pd.read_csv(path)
    required = {"Feature", "Sub_Group", "Group", "Feature_Weight", "Sub_Group_Weight"}
    missing = required - set(h.columns)
    if missing:
        raise ValueError(f"feature_hierarchy.csv missing columns: {missing}")

    fw  = h.groupby("Sub_Group")["Feature_Weight"].sum()
    sgw = (h.drop_duplicates(["Sub_Group", "Group"])
            .groupby("Group")["Sub_Group_Weight"].sum())

    bad_fw  = fw[abs(fw  - 1.0) > 0.01]
    bad_sgw = sgw[abs(sgw - 1.0) > 0.01]
    if not bad_fw.empty:
        raise ValueError(f"Feature_Weight does not sum to 1.0 for Sub_Groups: {bad_fw.index.tolist()}")
    if not bad_sgw.empty:
        raise ValueError(f"Sub_Group_Weight does not sum to 1.0 for Groups: {bad_sgw.index.tolist()}")

    return h


# ── Roll-up ────────────────────────────────────────────────────────────────────

def rollup(
    df: pd.DataFrame,
    hierarchy: pd.DataFrame,
    value_col: str,
    extra_group_cols: list = [],
) -> pd.DataFrame:
    """
    Roll up feature-level XAI importance to Sub_Group and Group levels.

    Parameters
    ----------
    df               : Feature-level importance CSV loaded as DataFrame
                       (shap_importance.csv / lime_importance.csv / surrogate_importance.csv)
    hierarchy        : Output of load_hierarchy()
    value_col        : Importance column to aggregate
                       ('Mean_Abs_SHAP', 'Mean_Abs_LIME', 'Feature_Importance')
    extra_group_cols : Additional slice columns to preserve per-group
                       (e.g. ['Surrogate_Model'] for surrogate outputs)

    Returns
    -------
    Combined long-format DataFrame with columns:
        <meta_cols>, [extra_group_cols], Level, Group, Sub_Group, Feature, <value_col>

    Level values
    ------------
    'Feature'   — original rows enriched with Group and Sub_Group
    'Sub_Group' — equal-weight mean of present features within each Sub_Group
    'Group'     — equal-weight mean of present Sub_Groups within each Group

    Weight renormalization
    ----------------------
    When a model lacks certain features (e.g. LSTM omits lag_2, roll_mean_*),
    weights renormalize automatically because the aggregation uses .mean() over
    only the rows present in the slice. The weights always sum to 1.0.
    """
    feat_map = hierarchy.set_index("Feature")[["Sub_Group", "Group"]]

    df = df.copy()
    df["Sub_Group"] = df["Feature"].map(feat_map["Sub_Group"])
    df["Group"]     = df["Feature"].map(feat_map["Group"])
    df["Level"]     = "Feature"

    n_unmapped = df["Sub_Group"].isna().sum()
    if n_unmapped:
        unmapped = df.loc[df["Sub_Group"].isna(), "Feature"].unique().tolist()
        warnings.warn(
            f"Rollup: {n_unmapped} rows with features not in hierarchy — dropped: {unmapped}"
        )
        df = df.dropna(subset=["Sub_Group"])

    slice_cols = [c for c in _META_COLS if c in df.columns] + extra_group_cols
    out_cols   = slice_cols + ["Level", "Group", "Sub_Group", "Feature", value_col]

    # ── Sub_Group level ────────────────────────────────────────────────────────
    sg_df = (
        df.groupby(slice_cols + ["Group", "Sub_Group"], sort=False)[value_col]
        .mean()
        .round(6)
        .reset_index()
    )
    sg_df["Level"]   = "Sub_Group"
    sg_df["Feature"] = ""

    # ── Group level ────────────────────────────────────────────────────────────
    g_df = (
        sg_df.groupby(slice_cols + ["Group"], sort=False)[value_col]
        .mean()
        .round(6)
        .reset_index()
    )
    g_df["Level"]     = "Group"
    g_df["Sub_Group"] = ""
    g_df["Feature"]   = ""

    # ── Combine all three levels ───────────────────────────────────────────────
    combined = pd.concat(
        [df.reindex(columns=out_cols),
         sg_df.reindex(columns=out_cols),
         g_df.reindex(columns=out_cols)],
        ignore_index=True,
    )

    level_order = {"Group": 0, "Sub_Group": 1, "Feature": 2}
    combined["_ord"] = combined["Level"].map(level_order)
    combined = (
        combined
        .sort_values(
            slice_cols + ["Group", "_ord", "Sub_Group", value_col],
            ascending=[True] * len(slice_cols) + [True, True, True, False],
        )
        .drop(columns="_ord")
        .reset_index(drop=True)
    )

    return combined


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_weights(
    rolled_df: pd.DataFrame,
    value_col: str,
    extra_group_cols: list = [],
) -> None:
    """
    Assert renormalized weights sum to 1.0 at both hierarchy levels for every slice.

    For equal-weight averaging, n features each with weight 1/n → sum = 1.0.
    This is guaranteed by .mean(), so this check validates bookkeeping is correct.

    Also warns if any Group-level value exceeds the maximum Feature-level value
    in that group (would indicate an aggregation error, not a weight violation).
    """
    slice_cols = [c for c in _META_COLS if c in rolled_df.columns] + extra_group_cols

    feat = rolled_df[rolled_df["Level"] == "Feature"].copy()
    sg   = rolled_df[rolled_df["Level"] == "Sub_Group"].copy()
    g    = rolled_df[rolled_df["Level"] == "Group"].copy()

    errors = []

    # ── Check 1: Sub_Group weights sum to 1.0 ─────────────────────────────────
    # Each Sub_Group value = mean of k features → implied weight per feature = 1/k → sum = 1.0
    # Verify by checking that Sub_Group value equals the mean of its Feature-level values
    sg_check = (
        feat.groupby(slice_cols + ["Group", "Sub_Group"])[value_col]
        .mean()
        .round(6)
        .reset_index()
        .rename(columns={value_col: "_expected"})
    )
    merged_sg = sg.merge(sg_check, on=slice_cols + ["Group", "Sub_Group"], how="left")
    bad_sg = merged_sg[abs(merged_sg[value_col] - merged_sg["_expected"]) > 1e-4]
    if not bad_sg.empty:
        errors.append(f"Sub_Group weight check failed for {len(bad_sg)} rows")

    # ── Check 2: Group weights sum to 1.0 ─────────────────────────────────────
    g_check = (
        sg.groupby(slice_cols + ["Group"])[value_col]
        .mean()
        .round(6)
        .reset_index()
        .rename(columns={value_col: "_expected"})
    )
    merged_g = g.merge(g_check, on=slice_cols + ["Group"], how="left")
    bad_g = merged_g[abs(merged_g[value_col] - merged_g["_expected"]) > 1e-4]
    if not bad_g.empty:
        errors.append(f"Group weight check failed for {len(bad_g)} rows")

    if errors:
        raise AssertionError("Weight validation failed:\n" + "\n".join(errors))

    # ── Check 3: No Group value exceeds max Feature value in its group ─────────
    feat_max = feat.groupby(slice_cols + ["Group"])[value_col].max().rename("feat_max")
    g_idx    = g.set_index(slice_cols + ["Group"])[value_col].rename("grp_val")
    ratio_df = g_idx.to_frame().join(feat_max, how="left")
    ratio_df["ratio"] = ratio_df["grp_val"] / ratio_df["feat_max"].replace(0, np.nan)
    bad_ratio = ratio_df[ratio_df["ratio"] > 1.05].dropna()
    if not bad_ratio.empty:
        warnings.warn(
            f"Group value exceeds max Feature value by >5% in {len(bad_ratio)} cases "
            f"(may indicate skew from small Sub_Groups with high single-feature values)"
        )

    print(f"  Validation passed — Sub_Group weights OK, Group weights OK"
          + (f", {len(bad_ratio)} ratio warnings" if not bad_ratio.empty else ""))
