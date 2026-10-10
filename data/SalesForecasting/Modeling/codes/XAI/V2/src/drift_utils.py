"""
Drift analysis utilities for SalesForecasting XAI V2 pipeline.

Public API
----------
calculate_psi(expected, actual, buckets=10) -> float
compute_descriptive_stats(df, feature_cols, dataset_name, target_col=None)
    -> (continuous_df, categorical_df)
compute_psi_report(benchmark_run, eval_run, benchmark_df, eval_df, feature_cols)
    -> (model_psi_df, feature_psi_df)
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_PSI_STATUS = lambda v: "Stable" if v < 0.10 else ("Monitor" if v < 0.25 else "Significant Drift")

_PERCENTILES = [0.5, 1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99, 99.5]
_PERC_LABELS = [f"p{p}" for p in _PERCENTILES]


# ── PSI ───────────────────────────────────────────────────────────────────────
def calculate_psi(expected: np.ndarray, actual: np.ndarray,
                  buckets: int = 10) -> float:
    """
    Population Stability Index between two distributions.

    Continuous path (cardinality > buckets): 10 equal-frequency bins derived
    from expected; outer edges set to ±∞ so all actual values are captured.
    Discrete path (cardinality ≤ buckets): unique values of expected as bins.
    Zero probabilities replaced with eps=1e-4 before log to avoid ±inf.
    Returns 0.0 for empty or all-NaN inputs.
    """
    eps = 1e-4
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual,   dtype=float)
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    unique_exp = np.unique(expected)

    if len(unique_exp) <= buckets:
        # Discrete / low-cardinality: use unique values as category labels
        exp_counts = np.array([(expected == v).sum() for v in unique_exp], dtype=float)
        act_counts = np.array([(actual   == v).sum() for v in unique_exp], dtype=float)
        exp_pct = exp_counts / len(expected)
        act_pct = act_counts / len(actual)
    else:
        # Continuous: equal-frequency bins from expected
        breakpoints = np.unique(np.percentile(expected, np.linspace(0, 100, buckets + 1)))
        breakpoints[0]  = -np.inf
        breakpoints[-1] =  np.inf
        exp_counts, _ = np.histogram(expected, bins=breakpoints)
        act_counts, _ = np.histogram(actual,   bins=breakpoints)
        exp_pct = exp_counts.astype(float) / len(expected)
        act_pct = act_counts.astype(float) / len(actual)

    exp_pct = np.where(exp_pct == 0, eps, exp_pct)
    act_pct = np.where(act_pct == 0, eps, act_pct)

    return float(np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct)))


# ── Descriptive stats ─────────────────────────────────────────────────────────
def _is_categorical(series: pd.Series, cardinality_threshold: int = 15) -> bool:
    if series.dtype == object or series.dtype == bool:
        return True
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        return series.nunique() < cardinality_threshold
    return False


def _continuous_row(series: pd.Series, name: str, dataset: str) -> dict:
    vals = series.dropna()
    row = {
        "Variable_Name": name,
        "Type"         : "Continuous",
        "Total_Count"  : len(series),
        "Null_Count"   : int(series.isna().sum()),
        "Mean"         : round(float(vals.mean()), 6) if len(vals) else None,
        "Std_Dev"      : round(float(vals.std()),  6) if len(vals) else None,
    }
    for label, pct in zip(_PERC_LABELS, _PERCENTILES):
        row[label] = round(float(np.percentile(vals, pct)), 6) if len(vals) else None
    row["Dataset"] = dataset
    return row


def _categorical_row(series: pd.Series, name: str, dataset: str,
                     top_n: int = 10) -> dict:
    counts = series.value_counts(dropna=True).head(top_n)
    total  = len(series)
    row = {
        "Variable_Name": name,
        "Type"         : "Categorical",
        "Total_Count"  : total,
        "Null_Count"   : int(series.isna().sum()),
    }
    for i, (val, cnt) in enumerate(counts.items(), start=1):
        row[f"Class_{i}_Value"] = val
        row[f"Class_{i}_Count"] = int(cnt)
        row[f"Class_{i}_Pct"]   = round(cnt / total * 100, 2) if total else 0.0
    row["Dataset"] = dataset
    return row


def compute_descriptive_stats(df: pd.DataFrame, feature_cols: list,
                               dataset_name: str,
                               target_col: str = None) -> tuple:
    """
    Descriptive statistics for one dataset.

    Analyzes: feature_cols + Segment (if present) + target_col (if provided).
    Returns (continuous_df, categorical_df).
    """
    cols_to_analyze = list(feature_cols)
    if "Segment" in df.columns and "Segment" not in cols_to_analyze:
        cols_to_analyze.append("Segment")
    if target_col and target_col in df.columns and target_col not in cols_to_analyze:
        cols_to_analyze.append(target_col)

    cont_rows, cat_rows = [], []
    for col in cols_to_analyze:
        if col not in df.columns:
            continue
        series = df[col]
        if _is_categorical(series):
            cat_rows.append(_categorical_row(series, col, dataset_name))
        else:
            cont_rows.append(_continuous_row(series, col, dataset_name))

    continuous_df  = pd.DataFrame(cont_rows) if cont_rows else pd.DataFrame()
    categorical_df = pd.DataFrame(cat_rows)  if cat_rows  else pd.DataFrame()
    return continuous_df, categorical_df


# ── PSI report ────────────────────────────────────────────────────────────────
def compute_psi_report(benchmark_run: dict, eval_run: dict,
                       benchmark_df: pd.DataFrame, eval_df: pd.DataFrame,
                       feature_cols: list) -> tuple:
    """
    PSI between one benchmark dataset and one evaluation dataset.

    model_psi_df columns : Benchmark, Evaluating_Dataset, Model, Model_Stability_PSI, Status
    feature_psi_df columns: Benchmark, Evaluating_Dataset, Assoc_Models, Feature, Feature_PSI, Status

    Covers:
      - Model prediction column PSI → model_psi_df
      - Each feature in feature_cols PSI → feature_psi_df
      - target_col (e.g. UnitSales) PSI → feature_psi_df  (Assoc_Models = "All")
    """
    bm_dataset   = benchmark_run["dataset"]
    eval_dataset = eval_run["dataset"]
    model_name   = benchmark_run["model"]
    pred_col     = benchmark_run["pred_col"]
    target_col   = benchmark_run.get("target_col")

    model_rows   = []
    feature_rows = []

    # ── Model prediction PSI ──────────────────────────────────────────────────
    if pred_col in benchmark_df.columns and pred_col in eval_df.columns:
        psi_val = calculate_psi(benchmark_df[pred_col].values,
                                eval_df[pred_col].values)
        model_rows.append({
            "Benchmark"          : bm_dataset,
            "Evaluating_Dataset" : eval_dataset,
            "Model"              : model_name,
            "Model_Stability_PSI": round(psi_val, 6),
            "Status"             : _PSI_STATUS(psi_val),
        })

    # ── Feature PSI ───────────────────────────────────────────────────────────
    for feat in feature_cols:
        if feat not in benchmark_df.columns or feat not in eval_df.columns:
            continue
        psi_val = calculate_psi(benchmark_df[feat].values,
                                eval_df[feat].values)
        feature_rows.append({
            "Benchmark"          : bm_dataset,
            "Evaluating_Dataset" : eval_dataset,
            "Assoc_Models"       : model_name,
            "Feature"            : feat,
            "Feature_PSI"        : round(psi_val, 6),
            "Status"             : _PSI_STATUS(psi_val),
        })

    # ── Target variable PSI ───────────────────────────────────────────────────
    if (target_col and target_col in benchmark_df.columns
            and target_col in eval_df.columns):
        psi_val = calculate_psi(benchmark_df[target_col].values,
                                eval_df[target_col].values)
        feature_rows.append({
            "Benchmark"          : bm_dataset,
            "Evaluating_Dataset" : eval_dataset,
            "Assoc_Models"       : "All",
            "Feature"            : target_col,
            "Feature_PSI"        : round(psi_val, 6),
            "Status"             : _PSI_STATUS(psi_val),
        })

    model_psi_df   = pd.DataFrame(model_rows)   if model_rows   else pd.DataFrame()
    feature_psi_df = pd.DataFrame(feature_rows) if feature_rows else pd.DataFrame()
    return model_psi_df, feature_psi_df
