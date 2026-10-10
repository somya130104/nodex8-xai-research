"""
XAI Pipeline — Regression orchestrator for SalesForecasting.

Usage
-----
  cd /Users/angshubh/SalesForecasting/Modeling/codes/XAI/V2
  python3 -u run_pipeline_regression.py                          # all steps, manifest.csv in same dir
  python3 -u run_pipeline_regression.py --steps perf,shap       # selective steps
  python3 -u run_pipeline_regression.py --steps all 2>&1 | tee Outputs/pipeline.log

Steps (run in order):
  perf       — model performance metrics → model_performance.csv
  shap       — SHAP feature importance  → shap_importance.csv
  shap_pt    — SHAP per-observation     → shap_local/*.csv
  lime       — LIME feature importance  → lime_importance.csv
  lime_pt    — LIME per-observation     → lime_local/*.csv
  surrogate  — Surrogate importance     → surrogate_importance.csv
  surr_pt    — Surrogate per-obs        → surrogate_local/*.csv
  drift      — PSI + descriptive stats  → drift_analysis/*.csv
  llm        — LLM narratives (Gemini)  → llm_narratives/*.md
  all        — runs every step above (default)

Notifications
-------------
  Every (step, run_id) pair emits a structured '── NOTIFY ──' block to stdout
  immediately after completion. Grep for that tag to extract all events from a log:
    grep '── NOTIFY ──' Outputs/pipeline.log

src/ files
----------
  All utilities come from V2/src/ (copies of XAI/src/ with sm.load() fix for Linear).
  Regression-only: Linear (FE_LSDV), Tree (XGBoost). LSTM rows set active=N in manifest.
"""

import argparse
import os
import pickle
import sys
import time
import traceback

import numpy as np
import pandas as pd

# ── Project root is the directory containing this script ──────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from src.xai_utils       import calculate_regression_metrics, calculate_classification_metrics
from src.shap_utils      import load_model_artifacts, compute_shap, aggregate_shap
from src.lime_utils      import build_lime_explainer, compute_lime_matrix, aggregate_lime
from src.surrogate_utils import aggregate_surrogate, compute_point_contributions
from src.drift_utils     import (compute_descriptive_stats, compute_psi_report)

VALID_MODEL_TYPES    = {"Regression", "Classification"}
VALID_MODEL_FAMILIES = {"Linear", "Tree", "Neural"}

ALL_STEPS = ["perf", "shap", "shap_pt", "lime", "lime_pt",
             "surrogate", "surr_pt", "drift", "llm"]


# ── Manifesto builder ─────────────────────────────────────────────────────────
def _resolve_path(raw: str, manifest_path: str) -> str:
    """
    Resolve a manifest path against the manifest directory.
    Also checks manifest_dir/notebooks/ as a legacy fallback.
    """
    manifest_dir  = os.path.dirname(os.path.abspath(manifest_path))
    notebooks_dir = os.path.join(manifest_dir, "notebooks")
    for base in [manifest_dir, notebooks_dir]:
        candidate = os.path.normpath(os.path.join(base, raw))
        if os.path.exists(candidate):
            return candidate
    return os.path.normpath(os.path.join(manifest_dir, raw))


def build_manifesto(manifest_path: str, project_dir: str) -> dict:
    manifest = pd.read_csv(manifest_path)
    active   = manifest[manifest["active"].str.upper() == "Y"].copy()
    print(f"  Active runs : {len(active)}  |  Inactive : {len(manifest) - len(active)}", flush=True)

    valid_configs, errors_log = [], []
    for _, row in active.iterrows():
        run_id    = row["run_id"]
        errs      = []
        abs_path  = _resolve_path(row["file_path"],      manifest_path)
        abs_pkl   = _resolve_path(row["model_pkl_path"], manifest_path)
        score_col = str(row.get("score_col", "") or "").strip()

        if not os.path.exists(abs_path):
            errs.append(f"prediction file not found: {abs_path}")
        else:
            try:
                sample = pd.read_csv(abs_path, nrows=5)
                cols   = sample.columns.tolist()
                for col_name, label in [(row["target_col"], "target_col"),
                                        (row["pred_col"],   "pred_col")]:
                    if col_name not in cols:
                        errs.append(f"{label} '{col_name}' missing from file")
                if score_col and score_col not in cols:
                    errs.append(f"score_col '{score_col}' missing from file")
                group_vars = [g.strip() for g in str(row["group_vars"]).split("|") if g.strip()]
                for gv in group_vars:
                    if gv not in cols:
                        errs.append(f"group_var '{gv}' missing from file")
            except Exception as e:
                errs.append(f"could not read file: {e}")

        if not os.path.exists(abs_pkl):
            errs.append(f"model pkl not found: {abs_pkl}")
        if str(row["model_type"]) not in VALID_MODEL_TYPES:
            errs.append(f"model_type '{row['model_type']}' not in {VALID_MODEL_TYPES}")
        if str(row["model_family"]) not in VALID_MODEL_FAMILIES:
            errs.append(f"model_family '{row['model_family']}' not in {VALID_MODEL_FAMILIES}")
        try:
            assert int(row["n_features"]) > 0
        except Exception:
            errs.append(f"n_features must be a positive integer, got '{row['n_features']}'")
        try:
            assert 0 < float(row["tolerance"]) < 1
        except Exception:
            errs.append(f"tolerance must be (0, 1), got '{row['tolerance']}'")

        if errs:
            errors_log.append((run_id, errs))
        else:
            group_vars = [g.strip() for g in str(row["group_vars"]).split("|") if g.strip()]
            valid_configs.append({
                "run_id"          : run_id,
                "name"            : run_id,
                "model"           : row["model_name"],
                "model_domain"    : row["model_domain"],
                "model_type"      : row["model_type"],
                "model_family"    : row["model_family"],
                "dataset"         : row["dataset"],
                "path"            : abs_path,
                "model_pkl_path"  : abs_pkl,
                "target_col"      : row["target_col"],
                "pred_col"        : row["pred_col"],
                "score_col"       : score_col,
                "n_features"      : int(row["n_features"]),
                "tolerance"       : float(row["tolerance"]),
                "group_vars"      : group_vars,
                "benchmark_run_id": ("" if pd.isna(row.get("benchmark", ""))
                                     else str(row.get("benchmark", "")).strip()),
            })

    if errors_log:
        print("\n  [ERROR] Runs failed validation and will be skipped:", flush=True)
        for run_id, errs in errors_log:
            for e in errs:
                print(f"    [{run_id}] {e}", flush=True)

    all_gvars = sorted({gv for cfg in valid_configs for gv in cfg["group_vars"]})
    manifesto = {"dataset_configs": valid_configs, "group_vars": all_gvars}
    print(f"  Valid runs  : {len(valid_configs)}  |  Group vars : {all_gvars}", flush=True)
    return manifesto


# ── Helpers ───────────────────────────────────────────────────────────────────
def _section(title: str):
    print(f"\n{'─'*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'─'*60}", flush=True)


def _stratified_sample(df, frac, min_rows, group_var=None):
    if group_var and group_var in df.columns:
        sample = (
            df.groupby(group_var, group_keys=False)
            .apply(lambda x: x.sample(frac=frac, random_state=42))
            .reset_index(drop=True)
        )
    else:
        sample = df.sample(frac=frac, random_state=42).reset_index(drop=True)
    if len(sample) < min_rows:
        sample = df.sample(min(min_rows, len(df)), random_state=42).reset_index(drop=True)
    return sample


def _save_csv(df: pd.DataFrame, path: str):
    """Write df to path atomically via a .tmp file — prevents partial/corrupt writes on crash."""
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _notify_run(step: str, run_id: str, model_family: str, elapsed: float,
                result_df=None, warnings=None):
    """
    Structured per-run notification emitted after every (step, run_id) completes.

    Grep for '── NOTIFY ──' to extract all events from a log file:
      grep '── NOTIFY ──' Outputs/pipeline.log

    Designed for future integration with alerting and monitoring systems.
    Fields:
      Status   — OK with elapsed time, or ERROR with message
      Top-3    — highest-importance features at Global level (where applicable)
      Value range — min/max of the importance metric
      Rows output — number of rows in the aggregated result
      VALIDATE — non-empty check; flags empty results for investigation
    """
    sep = "─" * 64
    print(f"\n{sep}", flush=True)
    print(f"  ── NOTIFY ── [{step.upper()}] {run_id}  (family={model_family})", flush=True)
    print(f"  Status   : OK  |  Elapsed: {elapsed:.1f}s", flush=True)

    metric_col = {
        "shap"     : "Mean_Abs_SHAP",
        "lime"     : "Mean_Abs_LIME",
        "surrogate": "Feature_Importance",
        "perf"     : None,
        "shap_pt"  : None,
        "lime_pt"  : None,
        "surr_pt"  : None,
    }.get(step)

    if result_df is not None and not result_df.empty and metric_col and metric_col in result_df.columns:
        if "Group_Variable" in result_df.columns:
            global_rows = result_df[result_df["Group_Variable"] == "Global"]
        else:
            global_rows = result_df
        if not global_rows.empty and "Feature" in global_rows.columns:
            top3 = global_rows.groupby("Feature")[metric_col].mean().nlargest(3)
            print(f"  Top-3 features (global |{metric_col}|):", flush=True)
            for feat, val in top3.items():
                print(f"    {feat:<42} {val:.5f}", flush=True)
        val_min = float(result_df[metric_col].min())
        val_max = float(result_df[metric_col].max())
        print(f"  Value range  : [{val_min:.5f}, {val_max:.5f}]", flush=True)
        print(f"  Rows output  : {len(result_df):,}", flush=True)

    elif result_df is not None and step == "perf" and not result_df.empty:
        for col in ["MAPE", "R2", "Accuracy_Pct"]:
            if col in result_df.columns:
                if "Group_Variable" in result_df.columns:
                    gval = result_df[result_df["Group_Variable"] == "Global"][col]
                else:
                    gval = result_df[col]
                if not gval.empty:
                    print(f"  {col:<14}: {float(gval.iloc[0]):.4f}", flush=True)
        print(f"  Rows output  : {len(result_df):,}", flush=True)

    elif result_df is None:
        print(f"  (file saved — point-level output, no aggregation to summarise)", flush=True)

    for w in (warnings or []):
        print(f"  [WARN]   : {w}", flush=True)

    if result_df is None:
        status = "✓ file saved"
    elif not result_df.empty:
        status = "✓ non-empty"
    else:
        status = "⚠ empty — check data or model artifacts"
    print(f"  [VALIDATE]: {status}", flush=True)
    print(sep, flush=True)


# ── Step 1: Model Performance ─────────────────────────────────────────────────
def run_perf(runs: list, output_dir: str) -> int:
    _section("STEP 1 — Model Performance")
    MIN_GROUP_SAMPLES = 5
    all_results = []
    n_failures  = 0

    for run in runs:
        t_run      = time.time()
        run_id     = run["run_id"]
        model_type = run["model_type"]
        pred_col   = run["pred_col"]
        score_col  = run.get("score_col") or pred_col
        target_col = run["target_col"]
        run_gvars  = run["group_vars"]
        print(f"\n  [{run_id}]  type={model_type}", flush=True)

        try:
            df = pd.read_csv(run["path"])
            if pred_col not in df.columns or target_col not in df.columns:
                print(f"    [SKIP] required column missing in prediction file", flush=True)
                n_failures += 1
                continue

            if "Decile" not in df.columns:
                rank_col = score_col if score_col in df.columns else pred_col
                df["Decile"] = pd.qcut(df[rank_col].rank(method="first"), 10,
                                       labels=[f"D{i}" for i in range(1, 11)])

            run_results = []

            def _compute(subset, gvar, gval):
                if len(subset) < MIN_GROUP_SAMPLES:
                    return
                actual = subset[target_col].values
                pred   = subset[pred_col].values
                score  = subset[score_col].values if score_col in subset.columns else pred
                base   = {
                    "Dataset": run["dataset"], "Model": run["model"],
                    "Model_Domain": run["model_domain"], "Model_Type": model_type,
                    "Group_Variable": gvar, "Group_Value": gval, "Samples": len(subset),
                }
                if model_type == "Regression":
                    metrics = calculate_regression_metrics(
                        actual, pred, n_features=run["n_features"], tolerance=run["tolerance"])
                else:
                    pred_bin = (pred >= 0.5).astype(int) if pred.max() <= 1 else pred.astype(int)
                    metrics  = calculate_classification_metrics(actual, score, pred_bin,
                                                                n_features=run["n_features"])
                run_results.append({**base, **metrics})

            _compute(df, "Global", "Global")
            for d in df["Decile"].unique():
                _compute(df[df["Decile"] == d], "Decile", str(d))
            for gvar in run_gvars:
                if gvar not in df.columns:
                    continue
                for gval, gidx in df.groupby(gvar).groups.items():
                    _compute(df.loc[gidx], gvar, str(gval))

            print(f"    Groups: Global + Decile + {run_gvars}", flush=True)
            run_df = pd.DataFrame(run_results)
            all_results.extend(run_results)
            _notify_run("perf", run_id, run["model_family"], time.time() - t_run, run_df)

        except Exception as e:
            print(f"    [ERROR] run_perf failed for {run_id}: {e}", flush=True)
            traceback.print_exc()
            _notify_run("perf", run_id, run["model_family"], time.time() - t_run,
                        pd.DataFrame(), [str(e)])
            n_failures += 1

    if all_results:
        out  = pd.DataFrame(all_results)
        path = os.path.join(output_dir, "model_performance.csv")
        _save_csv(out, path)
        print(f"\n  Saved: model_performance.csv  ({len(out):,} rows)", flush=True)
    else:
        print("\n  [WARN] No performance results produced", flush=True)
    return n_failures


# ── Step 2: SHAP Importance ───────────────────────────────────────────────────
def run_shap(runs: list, output_dir: str) -> int:
    _section("STEP 2 — SHAP Feature Importance")
    MIN_GROUP_SAMPLES = 5
    all_results      = []
    artifacts_cache  = {}
    background_cache = {}
    n_failures       = 0
    dev_lookup = {r["model"]: r for r in runs if r["dataset"] == "DEV"}

    for run in runs:
        t_run        = time.time()
        run_id       = run["run_id"]
        model_name   = run["model"]
        model_family = run["model_family"]
        model_type   = run["model_type"]
        run_gvars    = run["group_vars"]
        print(f"\n  [{run_id}]  family={model_family}  type={model_type}", flush=True)

        df = pd.read_csv(run["path"])

        if model_name not in background_cache:
            dev_run = dev_lookup.get(model_name)
            background_cache[model_name] = pd.read_csv(dev_run["path"]) if dev_run else df
        background_df = background_cache[model_name]

        if model_name not in artifacts_cache:
            model_obj, features, aux = load_model_artifacts(run)
            artifacts_cache[model_name] = (model_obj, features, aux)
            print(f"    Features : {len(features)}", flush=True)
        else:
            model_obj, features, aux = artifacts_cache[model_name]

        t0 = time.time()
        try:
            # compute_shap returns 2-tuple (arr, eff_features) in SalesForecasting src/
            shap_arr, eff_features = compute_shap(
                df, features, model_obj, model_family, aux, background_df,
                model_type=model_type,
            )
        except Exception as e:
            print(f"    [ERROR] compute_shap failed: {e}", flush=True)
            traceback.print_exc()
            _notify_run("shap", run_id, model_family, time.time() - t_run,
                        pd.DataFrame(), [str(e)])
            n_failures += 1
            continue

        elapsed = time.time() - t0
        print(f"    SHAP: {shap_arr.shape}  [{elapsed:.1f}s]", flush=True)

        agg_df = aggregate_shap(shap_arr, eff_features, df, run, run_gvars,
                                min_samples=MIN_GROUP_SAMPLES)
        all_results.append(agg_df)
        for gvar, grp in agg_df.groupby("Group_Variable"):
            print(f"    [{gvar}] {grp['Group_Value'].nunique()} bucket(s)", flush=True)
        _notify_run("shap", run_id, model_family, time.time() - t_run, agg_df)

    if all_results:
        out  = pd.concat(all_results, ignore_index=True)
        path = os.path.join(output_dir, "shap_importance.csv")
        _save_csv(out, path)
        print(f"\n  Saved: shap_importance.csv  ({len(out):,} rows)", flush=True)
    return n_failures


# ── Step 3: SHAP Point ────────────────────────────────────────────────────────
def run_shap_point(runs: list, output_dir: str) -> int:
    _section("STEP 3 — SHAP Per-Observation")
    local_dir = os.path.join(output_dir, "shap_local")
    os.makedirs(local_dir, exist_ok=True)

    ID_BASE = ["Segment", "Activeness_Segment",
               "Client_ID", "Client", "Cust_ID", "WeekYear", "Year", "WeekNum"]
    artifacts_cache  = {}
    background_cache = {}
    n_failures       = 0
    dev_lookup = {r["model"]: r for r in runs if r["dataset"] == "DEV"}

    for run in runs:
        t_run        = time.time()
        run_id       = run["run_id"]
        model_name   = run["model"]
        model_family = run["model_family"]
        model_type   = run["model_type"]
        print(f"\n  [{run_id}]  family={model_family}  type={model_type}", flush=True)

        df = pd.read_csv(run["path"])

        if model_name not in background_cache:
            dev_run = dev_lookup.get(model_name)
            background_cache[model_name] = pd.read_csv(dev_run["path"]) if dev_run else df
        background_df = background_cache[model_name]

        if model_name not in artifacts_cache:
            model_obj, features, aux = load_model_artifacts(run)
            artifacts_cache[model_name] = (model_obj, features, aux)
            print(f"    Features : {len(features)}", flush=True)
        else:
            model_obj, features, aux = artifacts_cache[model_name]

        t0 = time.time()
        try:
            shap_arr, eff_features = compute_shap(
                df, features, model_obj, model_family, aux, background_df,
                model_type=model_type,
            )
        except Exception as e:
            print(f"    [ERROR] compute_shap failed: {e}", flush=True)
            _notify_run("shap_pt", run_id, model_family, time.time() - t_run,
                        None, [str(e)])
            n_failures += 1
            continue

        elapsed = time.time() - t0
        print(f"    SHAP: {shap_arr.shape}  [{elapsed:.1f}s]", flush=True)

        id_cols = [c for c in ID_BASE + [run["target_col"], run["pred_col"]]
                   if c in df.columns]
        seen    = set()
        id_cols = [c for c in id_cols if not (c in seen or seen.add(c))]
        out     = df[id_cols].copy()
        for j, feat in enumerate(eff_features):
            out[f"shap_{feat}"] = shap_arr[:, j].round(6)

        model_key = model_name.lower().replace(" ", "_")
        ds_key    = run["dataset"].lower()
        fname     = f"{model_key}_shap_{ds_key}.csv"
        _save_csv(out, os.path.join(local_dir, fname))
        print(f"    Saved: {fname}  ({len(out):,} rows)", flush=True)
        _notify_run("shap_pt", run_id, model_family, time.time() - t_run)

    return n_failures


# ── Step 4: LIME Importance ───────────────────────────────────────────────────
def run_lime(runs: list, output_dir: str,
             sample_frac: float = 0.001, min_lime_rows: int = 30) -> int:
    _section("STEP 4 — LIME Feature Importance")
    MIN_GROUP_SAMPLES = 5
    all_results      = []
    artifacts_cache  = {}
    explainer_cache  = {}
    background_cache = {}
    n_failures       = 0
    dev_lookup = {r["model"]: r for r in runs if r["dataset"] == "DEV"}

    for run in runs:
        t_run         = time.time()
        run_id        = run["run_id"]
        model_name    = run["model"]
        model_family  = run["model_family"]
        model_type    = run["model_type"]
        run_gvars     = run["group_vars"]
        run_group_var = run_gvars[0] if run_gvars else None
        print(f"\n  [{run_id}]  family={model_family}  type={model_type}", flush=True)

        df = pd.read_csv(run["path"])

        if model_name not in background_cache:
            dev_run = dev_lookup.get(model_name)
            background_cache[model_name] = pd.read_csv(dev_run["path"]) if dev_run else df
        background_df = background_cache[model_name]

        if model_name not in artifacts_cache:
            model_obj, features, aux = load_model_artifacts(run)
            artifacts_cache[model_name] = (model_obj, features, aux)
            print(f"    Features : {len(features)}", flush=True)
            explainer, valid_features = build_lime_explainer(
                background_df, features, model_type=model_type)
            explainer_cache[model_name] = (explainer, valid_features)
            print(f"    Valid LIME features : {len(valid_features)}", flush=True)
        else:
            model_obj, features, aux     = artifacts_cache[model_name]
            explainer, valid_features    = explainer_cache[model_name]

        sampled_df = _stratified_sample(df, sample_frac, min_lime_rows, run_group_var)
        print(f"    Sampled rows : {len(sampled_df):,} of {len(df):,}", flush=True)

        try:
            lime_arr = compute_lime_matrix(
                sampled_df, explainer, model_obj, model_family, valid_features, aux,
                model_type=model_type, group_var=run_group_var,
            )
        except Exception as e:
            print(f"    [ERROR] compute_lime_matrix failed: {e}", flush=True)
            _notify_run("lime", run_id, model_family, time.time() - t_run,
                        pd.DataFrame(), [str(e)])
            n_failures += 1
            continue

        agg_df = aggregate_lime(lime_arr, valid_features, sampled_df, df, run,
                                run_gvars, min_samples=MIN_GROUP_SAMPLES)
        if not agg_df.empty:
            all_results.append(agg_df)
            for gvar, grp in agg_df.groupby("Group_Variable"):
                print(f"    [{gvar}] {grp['Group_Value'].nunique()} bucket(s)", flush=True)
        else:
            print(f"    [WARN] no groups met min_samples threshold", flush=True)
        _notify_run("lime", run_id, model_family, time.time() - t_run, agg_df)

    if all_results:
        out  = pd.concat(all_results, ignore_index=True)
        path = os.path.join(output_dir, "lime_importance.csv")
        _save_csv(out, path)
        print(f"\n  Saved: lime_importance.csv  ({len(out):,} rows)", flush=True)
    return n_failures


# ── Step 5: LIME Point ────────────────────────────────────────────────────────
def run_lime_point(runs: list, output_dir: str,
                   sample_frac: float = 0.10, min_lime_rows: int = 30) -> int:
    _section("STEP 5 — LIME Per-Observation")
    local_dir = os.path.join(output_dir, "lime_local")
    os.makedirs(local_dir, exist_ok=True)

    ID_BASE = ["Segment", "Activeness_Segment",
               "Client_ID", "Client", "Cust_ID", "WeekYear", "Year", "WeekNum"]
    artifacts_cache  = {}
    explainer_cache  = {}
    background_cache = {}
    n_failures       = 0
    dev_lookup = {r["model"]: r for r in runs if r["dataset"] == "DEV"}

    for run in runs:
        t_run         = time.time()
        run_id        = run["run_id"]
        model_name    = run["model"]
        model_family  = run["model_family"]
        model_type    = run["model_type"]
        run_gvars     = run["group_vars"]
        run_group_var = run_gvars[0] if run_gvars else None
        print(f"\n  [{run_id}]  family={model_family}  type={model_type}", flush=True)

        df = pd.read_csv(run["path"])

        if model_name not in background_cache:
            dev_run = dev_lookup.get(model_name)
            background_cache[model_name] = pd.read_csv(dev_run["path"]) if dev_run else df
        background_df = background_cache[model_name]

        if model_name not in artifacts_cache:
            model_obj, features, aux = load_model_artifacts(run)
            artifacts_cache[model_name] = (model_obj, features, aux)
            print(f"    Features : {len(features)}", flush=True)
            explainer, valid_features = build_lime_explainer(
                background_df, features, model_type=model_type)
            explainer_cache[model_name] = (explainer, valid_features)
        else:
            model_obj, features, aux  = artifacts_cache[model_name]
            explainer, valid_features = explainer_cache[model_name]

        sampled_df = _stratified_sample(df, sample_frac, min_lime_rows, run_group_var)
        print(f"    Sampled rows : {len(sampled_df):,} of {len(df):,}", flush=True)

        try:
            lime_arr = compute_lime_matrix(
                sampled_df, explainer, model_obj, model_family, valid_features, aux,
                model_type=model_type, group_var=run_group_var,
            )
        except Exception as e:
            print(f"    [ERROR] compute_lime_matrix failed: {e}", flush=True)
            _notify_run("lime_pt", run_id, model_family, time.time() - t_run,
                        None, [str(e)])
            n_failures += 1
            continue

        _gv_list = [run_group_var] if run_group_var else []
        id_cols  = [c for c in _gv_list + ID_BASE + [run["target_col"], run["pred_col"]]
                    if c in sampled_df.columns]
        seen     = set()
        id_cols  = [c for c in id_cols if not (c in seen or seen.add(c))]
        out      = sampled_df[id_cols].copy().reset_index(drop=True)
        for j, feat in enumerate(valid_features):
            out[f"lime_{feat}"] = lime_arr[:, j].round(6)

        model_key = model_name.lower().replace(" ", "_")
        ds_key    = run["dataset"].lower()
        fname     = f"{model_key}_lime_{ds_key}.csv"
        _save_csv(out, os.path.join(local_dir, fname))
        print(f"    Saved: {fname}  ({len(out):,} rows)", flush=True)
        _notify_run("lime_pt", run_id, model_family, time.time() - t_run)

    return n_failures


# ── Step 6: Surrogate Importance ──────────────────────────────────────────────
def run_surrogate(runs: list, output_dir: str) -> int:
    _section("STEP 6 — Surrogate Feature Importance")
    all_results     = []
    artifacts_cache = {}
    n_failures      = 0

    for run in runs:
        t_run      = time.time()
        run_id     = run["run_id"]
        model_name = run["model"]
        model_type = run["model_type"]
        print(f"\n  [{run_id}]  type={model_type}", flush=True)

        df = pd.read_csv(run["path"])

        if model_name not in artifacts_cache:
            _, features, _ = load_model_artifacts(run)
            artifacts_cache[model_name] = features
            print(f"    Features : {len(features)}", flush=True)
        else:
            features = artifacts_cache[model_name]

        try:
            agg_df = aggregate_surrogate(df, features, run)
        except Exception as e:
            print(f"    [ERROR] aggregate_surrogate failed: {e}", flush=True)
            traceback.print_exc()
            _notify_run("surrogate", run_id, run["model_family"], time.time() - t_run,
                        pd.DataFrame(), [str(e)])
            n_failures += 1
            continue

        if agg_df.empty:
            print(f"    [WARN] No surrogate results — skipping", flush=True)
            _notify_run("surrogate", run_id, run["model_family"], time.time() - t_run,
                        agg_df, ["Empty result — all subgroups may be single-class"])
            continue

        all_results.append(agg_df)
        for gvar, grp in agg_df.groupby("Group_Variable"):
            print(f"    [{gvar}] {grp['Group_Value'].nunique()} bucket(s)", flush=True)
        _notify_run("surrogate", run_id, run["model_family"], time.time() - t_run, agg_df)

    if all_results:
        out  = pd.concat(all_results, ignore_index=True)
        path = os.path.join(output_dir, "surrogate_importance.csv")
        _save_csv(out, path)
        print(f"\n  Saved: surrogate_importance.csv  ({len(out):,} rows)", flush=True)
    return n_failures


# ── Step 7: Surrogate Point ───────────────────────────────────────────────────
def run_surrogate_point(runs: list, output_dir: str,
                        sample_frac: float = 0.10, min_rows: int = 20) -> int:
    _section("STEP 7 — Surrogate Per-Observation")
    local_dir = os.path.join(output_dir, "surrogate_local")
    os.makedirs(local_dir, exist_ok=True)
    artifacts_cache = {}
    n_failures      = 0

    for run in runs:
        t_run      = time.time()
        run_id     = run["run_id"]
        model_name = run["model"]
        model_type = run["model_type"]
        print(f"\n  [{run_id}]  type={model_type}", flush=True)

        df = pd.read_csv(run["path"])

        if model_name not in artifacts_cache:
            _, features, _ = load_model_artifacts(run)
            artifacts_cache[model_name] = features
            print(f"    Features : {len(features)}", flush=True)
        else:
            features = artifacts_cache[model_name]

        try:
            out = compute_point_contributions(
                df, features, run,
                sample_frac=sample_frac,
                min_rows=min_rows,
            )
        except Exception as e:
            print(f"    [ERROR] compute_point_contributions failed: {e}", flush=True)
            traceback.print_exc()
            _notify_run("surr_pt", run_id, run["model_family"], time.time() - t_run,
                        None, [str(e)])
            n_failures += 1
            continue

        model_key = model_name.lower().replace(" ", "_")
        ds_key    = run["dataset"].lower()
        fname     = f"{model_key}_surrogate_{ds_key}.csv"
        _save_csv(out, os.path.join(local_dir, fname))
        surr_cols = [c for c in out.columns if c.startswith("surr_")]
        print(f"    Saved: {fname}  ({len(out):,} rows  {len(surr_cols)} contribution cols)", flush=True)
        _notify_run("surr_pt", run_id, run["model_family"], time.time() - t_run)

    return n_failures


# ── Step 8: Drift Analysis ────────────────────────────────────────────────────
def run_drift(runs: list, output_dir: str) -> int:
    """
    Step 8 — Drift Analysis (PSI + descriptive statistics).

    Benchmark for each eval run is read from run['benchmark_run_id'] (manifest 'benchmark' col).
    Produces 4 CSVs in Outputs/drift_analysis/.
    """
    _section("STEP 8 — Drift Analysis")
    drift_dir = os.path.join(output_dir, "drift_analysis")
    os.makedirs(os.path.join(drift_dir, "model_stability"),       exist_ok=True)
    os.makedirs(os.path.join(drift_dir, "population_stability"),  exist_ok=True)
    os.makedirs(os.path.join(drift_dir, "descriptive_statistics"), exist_ok=True)

    # ── Build feature_cols = union across all active model artifacts ──────────
    feature_set     = set()
    artifacts_cache = {}
    for run in runs:
        mn = run["model"]
        if mn not in artifacts_cache:
            _, feats, _ = load_model_artifacts(run)
            artifacts_cache[mn] = feats
        feature_set.update(artifacts_cache[mn])
    feature_cols = sorted(feature_set)
    target_col   = runs[0]["target_col"] if runs else "UnitSales"
    print(f"\n  Feature union : {len(feature_cols)} cols  |  Target: {target_col}", flush=True)

    # ── Load all CSVs once ────────────────────────────────────────────────────
    run_lookup = {r["run_id"]: r for r in runs}
    df_cache   = {}
    for run in runs:
        df_cache[run["run_id"]] = pd.read_csv(run["path"])

    # ── Descriptive statistics (all active runs) ──────────────────────────────
    all_cont, all_cat = [], []
    for run in runs:
        df = df_cache[run["run_id"]]
        cont_df, cat_df = compute_descriptive_stats(
            df, feature_cols, run["dataset"], target_col=target_col)
        if not cont_df.empty: all_cont.append(cont_df)
        if not cat_df.empty:  all_cat.append(cat_df)
        print(f"  [Desc] {run['run_id']}  continuous={len(cont_df)}  categorical={len(cat_df)}", flush=True)

    if all_cont:
        _save_csv(pd.concat(all_cont, ignore_index=True),
                  os.path.join(drift_dir, "descriptive_statistics", "continuous_features_stats.csv"))
        print(f"\n  Saved: continuous_features_stats.csv  ({sum(len(d) for d in all_cont):,} rows)", flush=True)
    if all_cat:
        _save_csv(pd.concat(all_cat, ignore_index=True),
                  os.path.join(drift_dir, "descriptive_statistics", "categorical_features_stats.csv"))
        print(f"  Saved: categorical_features_stats.csv  ({sum(len(d) for d in all_cat):,} rows)", flush=True)

    # ── PSI: one benchmark-eval pair per eval run ─────────────────────────────
    all_model_psi, all_feat_psi = [], []
    n_failures = 0

    for eval_run in runs:
        bm_run_id = eval_run.get("benchmark_run_id", "")
        if not bm_run_id:
            continue  # this IS the benchmark — skip PSI computation

        bm_run = run_lookup.get(bm_run_id)
        if bm_run is None:
            print(f"\n  [WARN] benchmark '{bm_run_id}' for {eval_run['run_id']} not in active runs — skipping", flush=True)
            continue

        t_run = time.time()
        eval_id = eval_run["run_id"]
        print(f"\n  [{eval_id}]  benchmark={bm_run_id}  family={eval_run['model_family']}", flush=True)

        bm_df   = df_cache[bm_run_id]
        eval_df = df_cache[eval_id]

        try:
            model_psi_df, feat_psi_df = compute_psi_report(
                bm_run, eval_run, bm_df, eval_df, feature_cols)
        except Exception as e:
            print(f"    [ERROR] compute_psi_report failed: {e}", flush=True)
            traceback.print_exc()
            n_failures += 1
            continue

        all_model_psi.append(model_psi_df)
        all_feat_psi.append(feat_psi_df)

        elapsed = time.time() - t_run

        # ── Per-pair notification ─────────────────────────────────────────────
        sep = "─" * 64
        print(f"\n{sep}", flush=True)
        print(f"  ── NOTIFY ── [DRIFT] {eval_id}  benchmark={bm_run_id}  (family={eval_run['model_family']})", flush=True)
        print(f"  Status   : OK  |  Elapsed: {elapsed:.1f}s", flush=True)

        if not model_psi_df.empty:
            for _, mrow in model_psi_df.iterrows():
                print(f"  Model PSI ({mrow['Model']:<12}) : {mrow['Model_Stability_PSI']:.5f}  → {mrow['Status']}", flush=True)

        if not feat_psi_df.empty:
            tgt_row = feat_psi_df[feat_psi_df["Feature"] == target_col]
            if not tgt_row.empty:
                tv = tgt_row.iloc[0]
                print(f"  Target PSI ({target_col:<11}) : {tv['Feature_PSI']:.5f}  → {tv['Status']}", flush=True)
            feat_only = feat_psi_df[feat_psi_df["Assoc_Models"] != "All"]
            if not feat_only.empty:
                counts = feat_only["Status"].value_counts()
                total  = len(feat_only)
                for status in ["Stable", "Monitor", "Significant Drift"]:
                    n = counts.get(status, 0)
                    line = f"  Features {status:<18}: {n:>3} / {total}"
                    if status == "Significant Drift" and n > 0:
                        drifted = feat_only[feat_only["Status"] == "Significant Drift"]["Feature"].tolist()
                        line += f"  → {drifted}"
                    print(line, flush=True)

        validate = "✓ non-empty" if (not model_psi_df.empty or not feat_psi_df.empty) else "⚠ empty — investigate"
        print(f"  [VALIDATE]: {validate}", flush=True)
        print(sep, flush=True)

    # ── Save PSI outputs ──────────────────────────────────────────────────────
    if all_model_psi:
        out = pd.concat(all_model_psi, ignore_index=True)
        _save_csv(out, os.path.join(drift_dir, "model_stability", "model_stability_psi.csv"))
        print(f"\n  Saved: model_stability_psi.csv  ({len(out):,} rows)", flush=True)
    if all_feat_psi:
        out = pd.concat(all_feat_psi, ignore_index=True)
        _save_csv(out, os.path.join(drift_dir, "population_stability", "feature_population_psi.csv"))
        print(f"  Saved: feature_population_psi.csv  ({len(out):,} rows)", flush=True)
    return n_failures


# ── Step 9: LLM Narrative Generation ─────────────────────────────────────────
def run_llm(output_dir: str) -> int:
    """
    Step 9 — LLM Narrative Generation.

    Runs run_llm_narratives.py as a subprocess. Requires GOOGLE_API_KEY env var.
    Skips gracefully if the key is absent (returns 0 — not counted as a failure).
    Returns 1 if the subprocess fails or any output file contains an error string.
    """
    import subprocess
    _section("STEP 9 — LLM Narrative Generation")

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        print("  [SKIP] GOOGLE_API_KEY not set — LLM narratives skipped.", flush=True)
        print("  Set the variable and run:  python3 run_llm_narratives.py --narratives all",
              flush=True)
        return 0

    script = os.path.join(PROJECT_DIR, "run_llm_narratives.py")
    if not os.path.exists(script):
        print(f"  [SKIP] run_llm_narratives.py not found at {script}", flush=True)
        return

    t0 = time.time()
    print(f"  Launching LLM narrative generation (this takes ~10-15 min)...", flush=True)

    result = subprocess.run(
        [sys.executable, "-u", script,
         "--narratives", "all",
         "--output", output_dir],
        env={**os.environ, "GOOGLE_API_KEY": api_key},
    )

    elapsed = time.time() - t0
    if result.returncode == 0:
        narrative_dir = os.path.join(output_dir, "llm_narratives")
        n_total  = 0
        n_errors = 0
        for dirpath, _, files in os.walk(narrative_dir):
            for fname in files:
                if not fname.endswith(".md"):
                    continue
                n_total += 1
                try:
                    with open(os.path.join(dirpath, fname), encoding="utf-8") as fh:
                        if "[GENERATION_FAILED]" in fh.read():
                            n_errors += 1
                except OSError:
                    pass
        status = "PARTIAL" if n_errors else "OK"
        print(f"\n  ── NOTIFY ── [LLM] Narratives complete", flush=True)
        print(f"  Status  : {status}  |  Elapsed: {elapsed:.0f}s", flush=True)
        print(f"  Reports : {n_total} markdown files in Outputs/llm_narratives/", flush=True)
        if n_errors:
            print(f"  [WARN]  : {n_errors}/{n_total} file(s) contain [GENERATION_FAILED] sections "
                  f"— rerun: python3 run_llm_narratives.py --narratives all", flush=True)
        return n_errors
    else:
        print(f"\n  [WARN] LLM step exited with code {result.returncode} "
              f"after {elapsed:.0f}s — check output above.", flush=True)
        return 1


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="XAI Regression Pipeline — SalesForecasting V2")
    parser.add_argument("--manifest", default=os.path.join(PROJECT_DIR, "manifest.csv"),
                        help="Path to manifest.csv (default: same directory as this script)")
    parser.add_argument("--output",   default=os.path.join(PROJECT_DIR, "Outputs"),
                        help="Output directory (default: V2/Outputs/)")
    parser.add_argument("--steps",    default="all",
                        help=f"Comma-separated steps or 'all'. Options: {', '.join(ALL_STEPS)}")
    parser.add_argument("--lime-sample-frac", type=float, default=0.001,
                        help="LIME aggregation sample fraction (default 0.1%)")
    parser.add_argument("--lime-min-rows",    type=int,   default=30,
                        help="LIME aggregation minimum rows per run (default 30)")
    parser.add_argument("--lime-pt-frac",     type=float, default=0.10,
                        help="LIME point-level sample fraction (default 10%%)")
    parser.add_argument("--surr-sample-frac", type=float, default=0.10,
                        help="Surrogate point-level sample fraction (default 10%%)")
    parser.add_argument("--surr-min-rows",    type=int,   default=20,
                        help="Surrogate point-level minimum rows (default 20)")
    args = parser.parse_args()

    steps_req = ALL_STEPS if args.steps.strip().lower() == "all" else \
        [s.strip() for s in args.steps.split(",")]
    unknown = [s for s in steps_req if s not in ALL_STEPS]
    if unknown:
        print(f"Unknown steps: {unknown}. Valid: {ALL_STEPS}", flush=True)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'═'*60}", flush=True)
    print(f"  XAI Regression Pipeline — SalesForecasting V2", flush=True)
    print(f"  Manifest : {args.manifest}", flush=True)
    print(f"  Output   : {args.output}", flush=True)
    print(f"  Steps    : {steps_req}", flush=True)
    print(f"{'═'*60}", flush=True)

    # ── Build manifesto ───────────────────────────────────────────────────────
    pkl_path = os.path.join(args.output, "manifesto_config.pkl")
    print("\n[Manifesto]", flush=True)
    manifesto = build_manifesto(args.manifest, PROJECT_DIR)
    with open(pkl_path, "wb") as f:
        pickle.dump(manifesto, f)
    print(f"  Saved: manifesto_config.pkl", flush=True)

    runs = manifesto["dataset_configs"]
    if not runs:
        print("No valid runs found. Check manifest.csv and fix errors above.", flush=True)
        sys.exit(1)

    t_total  = time.time()
    failures = 0

    if "perf"      in steps_req: failures += run_perf(runs, args.output)
    if "shap"      in steps_req: failures += run_shap(runs, args.output)
    if "shap_pt"   in steps_req: failures += run_shap_point(runs, args.output)
    if "lime"      in steps_req: failures += run_lime(runs, args.output,
                                                       sample_frac=args.lime_sample_frac,
                                                       min_lime_rows=args.lime_min_rows)
    if "lime_pt"   in steps_req: failures += run_lime_point(runs, args.output,
                                                             sample_frac=args.lime_pt_frac,
                                                             min_lime_rows=args.lime_min_rows)
    if "surrogate" in steps_req: failures += run_surrogate(runs, args.output)
    if "surr_pt"   in steps_req: failures += run_surrogate_point(runs, args.output,
                                                                   sample_frac=args.surr_sample_frac,
                                                                   min_rows=args.surr_min_rows)
    if "drift"     in steps_req: failures += run_drift(runs, args.output)
    if "llm"       in steps_req: failures += run_llm(args.output)

    elapsed = time.time() - t_total
    print(f"\n{'═'*60}", flush=True)
    if failures:
        print(f"  Pipeline finished with {failures} failure(s)  [{elapsed:.1f}s]", flush=True)
        print(f"  Review '── NOTIFY ──' blocks above for details.", flush=True)
    else:
        print(f"  Pipeline complete  [{elapsed:.1f}s]", flush=True)
    print(f"  Outputs → {args.output}", flush=True)
    print(f"{'═'*60}\n", flush=True)

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
