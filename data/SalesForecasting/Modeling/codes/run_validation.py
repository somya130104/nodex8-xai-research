#!/usr/bin/env python
"""
Data Validation Engine — CLI entry point.

Usage:
  python run_validation.py --manifest manifest.csv
  python run_validation.py --manifest manifest.csv --run-id XGBoost_DEV
  python run_validation.py --manifest manifest.csv --atol 0.001
  python run_validation.py --manifest manifest.csv --fail-fast

Exit codes:
  0  all runs PASS
  1  at least one FAIL
  2  at least one WARN, no FAIL
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validation.engine import ValidationEngine

_VALID_MODEL_TYPES    = {"Regression", "Classification"}
_VALID_MODEL_FAMILIES = {"Linear", "Tree", "Neural"}


def _resolve_path(raw: str, manifest_path: str) -> str:
    manifest_dir  = os.path.dirname(os.path.abspath(manifest_path))
    notebooks_dir = os.path.join(manifest_dir, "notebooks")
    for base in [manifest_dir, notebooks_dir]:
        candidate = os.path.normpath(os.path.join(base, raw))
        if os.path.exists(candidate):
            return candidate
    return os.path.normpath(os.path.join(manifest_dir, raw))


def _get_or(row: pd.Series, col: str, default):
    """row[col] if the column exists and is non-NaN, else default. Lets a
    manifest optionally carry columns another manifest doesn't (e.g. a
    project whose row identifier isn't 'Client_ID', or one that wants a
    relative tolerance) without breaking manifests that don't have them."""
    if col not in row.index:
        return default
    val = row[col]
    return default if pd.isna(val) else val


def _build_manifesto(manifest_path: str) -> dict:
    manifest = pd.read_csv(manifest_path)
    active = manifest[manifest["active"].str.upper() == "Y"].copy()
    print(f"  Active runs : {len(active)}  |  Inactive : {len(manifest) - len(active)}")

    valid_configs, errors_log = [], []
    for _, row in active.iterrows():
        run_id = row["run_id"]
        errs   = []

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
            except Exception as e:
                errs.append(f"could not read file: {e}")

        if not os.path.exists(abs_pkl):
            errs.append(f"model pkl not found: {abs_pkl}")
        if str(row["model_type"]) not in _VALID_MODEL_TYPES:
            errs.append(f"model_type '{row['model_type']}' not in {_VALID_MODEL_TYPES}")
        if str(row["model_family"]) not in _VALID_MODEL_FAMILIES:
            errs.append(f"model_family '{row['model_family']}' not in {_VALID_MODEL_FAMILIES}")

        if errs:
            errors_log.append((run_id, errs))
        else:
            valid_configs.append({
                "run_id":         run_id,
                "name":           run_id,
                "model":          row["model_name"],
                "model_domain":   row["model_domain"],
                "model_type":     row["model_type"],
                "model_family":   row["model_family"],
                "dataset":        row["dataset"],
                "path":           abs_path,
                "model_pkl_path": abs_pkl,
                "target_col":     row["target_col"],
                "pred_col":       row["pred_col"],
                "score_col":      score_col,
                "n_features":     int(row["n_features"]),
                "tolerance":      float(row["tolerance"]),
                "rtol":           float(_get_or(row, "rtol", 0.0)),
                "id_col":         str(_get_or(row, "id_col", "Client_ID")).strip(),
                "group_vars":     [g.strip() for g in str(row["group_vars"]).split("|") if g.strip()],
            })

    if errors_log:
        print("\n  [ERROR] The following runs failed manifest validation:")
        for run_id, errs in errors_log:
            for e in errs:
                print(f"    [{run_id}] {e}")

    print(f"  Valid runs  : {len(valid_configs)}")
    return {"dataset_configs": valid_configs}


def _section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Data Validation Engine")
    parser.add_argument("--manifest",  required=True, help="Path to manifest.csv")
    parser.add_argument("--run-id",    default=None,  help="Validate a single run_id only")
    parser.add_argument("--atol",      type=float, default=None,
                        help="Override absolute tolerance (default: per-manifest tolerance col)")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Abort after first FAIL result")
    args = parser.parse_args()

    _section("Building manifesto")
    manifesto = _build_manifesto(args.manifest)
    runs = manifesto["dataset_configs"]

    if args.run_id:
        runs = [r for r in runs if r["run_id"] == args.run_id]
        if not runs:
            print(f"[ERROR] run_id '{args.run_id}' not found in active manifest runs.")
            sys.exit(1)

    if args.atol is not None:
        for r in runs:
            r["tolerance"] = args.atol

    engine = ValidationEngine()

    _section(f"Validating {len(runs)} run(s)")
    results, outcomes = [], []
    for run in runs:
        result = engine.run(run)
        results.append(result)
        outcomes.append(result.get("outcome", "SKIPPED"))
        if args.fail_fast and result.get("outcome") in ("FAIL", "ERROR"):
            print(f"\n  [--fail-fast] Aborting after first {result.get('outcome')}.")
            break

    overall = (
        "FAIL"    if ("FAIL" in outcomes or "ERROR" in outcomes) else
        "WARN"    if "WARN"    in outcomes else
        "SKIPPED" if all(o == "SKIPPED" for o in outcomes) else
        "PASS"
    )

    _print_summary(results, outcomes, overall)
    _write_report(results, outcomes, overall, args)
    _write_fidelity_csv(results)
    _record_lineage(runs, results, args, overall)

    if overall == "FAIL":
        sys.exit(1)
    if overall == "WARN":
        sys.exit(2)
    sys.exit(0)


def _print_summary(results: list, outcomes: list, overall: str):
    _section("Validation Summary Report")

    OUTCOME_ICON = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIPPED": "[SKIP]", "ERROR": "[ERR!]"}

    header = (
        f"  {'Run ID':<25} {'Outcome':<8} {'Fidelity %':>12} "
        f"{'Rows':>8} {'Failing':>8} {'Max Delta':>12} {'Validated CSV / Error'}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        icon     = OUTCOME_ICON.get(r.get("outcome", "SKIPPED"), "[?]")
        fidelity = f"{r['fidelity_pct']:.4f}" if "fidelity_pct" in r else "N/A"
        n_rows   = f"{r.get('n_rows', 'N/A'):,}" if isinstance(r.get("n_rows"), int) else "N/A"
        n_fail   = f"{r.get('n_failing', 'N/A'):,}" if isinstance(r.get("n_failing"), int) else "N/A"
        max_d    = f"{r.get('max_abs_delta', 'N/A'):.6f}" if isinstance(r.get("max_abs_delta"), float) else "N/A"
        csv_name = r.get("error") or (Path(r.get("validated_csv", "")).name if r.get("validated_csv") else "—")
        print(
            f"  {r['run_id']:<25} {icon:<8} {fidelity:>12} "
            f"{n_rows:>8} {n_fail:>8} {max_d:>12}  {csv_name}"
        )

    print("  " + "-" * (len(header) - 2))
    n_pass    = outcomes.count("PASS")
    n_warn    = outcomes.count("WARN")
    n_fail    = outcomes.count("FAIL")
    n_skipped = outcomes.count("SKIPPED")
    n_error   = outcomes.count("ERROR")
    print(
        f"\n  Overall: {overall}  |  "
        f"PASS={n_pass}  WARN={n_warn}  FAIL={n_fail}  SKIP={n_skipped}  ERROR={n_error}"
    )


def _write_report(results: list, outcomes: list, overall: str, args):
    from validation.config import REPORTS_DIR
    REPORTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    report_path = REPORTS_DIR / f"validation_report_{ts}.json"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "manifest":     os.path.abspath(args.manifest),
        "atol_override": args.atol,
        "run_id_filter": args.run_id,
        "overall_outcome": overall,
        "counts": {
            "total":   len(results),
            "pass":    outcomes.count("PASS"),
            "warn":    outcomes.count("WARN"),
            "fail":    outcomes.count("FAIL"),
            "skipped": outcomes.count("SKIPPED"),
            "error":   outcomes.count("ERROR"),
        },
        "runs": results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Report saved → {report_path}")


def _write_fidelity_csv(results: list) -> None:
    from validation.config import REPORTS_DIR
    from validation.fidelity_report import write_csv

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = REPORTS_DIR / f"fidelity_scores_{ts}.csv"
    write_csv(results, str(out_path))
    print(f"  Fidelity scores CSV → {out_path}")


# ── Lineage integration ───────────────────────────────────────────────────────

def _record_lineage(manifesto_runs: list, results: list, args, overall: str) -> None:
    """Append a validation LineageRun to Outputs/lineage.ndjson."""
    try:
        import uuid
        from datetime import datetime, timezone
        from validation.config import REPORTS_DIR
        from lineage.tracker import LineageTracker
        from lineage.models import LineageRun

        _PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
        _OUTPUT_DIR  = os.path.join(_PROJECT_DIR, "Outputs")
        lineage_path = os.path.join(_OUTPUT_DIR, "lineage.ndjson")
        tracker = LineageTracker(lineage_path=lineage_path, project_root=_PROJECT_DIR)

        # Inputs: manifest + data files + model PKLs
        seen: set[str] = set()
        input_refs = []
        manifest_abs = os.path.abspath(args.manifest)
        if os.path.exists(manifest_abs):
            try:
                input_refs.append(tracker.make_ref(manifest_abs, "manifest", {}))
            except Exception:
                pass
        for run in manifesto_runs:
            for fpath, ntype, meta in [
                (run["path"],           "raw_data",       {"dataset": run["dataset"], "model": run["model"]}),
                (run["model_pkl_path"], "model_artifact", {"model": run["model"]}),
            ]:
                if fpath not in seen and os.path.exists(fpath):
                    try:
                        input_refs.append(tracker.make_ref(fpath, ntype, meta))
                        seen.add(fpath)
                    except Exception:
                        pass

        # Outputs: validated CSVs + the JSON report written by _write_report()
        output_refs = []
        for res in results:
            for key in ("validated_csv",):
                vpath = res.get(key)
                if vpath and os.path.exists(str(vpath)):
                    try:
                        output_refs.append(
                            tracker.make_ref(str(vpath), "validation_report",
                                             {"run_id": res.get("run_id"), "outcome": res.get("outcome")}))
                    except Exception:
                        pass
        # Also capture the most-recent JSON report in REPORTS_DIR
        if REPORTS_DIR.exists():
            json_reports = sorted(REPORTS_DIR.glob("validation_report_*.json"),
                                  key=lambda p: p.stat().st_mtime, reverse=True)
            if json_reports:
                try:
                    output_refs.append(
                        tracker.make_ref(str(json_reports[0]), "validation_report",
                                         {"overall": overall}))
                except Exception:
                    pass

        if not output_refs:
            return

        now = datetime.now(timezone.utc).isoformat()
        lrun = LineageRun(
            run_id       = str(uuid.uuid4()),
            run_type     = "validation",
            step         = "validate",
            started_at   = now,
            completed_at = now,
            status       = "success" if overall in ("PASS", "WARN") else "failed",
            inputs       = input_refs,
            outputs      = output_refs,
            params       = {
                "manifest":     os.path.abspath(args.manifest),
                "overall":      overall,
                "atol_override": args.atol,
                "run_id_filter": args.run_id,
            },
        )
        tracker.record(lrun)
    except Exception:
        pass   # lineage recording must never break validation


if __name__ == "__main__":
    main()
