from __future__ import annotations

import pandas as pd

from . import audit_writer, fidelity, fidelity_metrics, hasher, model_loader, tolerance_check
from .config import DEFAULT_ATOL
from .feature_aligner import align
from .inferencer import rerun_and_enrich


def load_model_artifacts(run: dict) -> tuple:
    """Thin adapter over model_loader.load(): takes the full run dict and
    returns (model_obj, features, aux) — kept as a standalone,
    engine-module-level function (rather than inlined) so it stays the one
    seam tests/other callers patch to substitute a fake model, the same role
    the old src.shap_utils.load_model_artifacts import used to play."""
    model_obj, features, aux, note = model_loader.load(run["model_pkl_path"])
    if model_obj is None and note:
        print(f"  [model_loader] {note}")
    return model_obj, features, aux


class ValidationEngine:
    def __init__(self, atol: float = DEFAULT_ATOL):
        self.atol = atol

    def run(self, run: dict) -> dict:
        """Never raises — a single run's data problem (a malformed CSV, a
        feature column that's the wrong dtype, ...) should not crash a batch
        of otherwise-independent runs. Anything unexpected becomes an ERROR
        outcome with the underlying reason, rather than a stack trace."""
        try:
            return self._run(run)
        except Exception as e:
            print(f"  [{run['run_id']}] ERROR — {type(e).__name__}: {e}")
            return {"run_id": run["run_id"], "outcome": "ERROR", "error": f"{type(e).__name__}: {e}"}

    def _run(self, run: dict) -> dict:
        atol = run.get("tolerance", self.atol)
        rtol = run.get("rtol", 0.0)
        id_col = run.get("id_col", "Client_ID")

        # "Neural" is an explicit, pre-existing signal that this model needs
        # custom loading code a generic loader can't provide (e.g. a
        # bespoke nn.Module only defined in the training script) — skip it
        # outright rather than let duck-typing decide, since a mock/partial
        # object can look "capable" without actually being usable.
        if run.get("model_family") == "Neural":
            print(f"  [{run['run_id']}] SKIPPED — Neural family requires custom loading the generic loader doesn't provide")
            return {"run_id": run["run_id"], "outcome": "SKIPPED"}

        # Stage 0 — hash raw input files
        data_hash  = audit_writer._sha256_file(run["path"])
        model_hash = audit_writer._sha256_file(run["model_pkl_path"])

        # Stage 1 — load model + align features (generic: bare estimator or
        # {"model":..., "features":...} bundle; no project-specific classes)
        model_obj, features, _aux = load_model_artifacts(run)
        problem_type = model_loader.capability(model_obj)

        if model_obj is None or problem_type is None:
            print(f"  [{run['run_id']}] SKIPPED — no usable predict/predict_proba on the loaded model")
            return {"run_id": run["run_id"], "outcome": "SKIPPED"}

        df = pd.read_csv(run["path"])
        X  = align(df, features)

        # Stage 1b — semantic hashes (inputs and model pinned before inference)
        H_data      = hasher.h_data(df, features, id_col=id_col)
        H_model_ver = hasher.h_model_version(model_obj, features)

        # Stage 2 — re-inference + enrich + save validated CSV
        score_col = run["score_col"]
        pred_col  = run.get("pred_col")
        enriched_df, validated_csv_path = rerun_and_enrich(
            df, model_obj, X, run, atol, rtol=rtol,
            h_model_version_str=H_model_ver, id_col=id_col,
        )

        # Stage 2b — output + decision hashes (scores now available)
        row_ids = df[id_col].values if id_col in df.columns else None
        H_out   = hasher.h_output(enriched_df[f"computed_{score_col}"].values, row_ids=row_ids)
        H_dec   = hasher.h_decision(H_data, H_model_ver, H_out)

        added_columns = [f"computed_{score_col}", f"abs_delta_{score_col}", "tolerance_pass"]
        if problem_type == "classification" and pred_col:
            added_columns.insert(1, f"computed_{pred_col}")

        # Stage 3 — per-row tolerance stats
        stats = tolerance_check.check(enriched_df, score_col, atol, rtol=rtol)

        # Stage 4 — fidelity score + outcome (legacy single-metric, atol-only)
        fidelity_pct, outcome = fidelity.score(stats["n_passing"], stats["n_rows"])

        # Stage 4b — high-level multi-metric composite fidelity score
        decision_threshold, threshold_source = fidelity_metrics.resolve_decision_threshold(
            run["model_pkl_path"], problem_type=problem_type
        )
        fmetrics = fidelity_metrics.compute(
            enriched_df, score_col, atol, rtol=rtol,
            decision_threshold=decision_threshold, problem_type=problem_type,
        )

        # Stage 5 — self-sealing audit record
        traceability = {
            "H_data":          H_data,
            "H_model_version": H_model_ver,
            "H_output":        H_out,
            "H_decision":      H_dec,
        }
        audit_writer.write(
            run=run,
            data_hash=data_hash,
            model_hash=model_hash,
            validated_csv_path=validated_csv_path,
            n_features_aligned=len(features),
            added_columns=added_columns,
            stats=stats,
            fidelity_pct=fidelity_pct,
            outcome=outcome,
            atol=atol,
            traceability_hashes=traceability,
            fidelity_metrics=fmetrics,
        )

        result = {
            "run_id":          run["run_id"],
            "model":           run.get("model"),
            "model_family":    run.get("model_family"),
            "problem_type":    problem_type,
            "dataset":         run.get("dataset"),
            "fidelity_pct":    fidelity_pct,
            "outcome":         outcome,
            "n_rows":          stats["n_rows"],
            "n_failing":       stats["n_failing"],
            "max_abs_delta":   stats["max_abs_delta"],
            "mean_abs_delta":  stats["mean_abs_delta"],
            "p95_abs_delta":   stats["p95_abs_delta"],
            "p99_abs_delta":   stats["p99_abs_delta"],
            "validated_csv":   validated_csv_path,
            "H_data":          H_data,
            "H_model_version": H_model_ver,
            "H_output":        H_out,
            "H_decision":      H_dec,
            "atol": atol, "rtol": rtol,
            "decision_threshold": decision_threshold,
            "decision_threshold_source": threshold_source,
            "composite_fidelity_score": fmetrics["composite_fidelity_score"],
            "composite_outcome":        fmetrics["outcome"],
            "tolerance_pass_rate":       fmetrics["tolerance_pass_rate"],
            "decision_agreement_rate":   fmetrics["decision_agreement_rate"],
            "rank_fidelity":             fmetrics["rank_fidelity"],
            "top_decile_overlap":        fmetrics["top_decile_overlap"],
            "distributional_similarity": fmetrics["distributional_similarity"],
            "calibration_score":         fmetrics["calibration_score"],
            "completeness":              fmetrics["completeness"],
            "mean_signed_bias":          fmetrics["mean_signed_bias"],
        }

        status_icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(outcome, "?")
        composite_icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(fmetrics["outcome"], "?")
        print(
            f"  [{run['run_id']}] {status_icon} {outcome}  "
            f"fidelity={fidelity_pct:.4f}%  "
            f"n_fail={stats['n_failing']}  "
            f"max_delta={stats['max_abs_delta']}  |  "
            f"composite {composite_icon} {fmetrics['composite_fidelity_score']:.2f}"
        )
        return result
