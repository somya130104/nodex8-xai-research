"""Integration tests for validation.engine — ValidationEngine.run()."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from validation.tests.helpers import (
    ATOL, FEATURE_NAMES, N_ROWS, PRED_COL, SCORE_COL,
    make_df, make_model, make_scores,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_run(tmp_path: Path, scores: np.ndarray, model_family: str = "Tree") -> dict:
    df       = make_df(scores)
    csv_path = tmp_path / "preds.csv"
    pkl_path = tmp_path / "model.pkl"
    df.to_csv(csv_path, index=False)
    pkl_path.write_bytes(b"fake-pkl")
    return {
        "run_id":         "MockModel_DEV",
        "name":           "MockModel_DEV",
        "model":          "MockModel",
        "model_domain":   "Fraud",
        "model_type":     "Classification",
        "model_family":   model_family,
        "dataset":        "DEV",
        "path":           str(csv_path),
        "model_pkl_path": str(pkl_path),
        "target_col":     PRED_COL,
        "pred_col":       PRED_COL,
        "score_col":      SCORE_COL,
        "n_features":     len(FEATURE_NAMES),
        "tolerance":      ATOL,
        "group_vars":     [],
    }


def _run_engine(tmp_path: Path, run: dict, model: MagicMock) -> dict:
    """Run the engine with all I/O patched to tmp_path."""
    audit_log = tmp_path / "audit.ndjson"
    with patch("validation.engine.load_model_artifacts",
               return_value=(model, FEATURE_NAMES, {})):
        with patch("validation.inferencer.REPORTS_DIR", tmp_path):
            with patch("validation.audit_writer.REPORTS_DIR", tmp_path):
                with patch("validation.audit_writer.AUDIT_LOG", audit_log):
                    from validation.engine import ValidationEngine
                    return ValidationEngine().run(run)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEngineReturnShape:
    def test_returns_dict(self, tmp_path):
        scores = make_scores()
        run    = _build_run(tmp_path, scores)
        result = _run_engine(tmp_path, run, make_model(scores))
        assert isinstance(result, dict)

    def test_result_has_required_keys(self, tmp_path):
        scores = make_scores()
        run    = _build_run(tmp_path, scores)
        result = _run_engine(tmp_path, run, make_model(scores))
        for key in ("run_id", "outcome", "fidelity_pct",
                    "n_rows", "n_failing", "max_abs_delta", "validated_csv",
                    "H_data", "H_model_version", "H_output", "H_decision"):
            assert key in result, f"missing key: {key}"

    def test_run_id_echoed(self, tmp_path):
        scores = make_scores()
        run    = _build_run(tmp_path, scores)
        result = _run_engine(tmp_path, run, make_model(scores))
        assert result["run_id"] == run["run_id"]

    def test_n_rows_matches_csv(self, tmp_path):
        scores = make_scores()
        run    = _build_run(tmp_path, scores)
        result = _run_engine(tmp_path, run, make_model(scores))
        assert result["n_rows"] == N_ROWS


class TestOutcomes:
    def test_perfect_model_is_pass(self, tmp_path):
        scores = make_scores()
        result = _run_engine(tmp_path, _build_run(tmp_path, scores), make_model(scores))
        assert result["outcome"] == "PASS"
        assert result["fidelity_pct"] == 100.0
        assert result["n_failing"]    == 0

    def test_neural_family_is_skipped(self, tmp_path):
        scores = make_scores()
        run    = _build_run(tmp_path, scores, model_family="Neural")
        result = _run_engine(tmp_path, run, MagicMock())
        assert result["outcome"] == "SKIPPED"
        assert set(result.keys()) == {"run_id", "outcome"}

    def test_none_model_obj_is_skipped(self, tmp_path):
        scores = make_scores()
        run    = _build_run(tmp_path, scores)
        audit_log = tmp_path / "audit.ndjson"
        with patch("validation.engine.load_model_artifacts",
                   return_value=(None, FEATURE_NAMES, {})):
            with patch("validation.inferencer.REPORTS_DIR", tmp_path):
                with patch("validation.audit_writer.REPORTS_DIR", tmp_path):
                    with patch("validation.audit_writer.AUDIT_LOG", audit_log):
                        from validation.engine import ValidationEngine
                        result = ValidationEngine().run(run)
        assert result["outcome"] == "SKIPPED"

    def test_warn_outcome_at_97pct_fidelity(self, tmp_path):
        """3 rows with delta 0.5 → 97% fidelity → WARN."""
        scores  = make_scores(seed=10)
        shifted = scores.copy()
        shifted[:3] = (scores[:3] + 0.5) % 1.0   # delta = 0.5 for first 3 rows
        bad_model = make_model(scores)
        bad_model.predict_proba.return_value = np.column_stack(
            [1.0 - shifted, shifted]
        )
        result = _run_engine(tmp_path, _build_run(tmp_path, scores), bad_model)
        assert result["outcome"] == "WARN"
        assert result["n_failing"] == 3

    def test_fail_outcome_for_random_model(self, tmp_path):
        """Model returning random scores → very low fidelity → FAIL."""
        scores = make_scores(seed=0)
        rand   = np.random.default_rng(999).uniform(0.0, 1.0, N_ROWS)
        bad_model = make_model(scores)
        bad_model.predict_proba.return_value = np.column_stack([1.0 - rand, rand])
        result = _run_engine(tmp_path, _build_run(tmp_path, scores), bad_model)
        assert result["outcome"] == "FAIL"


class TestSideEffects:
    def test_audit_log_written_on_pass(self, tmp_path):
        scores    = make_scores()
        run       = _build_run(tmp_path, scores)
        audit_log = tmp_path / "audit.ndjson"
        with patch("validation.engine.load_model_artifacts",
                   return_value=(make_model(scores), FEATURE_NAMES, {})):
            with patch("validation.inferencer.REPORTS_DIR", tmp_path):
                with patch("validation.audit_writer.REPORTS_DIR", tmp_path):
                    with patch("validation.audit_writer.AUDIT_LOG", audit_log):
                        from validation.engine import ValidationEngine
                        ValidationEngine().run(run)
        assert audit_log.exists()

    def test_audit_record_matches_result(self, tmp_path):
        scores    = make_scores()
        run       = _build_run(tmp_path, scores)
        audit_log = tmp_path / "audit.ndjson"
        with patch("validation.engine.load_model_artifacts",
                   return_value=(make_model(scores), FEATURE_NAMES, {})):
            with patch("validation.inferencer.REPORTS_DIR", tmp_path):
                with patch("validation.audit_writer.REPORTS_DIR", tmp_path):
                    with patch("validation.audit_writer.AUDIT_LOG", audit_log):
                        from validation.engine import ValidationEngine
                        result = ValidationEngine().run(run)
        record = json.loads(audit_log.read_text().strip())
        assert record["run_id"]      == result["run_id"]
        assert record["outcome"]     == result["outcome"]
        assert record["fidelity_pct"] == pytest.approx(result["fidelity_pct"])

    def test_no_audit_log_for_skipped_run(self, tmp_path):
        scores    = make_scores()
        run       = _build_run(tmp_path, scores, model_family="Neural")
        audit_log = tmp_path / "audit.ndjson"
        _run_engine(tmp_path, run, MagicMock())
        assert not audit_log.exists()

    def test_validated_csv_file_created(self, tmp_path):
        scores = make_scores()
        result = _run_engine(tmp_path, _build_run(tmp_path, scores), make_model(scores))
        assert Path(result["validated_csv"]).exists()

    def test_traceability_hashes_in_result(self, tmp_path):
        scores = make_scores()
        result = _run_engine(tmp_path, _build_run(tmp_path, scores), make_model(scores))
        for key in ("H_data", "H_model_version", "H_output", "H_decision"):
            assert result[key].startswith("sha256:"), f"{key} missing sha256 prefix"
            assert len(result[key]) == 71, f"{key} wrong length"

    def test_traceability_hashes_in_audit_record(self, tmp_path):
        scores    = make_scores()
        run       = _build_run(tmp_path, scores)
        audit_log = tmp_path / "audit.ndjson"
        with patch("validation.engine.load_model_artifacts",
                   return_value=(make_model(scores), FEATURE_NAMES, {})):
            with patch("validation.inferencer.REPORTS_DIR", tmp_path):
                with patch("validation.audit_writer.REPORTS_DIR", tmp_path):
                    with patch("validation.audit_writer.AUDIT_LOG", audit_log):
                        from validation.engine import ValidationEngine
                        result = ValidationEngine().run(run)
        record = json.loads(audit_log.read_text().strip())
        for key in ("H_data", "H_model_version", "H_output", "H_decision"):
            assert record[key] == result[key], f"{key} differs between result and audit log"

    def test_h_decision_binding_verifiable(self, tmp_path):
        """H_decision must equal SHA256(H_data || H_model_version || H_output)."""
        import hashlib
        scores = make_scores()
        result = _run_engine(tmp_path, _build_run(tmp_path, scores), make_model(scores))
        combined = result["H_data"][7:] + result["H_model_version"][7:] + result["H_output"][7:]
        expected = "sha256:" + hashlib.sha256(combined.encode()).hexdigest()
        assert result["H_decision"] == expected

    def test_row_hash_column_in_validated_csv(self, tmp_path):
        """Validated CSV must contain a row_hash column; all values unique and sha256-prefixed."""
        scores = make_scores()
        result = _run_engine(tmp_path, _build_run(tmp_path, scores), make_model(scores))
        df = pd.read_csv(result["validated_csv"])
        assert "row_hash" in df.columns, "row_hash column missing from validated CSV"
        assert df["row_hash"].str.startswith("sha256:").all()
        assert df["row_hash"].nunique() == len(df), "row hashes are not unique"

    def test_row_hash_independently_verifiable(self, tmp_path):
        """Each row_hash must equal h_row(client_id, H_model_version, computed_score)."""
        import hashlib as _hl
        from validation.hasher import h_row as _h_row
        scores = make_scores()
        result = _run_engine(tmp_path, _build_run(tmp_path, scores), make_model(scores))
        df     = pd.read_csv(result["validated_csv"])
        h_mv   = result["H_model_version"]
        for _, row in df.iterrows():
            expected = _h_row(row["Client_ID"], h_mv, row[f"computed_{SCORE_COL}"])
            assert row["row_hash"] == expected, f"row_hash mismatch for Client_ID={row['Client_ID']}"

    def test_idempotent_hashes_same_run_twice(self, tmp_path):
        """Same data + model → identical H_data, H_model_version, H_output, H_decision."""
        scores = make_scores()
        model  = make_model(scores)
        run2_dir = tmp_path / "run2"
        run2_dir.mkdir()
        r1 = _run_engine(tmp_path, _build_run(tmp_path, scores), model)
        r2 = _run_engine(run2_dir, _build_run(run2_dir, scores), model)
        for key in ("H_data", "H_model_version", "H_output"):
            assert r1[key] == r2[key], f"{key} is not idempotent"

    def test_tolerance_override_respected(self, tmp_path):
        """Tolerance in run config overrides engine default."""
        scores = make_scores()
        run    = _build_run(tmp_path, scores)
        run["tolerance"] = 1.0   # accept everything — even a random model passes
        rand  = np.random.default_rng(888).uniform(0.0, 1.0, N_ROWS)
        model = make_model(scores)
        model.predict_proba.return_value = np.column_stack([1.0 - rand, rand])
        result = _run_engine(tmp_path, run, model)
        assert result["outcome"] == "PASS"
