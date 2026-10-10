"""Tests for validation.inferencer — rerun_and_enrich()."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from validation.inferencer import rerun_and_enrich

SCORE_COL = "fraud_score"
PRED_COL  = "prediction"
ATOL      = 0.05
N         = 60


@pytest.fixture
def base_scores():
    return np.random.default_rng(7).uniform(0.0, 1.0, N)


@pytest.fixture
def df(base_scores):
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        SCORE_COL: base_scores,
        PRED_COL:  (base_scores >= 0.5).astype(int),
        "feat_x":  rng.standard_normal(N),
    })


@pytest.fixture
def X(df):
    return df[["feat_x"]].values.astype(float)


@pytest.fixture
def run():
    return {"run_id": "Test_DEV", "score_col": SCORE_COL, "pred_col": PRED_COL}


@pytest.fixture
def perfect_model(base_scores):
    m     = MagicMock()
    proba = np.column_stack([1.0 - base_scores, base_scores])
    m.predict_proba.return_value = proba
    return m


class TestOutputColumns:
    def test_adds_computed_score_column(self, df, X, run, perfect_model, tmp_path):
        with patch("validation.inferencer.REPORTS_DIR", tmp_path):
            enriched, _ = rerun_and_enrich(df, perfect_model, X, run, ATOL)
        assert f"computed_{SCORE_COL}" in enriched.columns

    def test_adds_computed_pred_column(self, df, X, run, perfect_model, tmp_path):
        with patch("validation.inferencer.REPORTS_DIR", tmp_path):
            enriched, _ = rerun_and_enrich(df, perfect_model, X, run, ATOL)
        assert f"computed_{PRED_COL}" in enriched.columns

    def test_adds_abs_delta_column(self, df, X, run, perfect_model, tmp_path):
        with patch("validation.inferencer.REPORTS_DIR", tmp_path):
            enriched, _ = rerun_and_enrich(df, perfect_model, X, run, ATOL)
        assert f"abs_delta_{SCORE_COL}" in enriched.columns

    def test_adds_tolerance_pass_column(self, df, X, run, perfect_model, tmp_path):
        with patch("validation.inferencer.REPORTS_DIR", tmp_path):
            enriched, _ = rerun_and_enrich(df, perfect_model, X, run, ATOL)
        assert "tolerance_pass" in enriched.columns

    def test_computed_pred_is_binary(self, df, X, run, perfect_model, tmp_path):
        with patch("validation.inferencer.REPORTS_DIR", tmp_path):
            enriched, _ = rerun_and_enrich(df, perfect_model, X, run, ATOL)
        assert set(enriched[f"computed_{PRED_COL}"].unique()).issubset({0, 1})


class TestDeltaCorrectness:
    def test_perfect_model_gives_zero_delta(self, df, X, run, perfect_model, tmp_path):
        with patch("validation.inferencer.REPORTS_DIR", tmp_path):
            enriched, _ = rerun_and_enrich(df, perfect_model, X, run, ATOL)
        np.testing.assert_allclose(
            enriched[f"abs_delta_{SCORE_COL}"].values,
            np.zeros(N),
            atol=1e-5,
        )

    def test_perfect_model_all_tolerance_pass(self, df, X, run, perfect_model, tmp_path):
        with patch("validation.inferencer.REPORTS_DIR", tmp_path):
            enriched, _ = rerun_and_enrich(df, perfect_model, X, run, ATOL)
        assert enriched["tolerance_pass"].all()

    def test_bad_model_causes_tolerance_failures(self, df, X, run, base_scores, tmp_path):
        bad_model = MagicMock()
        # Return scores shifted by 0.5 → delta = 0.5 >> ATOL for all rows
        shifted = (base_scores + 0.5) % 1.0
        bad_model.predict_proba.return_value = np.column_stack(
            [1.0 - shifted, shifted]
        )
        with patch("validation.inferencer.REPORTS_DIR", tmp_path):
            enriched, _ = rerun_and_enrich(df, bad_model, X, run, ATOL)
        assert not enriched["tolerance_pass"].all()

    def test_delta_is_absolute(self, df, X, run, base_scores, tmp_path):
        """Delta must always be non-negative regardless of sign of difference."""
        bad_model = MagicMock()
        lower = np.clip(base_scores - 0.1, 0.0, 1.0)
        bad_model.predict_proba.return_value = np.column_stack([1.0 - lower, lower])
        with patch("validation.inferencer.REPORTS_DIR", tmp_path):
            enriched, _ = rerun_and_enrich(df, bad_model, X, run, ATOL)
        assert (enriched[f"abs_delta_{SCORE_COL}"] >= 0).all()


class TestFileOutput:
    def test_csv_saved_to_reports_dir(self, df, X, run, perfect_model, tmp_path):
        with patch("validation.inferencer.REPORTS_DIR", tmp_path):
            _, out_path = rerun_and_enrich(df, perfect_model, X, run, ATOL)
        assert Path(out_path).exists()

    def test_output_path_contains_run_id(self, df, X, run, perfect_model, tmp_path):
        with patch("validation.inferencer.REPORTS_DIR", tmp_path):
            _, out_path = rerun_and_enrich(df, perfect_model, X, run, ATOL)
        assert run["run_id"] in out_path

    def test_saved_csv_has_enriched_columns(self, df, X, run, perfect_model, tmp_path):
        with patch("validation.inferencer.REPORTS_DIR", tmp_path):
            _, out_path = rerun_and_enrich(df, perfect_model, X, run, ATOL)
        saved = pd.read_csv(out_path)
        assert f"computed_{SCORE_COL}" in saved.columns
        assert "tolerance_pass"         in saved.columns

    def test_original_df_not_mutated(self, df, X, run, perfect_model, tmp_path):
        original_cols = set(df.columns)
        with patch("validation.inferencer.REPORTS_DIR", tmp_path):
            rerun_and_enrich(df, perfect_model, X, run, ATOL)
        assert set(df.columns) == original_cols
