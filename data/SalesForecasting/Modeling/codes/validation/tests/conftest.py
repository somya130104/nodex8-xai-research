"""
Shared fixtures for the validation test suite.

shap is an optional heavy dependency not installed in this venv.
Stub it before any engine import so src/shap_utils.py doesn't blow up.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Must run before any validation.engine import
sys.modules.setdefault("shap", MagicMock())

import pytest  # noqa: E402 — after shap stub

from validation.tests.helpers import (  # noqa: E402
    ATOL, FEATURE_NAMES, N_ROWS, PRED_COL, SCORE_COL,
    make_df, make_model, make_scores,
)


@pytest.fixture
def scores() -> "np.ndarray":
    return make_scores()


@pytest.fixture
def sample_df(scores):
    return make_df(scores)


@pytest.fixture
def perfect_model(scores):
    """Model whose output matches the stored scores exactly — delta = 0."""
    return make_model(scores)


@pytest.fixture
def run_cfg(tmp_path, sample_df) -> dict:
    """Minimal run-config dict with real CSV and a stub pkl written to tmp_path."""
    csv_path = tmp_path / "preds.csv"
    pkl_path = tmp_path / "model.pkl"
    sample_df.to_csv(csv_path, index=False)
    pkl_path.write_bytes(b"fake-pkl-content")
    return {
        "run_id":         "MockModel_DEV",
        "name":           "MockModel_DEV",
        "model":          "MockModel",
        "model_domain":   "Fraud",
        "model_type":     "Classification",
        "model_family":   "Tree",
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
