"""
Shared constants and factory functions for the validation test suite.
Imported by both conftest.py (for fixtures) and test_engine.py (for helpers).
No pytest dependency — plain Python only.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

N_ROWS        = 100
FEATURE_NAMES = ["f1", "f2", "f3", "f4", "f5"]
SCORE_COL     = "fraud_score"
PRED_COL      = "prediction"
ATOL          = 0.05


def make_scores(seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.0, 1.0, N_ROWS)


def make_df(scores: np.ndarray) -> pd.DataFrame:
    rng  = np.random.default_rng(1)
    data = {f: rng.standard_normal(N_ROWS) for f in FEATURE_NAMES}
    data["Client_ID"] = np.arange(N_ROWS)
    data[SCORE_COL] = scores
    data[PRED_COL]  = (scores >= 0.5).astype(int)
    return pd.DataFrame(data)


def make_model(scores: np.ndarray) -> MagicMock:
    """
    Mock sklearn-style model whose predict_proba returns `scores`.
    feature_importances_, classes_, get_params, and estimators_ are set to
    real values so h_model_version can serialise them without errors.
    """
    m     = MagicMock()
    proba = np.column_stack([1.0 - scores, scores])
    m.predict_proba.return_value  = proba
    m.get_params.return_value     = {"n_estimators": 10, "max_depth": 3}
    m.classes_                    = np.array([0, 1])
    m.feature_importances_        = np.ones(len(FEATURE_NAMES), dtype=np.float64) / len(FEATURE_NAMES)
    m.estimators_                 = []   # triggers sklearn (not XGBoost) path in h_model_version
    return m
