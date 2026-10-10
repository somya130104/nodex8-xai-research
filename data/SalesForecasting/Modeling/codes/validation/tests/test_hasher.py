"""Tests for validation.hasher — all four hash functions."""
from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from validation.hasher import h_data, h_decision, h_model_version, h_output, h_row

FEATURES = ["feat_b", "feat_a", "feat_c"]   # intentionally unsorted


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def base_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "feat_a": rng.uniform(0, 1, 50).round(6),
        "feat_b": rng.uniform(0, 1, 50).round(6),
        "feat_c": rng.uniform(0, 1, 50).round(6),
        "target": rng.integers(0, 2, 50),
    })


@pytest.fixture
def scores():
    return np.random.default_rng(1).uniform(0, 1, 50)


def _mock_model(n_features: int = 3):
    """Lightweight mock that satisfies h_model_version's attribute accesses."""
    m = MagicMock()
    m.__class__.__name__  = "RandomForestClassifier"
    m.__class__.__module__ = "sklearn.ensemble._forest"
    m.get_params.return_value = {"n_estimators": 100, "max_depth": None}
    m.classes_ = np.array([0, 1])
    m.feature_importances_ = np.ones(n_features, dtype=np.float64) / n_features
    # RandomForest path: has estimators_, not get_booster
    m.estimators_ = []
    if hasattr(m, "get_booster"):
        del m.get_booster
    return m


# ── h_data ────────────────────────────────────────────────────────────────────

class TestHData:
    def test_returns_sha256_prefix(self, base_df):
        result = h_data(base_df, FEATURES)
        assert result.startswith("sha256:")
        assert len(result) == 71   # "sha256:" + 64 hex chars

    def test_deterministic(self, base_df):
        assert h_data(base_df, FEATURES) == h_data(base_df, FEATURES)

    def test_column_order_independent(self, base_df):
        shuffled = base_df[["feat_c", "target", "feat_a", "feat_b"]]
        assert h_data(base_df, FEATURES) == h_data(shuffled, FEATURES)

    def test_non_feature_columns_included(self, base_df):
        df_extra = base_df.copy()
        df_extra["extra_col"] = 99   # adding any column changes the hash
        assert h_data(base_df, FEATURES) != h_data(df_extra, FEATURES)

    def test_row_order_independent_with_client_id(self, base_df):
        df = base_df.copy()
        df["Client_ID"] = np.arange(50)
        shuffled = df.sample(frac=1, random_state=7).reset_index(drop=True)
        assert h_data(df, FEATURES) == h_data(shuffled, FEATURES)

    def test_nan_filled_with_zero(self, base_df):
        df_nan = base_df.copy()
        df_nan.loc[0, "feat_a"] = float("nan")
        df_zero = base_df.copy()
        df_zero.loc[0, "feat_a"] = 0.0
        assert h_data(df_nan, FEATURES) == h_data(df_zero, FEATURES)

    def test_different_values_give_different_hash(self, base_df):
        modified = base_df.copy()
        modified.loc[0, "feat_a"] += 0.999
        assert h_data(base_df, FEATURES) != h_data(modified, FEATURES)

    def test_non_numeric_columns_included(self, base_df):
        df_seg = base_df.copy()
        df_seg["segment"] = "DEV"
        df_other = base_df.copy()
        df_other["segment"] = "OOT"
        assert h_data(df_seg, FEATURES) != h_data(df_other, FEATURES)


# ── h_model_version ───────────────────────────────────────────────────────────

class TestHModelVersion:
    def test_returns_sha256_prefix(self):
        result = h_model_version(_mock_model(), FEATURES)
        assert result.startswith("sha256:")
        assert len(result) == 71

    def test_deterministic(self):
        m = _mock_model()
        assert h_model_version(m, FEATURES) == h_model_version(m, FEATURES)

    def test_feature_order_independent(self):
        m = _mock_model()
        assert h_model_version(m, FEATURES) == h_model_version(m, list(reversed(FEATURES)))

    def test_different_hyperparams_give_different_hash(self):
        m1 = _mock_model()
        m2 = _mock_model()
        m1.get_params.return_value = {"n_estimators": 100}
        m2.get_params.return_value = {"n_estimators": 200}
        assert h_model_version(m1, FEATURES) != h_model_version(m2, FEATURES)

    def test_different_feature_importances_give_different_hash(self):
        m1 = _mock_model()
        m2 = _mock_model()
        m1.feature_importances_ = np.array([0.5, 0.3, 0.2])
        m2.feature_importances_ = np.array([0.1, 0.6, 0.3])
        assert h_model_version(m1, FEATURES) != h_model_version(m2, FEATURES)

    def test_different_feature_sets_give_different_hash(self):
        m = _mock_model()
        h1 = h_model_version(m, ["feat_a", "feat_b"])
        h2 = h_model_version(m, ["feat_a", "feat_b", "feat_c"])
        assert h1 != h2


# ── h_output ─────────────────────────────────────────────────────────────────

class TestHOutput:
    def test_returns_sha256_prefix(self, scores):
        result = h_output(scores)
        assert result.startswith("sha256:")
        assert len(result) == 71

    def test_deterministic(self, scores):
        assert h_output(scores) == h_output(scores)

    def test_row_order_normalised_by_row_ids(self, scores):
        row_ids = np.arange(50)
        shuffled_idx = np.random.default_rng(5).permutation(50)
        h1 = h_output(scores, row_ids=row_ids)
        h2 = h_output(scores[shuffled_idx], row_ids=row_ids[shuffled_idx])
        assert h1 == h2

    def test_no_row_ids_uses_original_order(self, scores):
        shuffled = scores[::-1]
        assert h_output(scores) != h_output(shuffled)

    def test_different_scores_give_different_hash(self, scores):
        modified = scores.copy()
        modified[0] += 0.5
        assert h_output(scores) != h_output(modified)

    def test_precision_normalised_to_6dp(self, scores):
        high_prec = scores.copy()
        rounded   = np.round(scores, 6)
        assert h_output(high_prec) == h_output(rounded)


# ── h_decision ────────────────────────────────────────────────────────────────

class TestHDecision:
    H1 = "sha256:" + "a" * 64
    H2 = "sha256:" + "b" * 64
    H3 = "sha256:" + "c" * 64

    def test_returns_sha256_prefix(self):
        result = h_decision(self.H1, self.H2, self.H3)
        assert result.startswith("sha256:")
        assert len(result) == 71

    def test_deterministic(self):
        assert h_decision(self.H1, self.H2, self.H3) == h_decision(self.H1, self.H2, self.H3)

    def test_changes_when_h_data_changes(self):
        alt = "sha256:" + "d" * 64
        assert h_decision(self.H1, self.H2, self.H3) != h_decision(alt, self.H2, self.H3)

    def test_changes_when_h_model_changes(self):
        alt = "sha256:" + "d" * 64
        assert h_decision(self.H1, self.H2, self.H3) != h_decision(self.H1, alt, self.H3)

    def test_changes_when_h_output_changes(self):
        alt = "sha256:" + "d" * 64
        assert h_decision(self.H1, self.H2, self.H3) != h_decision(self.H1, self.H2, alt)

    def test_binding_formula_matches_manual_calculation(self):
        combined  = self.H1[7:] + self.H2[7:] + self.H3[7:]
        expected  = "sha256:" + hashlib.sha256(combined.encode()).hexdigest()
        assert h_decision(self.H1, self.H2, self.H3) == expected

    def test_order_matters(self):
        h_abc = h_decision(self.H1, self.H2, self.H3)
        h_bac = h_decision(self.H2, self.H1, self.H3)
        assert h_abc != h_bac


# ── h_row ─────────────────────────────────────────────────────────────────────

class TestHRow:
    H_MV = "sha256:" + "e" * 64   # dummy H_model_version

    def test_returns_sha256_prefix(self):
        result = h_row(42, self.H_MV, 0.75)
        assert result.startswith("sha256:")
        assert len(result) == 71

    def test_deterministic(self):
        assert h_row(42, self.H_MV, 0.75) == h_row(42, self.H_MV, 0.75)

    def test_changes_when_client_id_changes(self):
        assert h_row(42, self.H_MV, 0.75) != h_row(99, self.H_MV, 0.75)

    def test_changes_when_model_version_changes(self):
        other = "sha256:" + "f" * 64
        assert h_row(42, self.H_MV, 0.75) != h_row(42, other, 0.75)

    def test_changes_when_score_changes(self):
        assert h_row(42, self.H_MV, 0.75) != h_row(42, self.H_MV, 0.76)

    def test_score_precision_normalised_to_6dp(self):
        assert h_row(42, self.H_MV, 0.750000) == h_row(42, self.H_MV, 0.7500001)

    def test_formula_matches_manual_calculation(self):
        parts    = f"42|{'e' * 64}|0.75"
        expected = "sha256:" + hashlib.sha256(parts.encode()).hexdigest()
        assert h_row(42, self.H_MV, 0.75) == expected

    def test_string_client_id_works(self):
        result = h_row("CLIENT_001", self.H_MV, 0.5)
        assert result.startswith("sha256:")
        assert len(result) == 71
