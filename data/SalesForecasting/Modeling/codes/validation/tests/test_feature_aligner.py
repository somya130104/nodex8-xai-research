"""Tests for validation.feature_aligner — align()."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from validation.feature_aligner import FeatureAlignmentError, align

FEATURES = ["a", "b", "c"]


def _df(**kwargs) -> pd.DataFrame:
    return pd.DataFrame({k: [v] for k, v in kwargs.items()})


class TestAlignHappyPath:
    def test_returns_numpy_float_array(self):
        X = align(_df(a=1.0, b=2.0, c=3.0), FEATURES)
        assert isinstance(X, np.ndarray)
        assert X.dtype == float

    def test_column_order_follows_features_not_df(self):
        df = _df(c=3.0, a=1.0, b=2.0)  # different column order in df
        X  = align(df, FEATURES)
        assert X[0, 0] == 1.0   # "a" first
        assert X[0, 1] == 2.0   # "b" second
        assert X[0, 2] == 3.0   # "c" third

    def test_shape_matches_rows_and_features(self):
        n   = 25
        df  = pd.DataFrame({f: np.ones(n) for f in FEATURES})
        X   = align(df, FEATURES)
        assert X.shape == (n, len(FEATURES))

    def test_extra_df_columns_are_ignored(self):
        df = _df(a=1, b=2, c=3, irrelevant=99)
        X  = align(df, FEATURES)
        assert X.shape == (1, 3)

    def test_nan_filled_with_zero(self):
        df = pd.DataFrame({"a": [np.nan], "b": [1.0], "c": [2.0]})
        X  = align(df, FEATURES)
        assert X[0, 0] == 0.0

    def test_integer_columns_cast_to_float(self):
        df = _df(a=1, b=2, c=3)
        X  = align(df, FEATURES)
        assert X.dtype == float


class TestAlignErrors:
    def test_missing_single_feature_raises(self):
        df = _df(a=1, b=2)   # "c" absent
        with pytest.raises(FeatureAlignmentError):
            align(df, FEATURES)

    def test_error_message_names_missing_feature(self):
        df = _df(a=1, b=2)
        with pytest.raises(FeatureAlignmentError, match="c"):
            align(df, FEATURES)

    def test_all_features_missing_raises(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        with pytest.raises(FeatureAlignmentError):
            align(df, FEATURES)

    def test_error_is_subclass_of_value_error(self):
        df = _df(a=1)
        with pytest.raises(ValueError):
            align(df, FEATURES)
