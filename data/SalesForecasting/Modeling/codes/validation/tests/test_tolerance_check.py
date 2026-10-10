"""Tests for validation.tolerance_check — check()."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from validation.tolerance_check import check

SCORE_COL = "fraud_score"
ATOL      = 0.05


def _df(deltas: list[float]) -> pd.DataFrame:
    return pd.DataFrame({f"abs_delta_{SCORE_COL}": deltas})


class TestPassFail:
    def test_all_within_tolerance(self):
        stats = check(_df([0.00, 0.01, 0.05]), SCORE_COL, ATOL)
        assert stats["n_rows"]    == 3
        assert stats["n_passing"] == 3
        assert stats["n_failing"] == 0

    def test_all_exceed_tolerance(self):
        stats = check(_df([0.10, 0.20, 0.30]), SCORE_COL, ATOL)
        assert stats["n_failing"] == 3
        assert stats["n_passing"] == 0

    def test_mixed_pass_fail(self):
        stats = check(_df([0.01, 0.06, 0.10]), SCORE_COL, ATOL)
        assert stats["n_passing"] == 1
        assert stats["n_failing"] == 2

    def test_exactly_at_tolerance_boundary_passes(self):
        stats = check(_df([ATOL]), SCORE_COL, ATOL)
        assert stats["n_passing"] == 1
        assert stats["n_failing"] == 0

    def test_just_above_tolerance_fails(self):
        stats = check(_df([ATOL + 1e-7]), SCORE_COL, ATOL)
        assert stats["n_failing"] == 1

    def test_row_count_is_correct(self):
        stats = check(_df([0.0] * 42), SCORE_COL, ATOL)
        assert stats["n_rows"] == 42


class TestDeltaStats:
    def test_max_abs_delta(self):
        stats = check(_df([0.01, 0.03, 0.09]), SCORE_COL, ATOL)
        assert stats["max_abs_delta"] == pytest.approx(0.09, abs=1e-6)

    def test_mean_abs_delta(self):
        stats = check(_df([0.0, 0.1, 0.2]), SCORE_COL, ATOL)
        assert stats["mean_abs_delta"] == pytest.approx(0.1, abs=1e-5)

    def test_p99_gte_p95(self):
        deltas = np.zeros(200)
        deltas[-10:] = 0.5
        stats = check(_df(deltas.tolist()), SCORE_COL, ATOL)
        assert stats["p99_abs_delta"] >= stats["p95_abs_delta"]

    def test_p95_gte_mean(self):
        rng    = np.random.default_rng(0)
        deltas = rng.uniform(0, 0.2, 500).tolist()
        stats  = check(_df(deltas), SCORE_COL, ATOL)
        assert stats["p95_abs_delta"] >= stats["mean_abs_delta"]

    def test_all_required_keys_present(self):
        stats = check(_df([0.01]), SCORE_COL, ATOL)
        for key in ("n_rows", "n_passing", "n_failing",
                    "max_abs_delta", "mean_abs_delta",
                    "p95_abs_delta", "p99_abs_delta"):
            assert key in stats, f"missing key: {key}"

    def test_zero_delta_gives_zero_stats(self):
        stats = check(_df([0.0, 0.0, 0.0]), SCORE_COL, ATOL)
        assert stats["max_abs_delta"]  == 0.0
        assert stats["mean_abs_delta"] == 0.0
        assert stats["p95_abs_delta"]  == 0.0
