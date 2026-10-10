"""Tests for validation.fidelity — score()."""
from __future__ import annotations

import pytest

from validation.config import PASS_THRESHOLD_PCT, WARN_THRESHOLD_PCT
from validation.fidelity import score


class TestScoreOutcomes:
    def test_perfect_fidelity_is_pass(self):
        pct, outcome = score(100, 100)
        assert outcome == "PASS"
        assert pct == 100.0

    def test_exactly_at_pass_threshold_is_pass(self):
        # 99.0% of 1000 rows = 990 passing
        n_rows = 1000
        n_pass = int(PASS_THRESHOLD_PCT / 100 * n_rows)
        _, outcome = score(n_pass, n_rows)
        assert outcome == "PASS"

    def test_just_below_pass_threshold_is_warn(self):
        # 989/1000 = 98.9% → WARN
        n_rows = 1000
        n_pass = int(PASS_THRESHOLD_PCT / 100 * n_rows) - 1
        _, outcome = score(n_pass, n_rows)
        assert outcome == "WARN"

    def test_exactly_at_warn_threshold_is_warn(self):
        # 95.0% of 1000 rows = 950 passing
        n_rows = 1000
        n_pass = int(WARN_THRESHOLD_PCT / 100 * n_rows)
        _, outcome = score(n_pass, n_rows)
        assert outcome == "WARN"

    def test_just_below_warn_threshold_is_fail(self):
        # 949/1000 = 94.9% → FAIL
        n_rows = 1000
        n_pass = int(WARN_THRESHOLD_PCT / 100 * n_rows) - 1
        _, outcome = score(n_pass, n_rows)
        assert outcome == "FAIL"

    def test_zero_passing_is_fail(self):
        _, outcome = score(0, 100)
        assert outcome == "FAIL"

    def test_zero_rows_returns_zero_fidelity(self):
        pct, _ = score(0, 0)
        assert pct == 0.0


class TestScorePrecision:
    def test_fidelity_pct_rounded_to_4dp(self):
        pct, _ = score(1, 3)
        assert pct == round(1 / 3 * 100, 4)

    def test_returns_float_and_str(self):
        pct, outcome = score(50, 100)
        assert isinstance(pct, float)
        assert isinstance(outcome, str)

    def test_fidelity_matches_formula(self):
        pct, _ = score(73, 200)
        assert pct == pytest.approx(36.5, abs=0.0001)

    def test_all_passing_gives_100(self):
        pct, _ = score(500, 500)
        assert pct == 100.0
