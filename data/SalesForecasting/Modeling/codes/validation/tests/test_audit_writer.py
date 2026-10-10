"""Tests for validation.audit_writer — SHA-256 helpers and write()."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from validation.audit_writer import _sha256_file, _sha256_str, write


# ── SHA-256 helper tests ──────────────────────────────────────────────────────

class TestSha256File:
    def test_returns_sha256_prefixed_hex(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"hello world")
        result = _sha256_file(str(f))
        assert result.startswith("sha256:")
        assert len(result) == len("sha256:") + 64

    def test_deterministic_for_same_content(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"deterministic")
        assert _sha256_file(str(f)) == _sha256_file(str(f))

    def test_different_content_gives_different_hash(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"content-a")
        f2.write_bytes(b"content-b")
        assert _sha256_file(str(f1)) != _sha256_file(str(f2))

    def test_hash_matches_expected_value(self, tmp_path):
        content = b"test content"
        f = tmp_path / "test.bin"
        f.write_bytes(content)
        expected = "sha256:" + hashlib.sha256(content).hexdigest()
        assert _sha256_file(str(f)) == expected


class TestSha256Str:
    def test_returns_sha256_prefixed_hex(self):
        result = _sha256_str("hello")
        assert result.startswith("sha256:")

    def test_matches_expected_digest(self):
        result   = _sha256_str("hello")
        expected = "sha256:" + hashlib.sha256(b"hello").hexdigest()
        assert result == expected

    def test_empty_string(self):
        result = _sha256_str("")
        assert result.startswith("sha256:")
        assert len(result) == len("sha256:") + 64


# ── write() tests ─────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_run():
    return {
        "run_id":         "TestModel_OOS",
        "model":          "TestModel",
        "model_family":   "Tree",
        "dataset":        "OOS",
        "path":           "",
        "model_pkl_path": "",
        "score_col":      "fraud_score",
    }


@pytest.fixture
def sample_stats():
    return {
        "n_rows":         100,
        "n_passing":      98,
        "n_failing":      2,
        "max_abs_delta":  0.08,
        "mean_abs_delta": 0.002,
        "p95_abs_delta":  0.05,
        "p99_abs_delta":  0.07,
    }


@pytest.fixture
def validated_csv(tmp_path) -> str:
    p = tmp_path / "validated.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    return str(p)


def _do_write(tmp_path, run, stats, validated_csv, **kwargs):
    audit_log = tmp_path / "audit.ndjson"
    defaults  = dict(
        run=run,
        data_hash="sha256:aabb",
        model_hash="sha256:ccdd",
        validated_csv_path=validated_csv,
        n_features_aligned=5,
        added_columns=["computed_fraud_score", "abs_delta_fraud_score"],
        stats=stats,
        fidelity_pct=98.0,
        outcome="WARN",
        atol=0.05,
    )
    defaults.update(kwargs)
    with patch("validation.audit_writer.REPORTS_DIR", tmp_path):
        with patch("validation.audit_writer.AUDIT_LOG", audit_log):
            write(**defaults)
    return audit_log


class TestWrite:
    def test_creates_audit_log(self, tmp_path, minimal_run, sample_stats, validated_csv):
        log = _do_write(tmp_path, minimal_run, sample_stats, validated_csv)
        assert log.exists()

    def test_appends_one_line_per_call(self, tmp_path, minimal_run, sample_stats, validated_csv):
        log = _do_write(tmp_path, minimal_run, sample_stats, validated_csv)
        assert len(log.read_text().strip().splitlines()) == 1

    def test_second_call_appends_second_line(self, tmp_path, minimal_run, sample_stats, validated_csv):
        _do_write(tmp_path, minimal_run, sample_stats, validated_csv)
        _do_write(tmp_path, minimal_run, sample_stats, validated_csv)
        log = tmp_path / "audit.ndjson"
        assert len(log.read_text().strip().splitlines()) == 2

    def test_record_is_valid_json(self, tmp_path, minimal_run, sample_stats, validated_csv):
        log    = _do_write(tmp_path, minimal_run, sample_stats, validated_csv)
        record = json.loads(log.read_text().strip())
        assert isinstance(record, dict)

    def test_required_fields_present(self, tmp_path, minimal_run, sample_stats, validated_csv):
        log    = _do_write(tmp_path, minimal_run, sample_stats, validated_csv)
        record = json.loads(log.read_text().strip())
        for key in ("run_id", "outcome", "fidelity_pct", "record_hash", "timestamp",
                    "n_rows", "n_failing", "atol"):
            assert key in record, f"missing key: {key}"

    def test_record_hash_is_populated(self, tmp_path, minimal_run, sample_stats, validated_csv):
        log    = _do_write(tmp_path, minimal_run, sample_stats, validated_csv)
        record = json.loads(log.read_text().strip())
        assert record["record_hash"].startswith("sha256:")
        assert len(record["record_hash"]) == len("sha256:") + 64

    def test_run_id_matches(self, tmp_path, minimal_run, sample_stats, validated_csv):
        log    = _do_write(tmp_path, minimal_run, sample_stats, validated_csv)
        record = json.loads(log.read_text().strip())
        assert record["run_id"] == minimal_run["run_id"]

    def test_outcome_stored_correctly(self, tmp_path, minimal_run, sample_stats, validated_csv):
        log    = _do_write(tmp_path, minimal_run, sample_stats, validated_csv,
                           outcome="PASS", fidelity_pct=100.0)
        record = json.loads(log.read_text().strip())
        assert record["outcome"] == "PASS"

    def test_fidelity_pct_stored(self, tmp_path, minimal_run, sample_stats, validated_csv):
        log    = _do_write(tmp_path, minimal_run, sample_stats, validated_csv,
                           fidelity_pct=97.5)
        record = json.loads(log.read_text().strip())
        assert record["fidelity_pct"] == pytest.approx(97.5)

    def test_record_hash_differs_per_run(self, tmp_path, minimal_run, sample_stats, validated_csv):
        log = _do_write(tmp_path, minimal_run, sample_stats, validated_csv)
        log2_path = tmp_path / "audit2.ndjson"
        minimal_run2 = {**minimal_run, "run_id": "OtherModel_OOS"}
        with patch("validation.audit_writer.REPORTS_DIR", tmp_path):
            with patch("validation.audit_writer.AUDIT_LOG", log2_path):
                write(
                    run=minimal_run2,
                    data_hash="sha256:aabb",
                    model_hash="sha256:ccdd",
                    validated_csv_path=validated_csv,
                    n_features_aligned=5,
                    added_columns=[],
                    stats=sample_stats,
                    fidelity_pct=98.0,
                    outcome="WARN",
                    atol=0.05,
                )
        r1 = json.loads(log.read_text().strip())
        r2 = json.loads(log2_path.read_text().strip())
        assert r1["record_hash"] != r2["record_hash"]
