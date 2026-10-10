"""Tests for validation.verifier — chain integrity verification."""
from __future__ import annotations

import hashlib
import json

import pytest

from validation.verifier import _SENTINEL, verify_chain


# ── Test helpers ──────────────────────────────────────────────────────────────

def _make_record(run_id: str, prev_hash: str, **fields) -> dict:
    """Build a correctly self-sealed, correctly chained audit record."""
    record = {
        "timestamp": "2024-01-01T00:00:00+00:00",
        "run_id": run_id,
        "outcome": "PASS",
        "fidelity_pct": 100.0,
        "— self-seal —": "",
        "prev_hash": prev_hash,
        "record_hash": "",
    }
    record.update(fields)
    record["record_hash"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(record, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return record


def _write_log(path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestVerifyChain:

    # ── Edge cases ─────────────────────────────────────────────────────────────

    def test_nonexistent_file_is_valid(self, tmp_path):
        result = verify_chain(tmp_path / "missing.ndjson")
        assert result["ok"] is True
        assert result["n_records"] == 0

    def test_empty_file_is_valid(self, tmp_path):
        log = tmp_path / "audit.ndjson"
        log.write_text("")
        result = verify_chain(log)
        assert result["ok"] is True
        assert result["n_records"] == 0

    # ── Valid chains ───────────────────────────────────────────────────────────

    def test_single_record_is_valid(self, tmp_path):
        log = tmp_path / "audit.ndjson"
        _write_log(log, [_make_record("Run1", _SENTINEL)])
        assert verify_chain(log) == {"ok": True, "n_records": 1}

    def test_three_record_chain_is_valid(self, tmp_path):
        log = tmp_path / "audit.ndjson"
        r1 = _make_record("Run1", _SENTINEL)
        r2 = _make_record("Run2", r1["record_hash"])
        r3 = _make_record("Run3", r2["record_hash"])
        _write_log(log, [r1, r2, r3])
        result = verify_chain(log)
        assert result["ok"] is True
        assert result["n_records"] == 3

    # ── Content tampering ──────────────────────────────────────────────────────

    def test_tampered_field_in_first_record_detected(self, tmp_path):
        log = tmp_path / "audit.ndjson"
        r1 = _make_record("Run1", _SENTINEL)
        r1_tampered = dict(r1)
        r1_tampered["outcome"] = "FAIL"   # change but keep old record_hash
        _write_log(log, [r1_tampered])
        result = verify_chain(log)
        assert result["ok"] is False
        assert result["first_broken_at"] == 0
        assert "record_hash mismatch" in result["reason"]

    def test_tampered_field_in_middle_record_detected(self, tmp_path):
        log = tmp_path / "audit.ndjson"
        r1 = _make_record("Run1", _SENTINEL)
        r2 = _make_record("Run2", r1["record_hash"])
        r3 = _make_record("Run3", r2["record_hash"])
        r2_tampered = dict(r2)
        r2_tampered["fidelity_pct"] = 50.0  # change but keep old record_hash
        _write_log(log, [r1, r2_tampered, r3])
        result = verify_chain(log)
        assert result["ok"] is False
        assert result["first_broken_at"] == 1
        assert "record_hash mismatch" in result["reason"]
        assert result["run_id"] == "Run2"

    def test_tampered_field_in_last_record_detected(self, tmp_path):
        log = tmp_path / "audit.ndjson"
        r1 = _make_record("Run1", _SENTINEL)
        r2 = _make_record("Run2", r1["record_hash"])
        r2_tampered = dict(r2)
        r2_tampered["outcome"] = "FAIL"
        _write_log(log, [r1, r2_tampered])
        result = verify_chain(log)
        assert result["ok"] is False
        assert result["first_broken_at"] == 1

    # ── Chain / ordering tampering ─────────────────────────────────────────────

    def test_deleted_record_breaks_chain(self, tmp_path):
        """Removing a record from the middle causes the next record's prev_hash to break."""
        log = tmp_path / "audit.ndjson"
        r1 = _make_record("Run1", _SENTINEL)
        r2 = _make_record("Run2", r1["record_hash"])
        r3 = _make_record("Run3", r2["record_hash"])
        _write_log(log, [r1, r3])   # r2 deleted
        result = verify_chain(log)
        assert result["ok"] is False
        assert result["first_broken_at"] == 1
        assert "prev_hash mismatch" in result["reason"]

    def test_reordered_records_break_chain(self, tmp_path):
        """Swapping two records breaks the chain at the first mis-ordered record."""
        log = tmp_path / "audit.ndjson"
        r1 = _make_record("Run1", _SENTINEL)
        r2 = _make_record("Run2", r1["record_hash"])
        r3 = _make_record("Run3", r2["record_hash"])
        _write_log(log, [r1, r3, r2])   # r2 and r3 swapped
        result = verify_chain(log)
        assert result["ok"] is False
        assert result["first_broken_at"] == 1

    def test_injected_record_breaks_chain(self, tmp_path):
        """Inserting a forged record in the middle breaks continuity."""
        log = tmp_path / "audit.ndjson"
        r1 = _make_record("Run1", _SENTINEL)
        r2 = _make_record("Run2", r1["record_hash"])
        forged = _make_record("Forged", r1["record_hash"])  # same prev as r2
        _write_log(log, [r1, forged, r2])
        result = verify_chain(log)
        assert result["ok"] is False
        assert result["first_broken_at"] == 2   # r2's prev_hash no longer matches

    # ── Legacy / backward compat ───────────────────────────────────────────────

    def test_legacy_record_without_prev_hash_passes(self, tmp_path):
        """Records written before chain hashing have no prev_hash; still verifiable."""
        log = tmp_path / "audit.ndjson"
        legacy = {"run_id": "OldRun", "outcome": "PASS", "record_hash": ""}
        legacy["record_hash"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(legacy, separators=(",", ":")).encode()
            ).hexdigest()
        )
        _write_log(log, [legacy])
        result = verify_chain(log)
        assert result["ok"] is True
        assert result["n_records"] == 1

    def test_legacy_then_chained_records_verify(self, tmp_path):
        """Legacy records followed by chained records all pass."""
        log = tmp_path / "audit.ndjson"
        legacy = {"run_id": "OldRun", "outcome": "PASS", "record_hash": ""}
        legacy["record_hash"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(legacy, separators=(",", ":")).encode()
            ).hexdigest()
        )
        chained = _make_record("NewRun", legacy["record_hash"])
        _write_log(log, [legacy, chained])
        result = verify_chain(log)
        assert result["ok"] is True
        assert result["n_records"] == 2

    def test_tampered_legacy_record_detected(self, tmp_path):
        """Even without prev_hash, a tampered legacy record_hash is caught."""
        log = tmp_path / "audit.ndjson"
        legacy = {"run_id": "OldRun", "outcome": "PASS", "record_hash": ""}
        legacy["record_hash"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(legacy, separators=(",", ":")).encode()
            ).hexdigest()
        )
        legacy["outcome"] = "FAIL"   # tamper after hash computed
        _write_log(log, [legacy])
        result = verify_chain(log)
        assert result["ok"] is False
        assert result["first_broken_at"] == 0
