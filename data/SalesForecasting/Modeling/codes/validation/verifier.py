"""
Chain verifier for validation_audit.ndjson.

Each audit record contains two self-sealing fields:

  prev_hash   — record_hash of the immediately preceding record.
                The first chained record uses sentinel "sha256:000...0".
  record_hash — SHA-256 of the complete record JSON with record_hash set to "".

verify_chain() walks every line and checks:
  1. record_hash is correct  (content tamper check)
  2. prev_hash matches the previous record's record_hash  (chain continuity check)

Legacy records written before chain hashing was introduced have no prev_hash
field; they pass only the record_hash check, which is still meaningful.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_SENTINEL = "sha256:" + "0" * 64


def verify_chain(audit_log_path) -> dict:
    """
    Verify the integrity of an NDJSON audit log.

    Returns on success:
        {"ok": True, "n_records": N}

    Returns on first broken record:
        {"ok": False, "n_records": N, "first_broken_at": i,
         "run_id": "...", "reason": "..."}

    "first_broken_at" is the 0-based index of the earliest invalid record.
    """
    path = Path(audit_log_path)
    if not path.exists() or path.stat().st_size == 0:
        return {"ok": True, "n_records": 0}

    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))

    expected_prev = _SENTINEL

    for i, record in enumerate(records):
        stored_hash = record.get("record_hash", "")

        # ── Check 1: record_hash (content tamper) ────────────────────────────
        check = {k: ("" if k == "record_hash" else v) for k, v in record.items()}
        recomputed = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(check, separators=(",", ":")).encode()
            ).hexdigest()
        )
        if stored_hash != recomputed:
            return {
                "ok": False,
                "n_records": len(records),
                "first_broken_at": i,
                "run_id": record.get("run_id", "?"),
                "reason": (
                    f"record {i} (run_id={record.get('run_id', '?')}): "
                    "record_hash mismatch — content was modified"
                ),
            }

        # ── Check 2: prev_hash continuity (chain tamper) ─────────────────────
        if "prev_hash" in record:
            if record["prev_hash"] != expected_prev:
                return {
                    "ok": False,
                    "n_records": len(records),
                    "first_broken_at": i,
                    "run_id": record.get("run_id", "?"),
                    "reason": (
                        f"record {i} (run_id={record.get('run_id', '?')}): "
                        "prev_hash mismatch — a preceding record may have been "
                        "deleted, reordered, or replaced"
                    ),
                }

        expected_prev = stored_hash

    return {"ok": True, "n_records": len(records)}
