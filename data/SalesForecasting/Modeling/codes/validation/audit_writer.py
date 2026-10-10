from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import AUDIT_LOG, REPORTS_DIR

_SENTINEL_HASH = "sha256:" + "0" * 64


def _last_record_hash(log_path: Path) -> str:
    """Return the record_hash of the last written record, or the sentinel for the first."""
    if not log_path.exists():
        return _SENTINEL_HASH
    last_line = ""
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                last_line = stripped
    if not last_line:
        return _SENTINEL_HASH
    try:
        return json.loads(last_line).get("record_hash", _SENTINEL_HASH)
    except (json.JSONDecodeError, KeyError):
        return _SENTINEL_HASH


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _sha256_str(s: str) -> str:
    return f"sha256:{hashlib.sha256(s.encode()).hexdigest()}"


def write(
    run: dict,
    data_hash: str,
    model_hash: str,
    validated_csv_path: str,
    n_features_aligned: int,
    added_columns: list[str],
    stats: dict,
    fidelity_pct: float,
    outcome: str,
    atol: float,
    traceability_hashes: dict | None = None,
    fidelity_metrics: dict | None = None,
) -> str:
    prev_hash = _last_record_hash(AUDIT_LOG)
    validated_csv_hash = _sha256_file(validated_csv_path)
    now = datetime.now(timezone.utc).isoformat(timespec="microseconds")

    record = {
        "timestamp":           now,
        "validation_id":       str(uuid.uuid4()),

        "— run identity —":    "",
        "run_id":              run["run_id"],
        "model_name":          run["model"],
        "model_family":        run["model_family"],
        "dataset":             run["dataset"],
        "source_csv_path":     run["path"],
        "model_pkl_path":      run["model_pkl_path"],

        "— cryptographic anchors —": "",
        "data_hash":           data_hash,
        "model_hash":          model_hash,
        "validated_csv_hash":  validated_csv_hash,

        "— dataset registration —": "",
        "validated_csv_path":  validated_csv_path,
        "registered_at":       datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "n_rows":              stats["n_rows"],
        "n_features_aligned":  n_features_aligned,
        "added_columns":       added_columns,

        "— validation result —": "",
        "score_col":           run["score_col"],
        "atol":                atol,
        "n_passing":           stats["n_passing"],
        "n_failing":           stats["n_failing"],
        "fidelity_pct":        fidelity_pct,
        "max_abs_delta":       stats["max_abs_delta"],
        "mean_abs_delta":      stats["mean_abs_delta"],
        "p95_abs_delta":       stats["p95_abs_delta"],
        "p99_abs_delta":       stats["p99_abs_delta"],
        "outcome":             outcome,

        "— traceability hashes —": "",
        "H_data":              (traceability_hashes or {}).get("H_data",          ""),
        "H_model_version":     (traceability_hashes or {}).get("H_model_version", ""),
        "H_output":            (traceability_hashes or {}).get("H_output",        ""),
        "H_decision":          (traceability_hashes or {}).get("H_decision",      ""),

        "— high-level fidelity score —": "",
        "composite_fidelity_score": (fidelity_metrics or {}).get("composite_fidelity_score", ""),
        "composite_outcome":        (fidelity_metrics or {}).get("outcome", ""),
        "decision_threshold":       (fidelity_metrics or {}).get("decision_threshold", ""),
        "tolerance_pass_rate":       (fidelity_metrics or {}).get("tolerance_pass_rate", ""),
        "decision_agreement_rate":   (fidelity_metrics or {}).get("decision_agreement_rate", ""),
        "rank_fidelity":             (fidelity_metrics or {}).get("rank_fidelity", ""),
        "top_decile_overlap":        (fidelity_metrics or {}).get("top_decile_overlap", ""),
        "distributional_similarity": (fidelity_metrics or {}).get("distributional_similarity", ""),
        "calibration_score":         (fidelity_metrics or {}).get("calibration_score", ""),
        "completeness":              (fidelity_metrics or {}).get("completeness", ""),
        "mean_signed_bias":          (fidelity_metrics or {}).get("mean_signed_bias", ""),

        "— self-seal —":       "",
        "prev_hash":           prev_hash,
        "record_hash":         "",
    }

    record["record_hash"] = _sha256_str(json.dumps(record, separators=(",", ":")))

    REPORTS_DIR.mkdir(exist_ok=True)
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")

    return str(AUDIT_LOG)
