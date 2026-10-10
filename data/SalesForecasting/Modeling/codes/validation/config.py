from pathlib import Path

BASE_DIR    = Path(__file__).parents[1]
REPORTS_DIR = BASE_DIR / "validation_reports"
AUDIT_LOG   = REPORTS_DIR / "validation_audit.ndjson"

DEFAULT_ATOL       = 0.05
PASS_THRESHOLD_PCT = 99.0
WARN_THRESHOLD_PCT = 95.0
