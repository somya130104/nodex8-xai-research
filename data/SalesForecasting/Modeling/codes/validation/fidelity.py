from __future__ import annotations

from .config import PASS_THRESHOLD_PCT, WARN_THRESHOLD_PCT


def score(n_passing: int, n_rows: int) -> tuple[float, str]:
    fidelity_pct = round(n_passing / n_rows * 100, 4) if n_rows else 0.0
    if fidelity_pct >= PASS_THRESHOLD_PCT:
        outcome = "PASS"
    elif fidelity_pct >= WARN_THRESHOLD_PCT:
        outcome = "WARN"
    else:
        outcome = "FAIL"
    return fidelity_pct, outcome
