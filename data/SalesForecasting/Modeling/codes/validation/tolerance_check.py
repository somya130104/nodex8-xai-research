from __future__ import annotations

import numpy as np
import pandas as pd


def check(df: pd.DataFrame, score_col: str, atol: float, rtol: float = 0.0) -> dict:
    """Passing rule: |computed - stored| <= atol + rtol*|stored| — the same
    combined absolute+relative tolerance numpy.isclose uses. rtol=0 (default)
    reproduces the original pure-absolute-tolerance behavior exactly.

    A pure absolute tolerance makes sense for a bounded score (e.g. a [0,1]
    probability); for an unbounded regression target (e.g. weekly unit
    sales ranging from 0 to 18,000) a fixed absolute tolerance is either too
    loose for small values or impossibly tight for large ones, so rtol lets
    the tolerance scale with the magnitude of the stored value.

    `score_col` itself is optional — only `abs_delta_{score_col}` is
    required. Without the raw stored values there's nothing for rtol to
    scale against, so it's simply not applied (equivalent to rtol=0).
    """
    delta  = df[f"abs_delta_{score_col}"].to_numpy(dtype=float)
    if rtol and score_col in df.columns:
        stored = df[score_col].to_numpy(dtype=float)
        tol_band = atol + rtol * np.abs(stored)
    else:
        tol_band = atol
    n_total = len(delta)
    passing = delta <= tol_band
    n_passing = int(passing.sum())
    return {
        "n_rows":         n_total,
        "n_passing":      n_passing,
        "n_failing":      n_total - n_passing,
        "max_abs_delta":  round(float(np.max(delta)),  6),
        "mean_abs_delta": round(float(np.mean(delta)), 6),
        "p95_abs_delta":  round(float(np.percentile(delta, 95)), 6),
        "p99_abs_delta":  round(float(np.percentile(delta, 99)), 6),
    }
