from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import hasher
from .config import REPORTS_DIR


def rerun_and_enrich(
    df: pd.DataFrame,
    model_obj,
    X: np.ndarray,
    run: dict,
    atol: float,
    rtol: float = 0.0,
    h_model_version_str: str | None = None,
    id_col: str = "Client_ID",
) -> tuple[pd.DataFrame, str]:
    """
    Re-runs inference and appends computed columns, saves enriched CSV.
    Returns (enriched_df, output_path).

    Classifiers (predict_proba) get a computed_{pred_col} column too, since a
    binary prediction is well-defined at the standard 0.5 cut. Regressors
    (predict only) don't — there's no generic decision cut for a continuous
    target, so only the score itself is reproduced.

    tolerance_pass uses the same combined absolute+relative rule as
    tolerance_check.check(): |delta| <= atol + rtol*|stored|.
    """
    score_col = run["score_col"]
    pred_col  = run.get("pred_col")

    is_classifier = hasattr(model_obj, "predict_proba")
    if is_classifier:
        computed_scores = model_obj.predict_proba(X)[:, 1]
    else:
        computed_scores = np.asarray(model_obj.predict(X), dtype=float)

    df = df.copy()
    df[f"computed_{score_col}"] = computed_scores.round(6)
    if is_classifier and pred_col:
        df[f"computed_{pred_col}"] = (computed_scores >= 0.5).astype(int)

    stored = df[score_col].to_numpy(dtype=float)
    delta  = np.abs(stored - computed_scores)
    df[f"abs_delta_{score_col}"] = delta.round(6)
    df["tolerance_pass"] = delta <= (atol + rtol * np.abs(stored))

    if h_model_version_str and id_col in df.columns:
        df["row_hash"] = [
            hasher.h_row(cid, h_model_version_str, s)
            for cid, s in zip(df[id_col], computed_scores)
        ]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = REPORTS_DIR / f"{run['run_id']}_validated_{ts}.csv"
    REPORTS_DIR.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)

    return df, str(out_path)
