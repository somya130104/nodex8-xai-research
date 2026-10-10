"""
Traceability Rail — semantic hashes for inference-time auditability.

Four public functions, each returning "sha256:<64-hex>":

  h_data(df, features)                          — feature matrix fingerprint
  h_model_version(model_obj, features)          — architecture + weights
  h_output(computed_scores, row_ids=None)       — scores array fingerprint
  h_decision(h_data, h_model_version, h_output) — master binding signature

Canonical serialisation rule (same across all functions):
  sort keys → sort rows by Client_ID → round floats to 6dp →
  compact JSON (sort_keys=True, no whitespace) → UTF-8 → SHA-256
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys

import numpy as np
import pandas as pd


def h_data(df: pd.DataFrame, features: list, id_col: str = "Client_ID") -> str:
    """
    Hash the complete input dataset — every column (identifiers, descriptives,
    and model features).  This ensures H_data detects any mutation of the data,
    including changes to segment labels, row identifiers, dates, or other
    non-feature fields, not just changes to model inputs.

    Column order is normalised (sorted alphabetically).
    Row order is normalised by `id_col` when present (default "Client_ID";
    pass e.g. id_col="Cust_ID" for a project that identifies rows differently).
    Numeric values are rounded to 6 dp; non-numeric values are cast to str.
    The `features` parameter is retained for call-site compatibility but does
    not filter which columns are hashed.
    """
    sorted_cols = sorted(df.columns.tolist())
    matrix = df[sorted_cols].copy()

    num_cols = matrix.select_dtypes(include=[np.number]).columns
    str_cols = matrix.select_dtypes(exclude=[np.number]).columns
    matrix[num_cols] = matrix[num_cols].fillna(0).round(6)
    matrix[str_cols] = matrix[str_cols].fillna("").astype(str)

    if id_col in df.columns:
        order  = df[id_col].argsort().values
        matrix = matrix.iloc[order]

    canonical = json.dumps(
        matrix.to_dict(orient="records"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def h_model_version(model_obj, features: list) -> str:
    """
    Hash model architecture, every hyperparameter, feature names,
    weight fingerprint, and library versions.

    `classes_` only exists on classifiers — regressors (e.g. XGBRegressor)
    don't have it, so it's included when present and simply omitted (not
    faked as empty) otherwise.
    """
    sorted_feats = sorted(str(f) for f in features)

    components: dict = {
        "model_class":   model_obj.__class__.__name__,
        "model_module":  model_obj.__class__.__module__,
        "hyperparams":   {k: str(v) for k, v in
                          sorted(model_obj.get_params().items())},
        "n_features":    len(features),
        "feature_names": sorted_feats,
    }
    if hasattr(model_obj, "classes_"):
        components["classes"] = sorted(str(c) for c in model_obj.classes_)

    import sklearn
    components["sklearn_version"] = sklearn.__version__
    components["python_version"]  = sys.version.split()[0]

    try:
        import xgboost
        components["xgboost_version"] = xgboost.__version__
    except ImportError:
        pass

    # Weight fingerprint — float32 for precision independence
    fi_bytes = model_obj.feature_importances_.astype(np.float32).tobytes()
    components["fi_hash"] = hashlib.sha256(fi_bytes).hexdigest()

    # Tree structure hash
    if hasattr(model_obj, "get_booster"):           # XGBoost
        dump = "\n".join(model_obj.get_booster().get_dump())
        components["tree_hash"] = hashlib.sha256(dump.encode()).hexdigest()
    else:                                            # RandomForest / sklearn
        tree_bytes = pickle.dumps(model_obj.estimators_)
        components["tree_hash"] = hashlib.sha256(tree_bytes).hexdigest()

    canonical = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def h_output(computed_scores: np.ndarray, row_ids=None) -> str:
    """
    Hash the computed probability scores array.
    If row_ids are provided the scores are sorted to canonical row order first.
    """
    scores = np.round(computed_scores.astype(float), 6)
    if row_ids is not None:
        scores = scores[np.argsort(row_ids)]
    canonical = json.dumps(scores.tolist(), separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def h_decision(h_data_str: str, h_model_str: str, h_output_str: str) -> str:
    """
    Master binding signature: SHA256(H_data_hex ‖ H_model_hex ‖ H_output_hex).
    Any change to inputs, model, or outputs produces a different H_decision.
    """
    combined = h_data_str[7:] + h_model_str[7:] + h_output_str[7:]
    return f"sha256:{hashlib.sha256(combined.encode()).hexdigest()}"


def h_row(client_id, h_model_version_str: str, computed_score: float) -> str:
    """
    Per-row fingerprint binding a single prediction to its model version.

    client_id           — the row identifier (Client_ID value or row index)
    h_model_version_str — the pre-computed H_model_version for this run
    computed_score      — the model's probability output, rounded to 6dp

    Formula: SHA256("<client_id>|<model_hex>|<score_6dp>")
    Same formula works for real-time single-row scoring — pre-compute
    H_model_version once at startup, then call h_row per prediction.
    """
    parts = f"{client_id}|{h_model_version_str[7:]}|{round(computed_score, 6)}"
    return f"sha256:{hashlib.sha256(parts.encode()).hexdigest()}"
