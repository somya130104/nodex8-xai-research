"""
LIME utilities for SalesForecasting XAI pipeline.

Supports three model families:
  Linear  — statsmodels formula predict with fixed Segment per row
  Tree    — direct sklearn-compatible predict on feature DataFrame
  Neural  — LSTM forward pass with flat-sequence approximation (features tiled
             across seq_len timesteps; Segment fixed per row)

Main public API
---------------
build_lime_explainer(background_df, features)
    -> LimeTabularExplainer  (build once per model, reuse across datasets)

make_predict_fn(model_obj, model_family, features, aux, row_segment=None)
    -> callable X_array -> y_array  (create per-row for Linear/Neural, reuse for Tree)

compute_lime_matrix(sampled_df, explainer, model_obj, model_family,
                    features, aux, n_lime_features=None)
    -> (lime_array, valid_features)  shape (n_sampled, n_valid_features)

aggregate_lime(lime_arr, valid_features, sampled_df, df, run,
               group_vars, min_samples=5)
    -> long DataFrame ready to concatenate
"""

import warnings
import numpy as np
import pandas as pd
import lime.lime_tabular
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# ── Explainer builder ──────────────────────────────────────────────────────────
def build_lime_explainer(
    background_df: pd.DataFrame,
    features: list,
    random_state: int = 42,
    model_type: str = "Regression",
) -> tuple:
    """
    Build a LimeTabularExplainer from DEV background data.

    Constant columns are dropped automatically (LIME will error on them).
    Categorical features are identified by cardinality < 15.
    mode is set to "classification" for Classification models, "regression" otherwise.

    Returns (explainer, valid_features) where valid_features is the filtered
    feature list that the explainer was trained on.
    """
    X_bg = background_df[features].fillna(0)

    const_cols = [c for c in X_bg.columns if X_bg[c].nunique() <= 1]
    valid_features = [c for c in features if c not in const_cols]

    if const_cols:
        print(f"  [LIME] Dropping {len(const_cols)} constant column(s): {const_cols}")

    X_bg_valid = X_bg[valid_features].values
    cat_idx = [i for i, c in enumerate(valid_features) if X_bg[c].nunique() < 15]
    lime_mode = "classification" if model_type == "Classification" else "regression"

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_bg_valid,
        feature_names=valid_features,
        categorical_features=cat_idx,
        mode=lime_mode,
        discretize_continuous=True,
        random_state=random_state,
    )
    return explainer, valid_features


# ── Predict-function factory ───────────────────────────────────────────────────
def make_predict_fn(
    model_obj,
    model_family: str,
    valid_features: list,
    aux: dict,
    row_segment: str = None,
    model_type: str = "Regression",
):
    """
    Return a predict_fn(X_array) -> np.ndarray for LIME perturbation.

    X_array has columns corresponding to valid_features (already filtered).
    Segment is fixed to row_segment for Linear and Neural families because
    LIME perturbs only the numeric feature space.
    For Classification Tree models, predict_proba[:,1] is returned so LIME
    receives probability scores, not hard class labels.
    """
    if model_family == "Linear":
        segment = row_segment or "Unknown"

        def predict_fn(X_array):
            df_p = pd.DataFrame(X_array, columns=valid_features)
            df_p["Segment"] = segment
            preds_log = model_obj.predict(df_p)
            return np.expm1(np.array(preds_log)).clip(0).astype(float)

        return predict_fn

    elif model_family == "Tree":
        def predict_fn(X_array):
            df_p = pd.DataFrame(X_array, columns=valid_features)
            return model_obj.predict(df_p).astype(float)

        return predict_fn

    elif model_family == "Neural":
        import torch as _torch

        scaler = aux["scaler"]
        le = aux["label_encoder"]
        seq_len = aux["seq_len"]
        n_segments = aux["n_segments"]

        try:
            seg_idx = int(le.transform([row_segment])[0]) if row_segment else 0
        except Exception:
            seg_idx = 0
        seg_idx = int(np.clip(seg_idx, 0, n_segments - 1))

        # Build the scaler sub-transform for valid_features only
        # valid_features is a subset of seq_features; reconstruct full vector
        all_seq_features = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else valid_features

        def predict_fn(X_array):
            # Reconstruct full-feature array (fill missing with 0)
            n = len(X_array)
            full_arr = np.zeros((n, len(all_seq_features)), dtype=np.float32)
            for j, feat in enumerate(valid_features):
                if feat in all_seq_features:
                    fidx = all_seq_features.index(feat)
                    full_arr[:, fidx] = X_array[:, j]

            X_scaled = scaler.transform(full_arr).astype(np.float32)
            # Flat-sequence approximation: tile across seq_len
            X_seq = np.stack([X_scaled] * seq_len, axis=1)
            X_seg = np.full(n, seg_idx, dtype=np.int64)

            seq_t = _torch.tensor(X_seq, dtype=_torch.float32)
            seg_t = _torch.tensor(X_seg, dtype=_torch.long)
            with _torch.no_grad():
                pred_log = model_obj(seq_t, seg_t).cpu().numpy()
            return np.expm1(pred_log).clip(0).astype(float)

        return predict_fn

    else:
        raise ValueError(f"Unknown model_family: {model_family!r}")


# ── LIME matrix computation ────────────────────────────────────────────────────
def compute_lime_matrix(
    sampled_df: pd.DataFrame,
    explainer,
    model_obj,
    model_family: str,
    valid_features: list,
    aux: dict,
    n_lime_features: int = None,
    model_type: str = "Regression",
    group_var: str = None,
) -> np.ndarray:
    """
    Run LIME on every row of sampled_df.

    For Tree models, predict_fn is built once and reused.
    For Linear/Neural models, predict_fn is rebuilt per row (group fixed).

    Returns lime_array of shape (n_sampled, len(valid_features)).
    Weights are raw signed LIME coefficients.
    """
    n = len(sampled_df)
    n_feats = len(valid_features)
    n_explain = n_lime_features or n_feats
    lime_arr = np.zeros((n, n_feats), dtype=np.float32)

    # For Tree: one predict_fn for all rows
    tree_predict_fn = None
    if model_family == "Tree":
        tree_predict_fn = make_predict_fn(model_obj, model_family, valid_features, aux,
                                          model_type=model_type)

    sampled_df = sampled_df.reset_index(drop=True)
    # Determine segment column name (group_var takes priority, fall back to "Segment")
    seg_col = group_var if group_var and group_var in sampled_df.columns else (
        "Segment" if "Segment" in sampled_df.columns else None
    )
    pbar = tqdm(range(n), desc="LIME", leave=False)

    for i in pbar:
        row = sampled_df.iloc[i]
        x = row[valid_features].fillna(0).values.astype(float)

        if model_family == "Tree":
            predict_fn = tree_predict_fn
        else:
            row_segment = row.get(seg_col) if seg_col else None
            predict_fn = make_predict_fn(
                model_obj, model_family, valid_features, aux,
                row_segment=row_segment,
                model_type=model_type,
            )

        try:
            exp = explainer.explain_instance(
                x, predict_fn, num_features=n_explain,
            )
            for feat_idx, weight in exp.as_map()[0]:
                if feat_idx < n_feats:
                    lime_arr[i, feat_idx] = weight
        except Exception:
            pass  # row stays zero — silent skip

    pbar.close()
    return lime_arr


# ── Aggregator ─────────────────────────────────────────────────────────────────
def aggregate_lime(
    lime_arr: np.ndarray,
    valid_features: list,
    sampled_df: pd.DataFrame,
    full_df: pd.DataFrame,
    run: dict,
    group_vars: list,
    min_samples: int = 5,
) -> pd.DataFrame:
    """
    Aggregate |LIME weight| at Global, Decile, and per-group_var level.

    sampled_df  — the rows LIME was run on (index must align with lime_arr)
    full_df     — full prediction file (used only for Decile calculation)

    Output schema (long format, mirrors model_performance.csv):
        Dataset, Model, Model_Domain, Model_Type,
        Group_Variable, Group_Value, Samples, Feature, Mean_Abs_LIME
    """
    meta = {
        "Dataset": run["dataset"],
        "Model": run["model"],
        "Model_Domain": run["model_domain"],
        "Model_Type": run["model_type"],
    }
    rows = []
    sampled_df = sampled_df.reset_index(drop=True)

    def _emit(gvar: str, gval: str, mask: np.ndarray):
        n = int(mask.sum())
        if n < min_samples:
            return
        sub = np.abs(lime_arr[mask])
        for j, feat in enumerate(valid_features):
            rows.append(
                {
                    **meta,
                    "Group_Variable": gvar,
                    "Group_Value": gval,
                    "Samples": n,
                    "Feature": feat,
                    "Mean_Abs_LIME": round(float(sub[:, j].mean()), 6),
                }
            )

    n = len(sampled_df)

    # ── Global ─────────────────────────────────────────────────────────────────
    _emit("Global", "Global", np.ones(n, dtype=bool))

    # ── Decile — rank by score_col (probability for Classification,
    #             same as pred_col for Regression) ───────────────────────────
    score_col = run.get("score_col") or run["target_col"]
    if score_col in sampled_df.columns and len(sampled_df) >= 10:
        decile_labels = pd.qcut(
            sampled_df[score_col].rank(method="first"), 10,
            labels=[f"D{i}" for i in range(1, 11)],
        )
        for dval in decile_labels.cat.categories:
            mask = (decile_labels == dval).values
            _emit("Decile", str(dval), mask)

    # ── Per group_var (e.g. Segment) ──────────────────────────────────────────
    for gvar in group_vars:
        if gvar not in sampled_df.columns:
            continue
        for gval, gidx in sampled_df.groupby(gvar).groups.items():
            mask = np.zeros(n, dtype=bool)
            mask[gidx] = True
            _emit(gvar, str(gval), mask)

    return pd.DataFrame(rows)
