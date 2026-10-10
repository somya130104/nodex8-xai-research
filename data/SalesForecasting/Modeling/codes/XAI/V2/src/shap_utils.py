"""
SHAP utilities for SalesForecasting XAI pipeline.

Supports three model families:
  Linear  — exact analytical SHAP (coef * (x - E[x])), FE_LSDV / statsmodels
  Tree    — exact TreeExplainer SHAP, XGBoost / LightGBM / sklearn trees
  Neural  — GradientExplainer SHAP with flat-sequence approximation, LSTM / PyTorch

Main public API
---------------
load_model_artifacts(run)           -> (model_obj, features, aux_dict)
compute_shap(df, features, model_obj, model_family, aux, background_df)
                                    -> (shap_array, effective_feature_names)
aggregate_shap(shap_arr, eff_features, df, run, group_vars, min_samples=5)
                                    -> long DataFrame ready to concatenate
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

VALID_FAMILIES = {"Linear", "Tree", "Neural"}


def _get_torch():
    """Lazy-load torch — only needed for Neural family."""
    import torch
    import torch.nn as nn
    return torch, nn


def _load_features_no_torch(pkl_path: str) -> list:
    """
    Extract seq_features from an LSTM pkl when torch is not installed.

    Uses a custom Unpickler that maps every torch.* class to a harmless
    dummy factory so the rest of the dict (plain Python objects) can be
    read without raising ModuleNotFoundError.
    """
    def _dummy(*args, **kwargs):
        return None

    class _SafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("torch"):
                return _dummy
            return super().find_class(module, name)

    with open(pkl_path, "rb") as f:
        pkg = _SafeUnpickler(f).load()
    return list(pkg.get("seq_features", []))


# ── Artifact loader ────────────────────────────────────────────────────────────
def load_model_artifacts(run: dict) -> tuple:
    """
    Load model pickle and return (model_obj, features, aux_dict).

    model_obj  — the object needed for SHAP computation (None for Neural when
                 torch is unavailable — features-only mode for surrogate notebooks)
    features   — list of feature names this model uses (derived from pkl)
    aux_dict   — family-specific extras (coefs, scaler, label_encoder, etc.)
    """
    family = run["model_family"]
    if family not in VALID_FAMILIES:
        raise ValueError(f"model_family must be one of {VALID_FAMILIES}, got {family!r}")

    # Neural pkl contains torch tensors — try full load; fall back to feature-only
    if family == "Neural":
        try:
            torch, nn = _get_torch()
        except ModuleNotFoundError:
            features = _load_features_no_torch(run["model_pkl_path"])
            if not features:
                raise RuntimeError(
                    f"Could not extract seq_features from {run['model_pkl_path']} "
                    f"(torch not installed and fallback found no features)."
                )
            print(f"  [Neural] torch not available — loaded {len(features)} features only (no model object)")
            return None, features, {}

    if family == "Linear":
        # sm.load() handles patsy EvalEnvironment correctly in script context;
        # pickle.load() fails with AttributeError on f_locals outside a notebook frame.
        import statsmodels.api as _sm
        result = _sm.load(run["model_pkl_path"])
        features = [p for p in result.params.index
                    if p != "Intercept" and not p.startswith("C(")]
        return result, features, {"coefs": result.params}

    with open(run["model_pkl_path"], "rb") as f:
        pkg = pickle.load(f)

    if family == "Tree":
        # XGBoost: {"model": booster, "label_encoder": le, "features": [...]}
        return pkg["model"], list(pkg["features"]), {
            "label_encoder": pkg.get("label_encoder"),
        }

    elif family == "Neural":
        # torch and nn already loaded above (or we returned early in feature-only mode)
        device = torch.device("cpu")

        class _SalesLSTM(nn.Module):
            def __init__(self, n_features, n_segments, embed_dim=4,
                         hidden_size=64, num_layers=2, dropout=0.2):
                super().__init__()
                self.seg_embed = nn.Embedding(n_segments, embed_dim)
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size + embed_dim, 32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, 1),
                )

            def forward(self, x_seq, x_seg):
                _, (h_n, _) = self.lstm(x_seq)
                h = h_n[-1]
                seg = self.seg_embed(x_seg)
                return self.fc(torch.cat([h, seg], 1)).squeeze(1)

        # LSTM: dict with model_state, model_cfg, label_encoder, scaler, seq_features, seq_len
        cfg = pkg["model_cfg"]
        model = _SalesLSTM(
            n_features=cfg["n_features"],
            n_segments=cfg["n_segments"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        ).to(device)
        model.load_state_dict(pkg["model_state"])
        model.eval()
        return model, list(pkg["seq_features"]), {
            "scaler": pkg["scaler"],
            "label_encoder": pkg["label_encoder"],
            "seq_len": pkg["seq_len"],
            "n_segments": cfg["n_segments"],
        }


# ── Per-family SHAP computers ──────────────────────────────────────────────────
def _shap_linear(df: pd.DataFrame, features: list, aux: dict,
                 background_df: pd.DataFrame) -> tuple:
    """
    Exact linear SHAP: contribution_j = coef_j * (x_j - E_DEV[x_j]).

    Returns (shap_array, eff_features) where eff_features adds 'Segment_Effect'
    to capture the combined contribution of all C(Segment) dummies.
    """
    coefs = aux["coefs"]
    X = df[features].fillna(0).values
    E = background_df[features].fillna(0).mean().values
    coef_vals = np.array([coefs.get(f, 0.0) for f in features])
    shap_numeric = (X - E) * coef_vals  # (n, n_features)

    # Aggregate all segment dummy contributions into one 'Segment_Effect' column
    seg_params = {k: v for k, v in coefs.items() if k.startswith("C(")}
    seg_effect = np.zeros(len(df))
    if "Segment" in df.columns and seg_params:
        seg_means = {}
        if "Segment" in background_df.columns:
            for key in seg_params:
                cat = key.split("[T.")[-1].rstrip("]")
                seg_means[key] = (background_df["Segment"] == cat).mean()
        for key, coef in seg_params.items():
            cat = key.split("[T.")[-1].rstrip("]")
            x_dummy = (df["Segment"] == cat).astype(float).values
            e_dummy = seg_means.get(key, 0.0)
            seg_effect += coef * (x_dummy - e_dummy)

    arr = np.column_stack([shap_numeric, seg_effect])
    eff_features = features + ["Segment_Effect"]
    return arr, eff_features


def _shap_tree(df: pd.DataFrame, features: list, model_obj,
               background_df: pd.DataFrame,
               model_type: str = "Regression") -> tuple:
    """
    SHAP for tree-based or sklearn-compatible models.

    Primary path  — shap.TreeExplainer (exact, fast): XGBoost, RF, LightGBM.
    Fallback path — shap.KernelExplainer (approximate): any sklearn predict_proba
                    model that fails TreeExplainer (e.g. Autoencoder wrapper).
                    Capped at MAX_KERNEL_ROWS rows; remaining rows are zero-filled.

    For Classification, takes class-1 SHAP values.
    """
    MAX_KERNEL_ROWS = 500
    X = df[features].fillna(0)

    try:
        explainer = shap.TreeExplainer(model_obj)
        vals = explainer.shap_values(X, check_additivity=False)
        # RF binary: list [class0_arr, class1_arr] or 3-D (n, n_feat, n_class)
        if isinstance(vals, list):
            vals = vals[1] if model_type == "Classification" and len(vals) > 1 else vals[0]
        elif isinstance(vals, np.ndarray) and vals.ndim == 3:
            vals = vals[:, :, 1] if model_type == "Classification" else vals[:, :, 0]
        return vals.astype(np.float32), features

    except Exception:
        # KernelExplainer fallback — works for any sklearn-compatible predict_proba wrapper
        bg = background_df[features].fillna(0).sample(
            min(50, len(background_df)), random_state=42)

        def _pf(x_arr):
            proba = model_obj.predict_proba(pd.DataFrame(x_arr, columns=features))
            return proba[:, 1] if model_type == "Classification" else proba[:, 0]

        explainer_k = shap.KernelExplainer(_pf, bg.values)
        X_cap  = X if len(X) <= MAX_KERNEL_ROWS else X.sample(MAX_KERNEL_ROWS, random_state=42)
        sv_cap = explainer_k.shap_values(X_cap.values, nsamples=50, silent=True)
        if isinstance(sv_cap, list):
            sv_cap = sv_cap[0]

        vals = np.zeros((len(X), len(features)), dtype=np.float32)
        vals[X_cap.index] = sv_cap
        return vals, features


def _shap_neural(df: pd.DataFrame, features: list, model_obj,
                 aux: dict, background_df: pd.DataFrame,
                 n_bg: int = 50, batch_size: int = 256) -> tuple:
    """
    GradientExplainer SHAP for LSTM using flat-sequence approximation.

    Each row's feature vector is tiled across seq_len timesteps, making the
    LSTM effectively a function of instantaneous features. A fixed average
    segment index is used for the background pass so the GradientExplainer
    receives a single-tensor input it can differentiate through.
    """
    torch, nn = _get_torch()
    device = torch.device("cpu")

    scaler = aux["scaler"]
    le = aux["label_encoder"]
    seq_len = aux["seq_len"]
    n_segments = aux["n_segments"]

    def _encode_segment(sub_df: pd.DataFrame) -> np.ndarray:
        segs = sub_df["Segment"].values if "Segment" in sub_df.columns else np.array([""] * len(sub_df))
        try:
            return le.transform(segs).astype(np.int64)
        except Exception:
            return np.zeros(len(sub_df), dtype=np.int64)

    # Background subset from DEV
    bg_df = background_df.sample(min(n_bg, len(background_df)), random_state=42)
    bg_scaled = scaler.transform(bg_df[features].fillna(0).values).astype(np.float32)
    bg_segs = _encode_segment(bg_df)
    avg_seg = int(np.clip(bg_segs.mean().round(), 0, n_segments - 1))
    fixed_seg = torch.tensor([avg_seg], dtype=torch.long).to(device)
    bg_2d = torch.tensor(bg_scaled, dtype=torch.float32).to(device)

    # Wrapper: takes (batch, n_features) 2-D tensor, tiles into sequence, uses fixed segment
    class _FlatSeqWrapper(nn.Module):
        def __init__(self, core, seg, sl):
            super().__init__()
            self.core = core
            self.seg = seg
            self.sl = sl

        def forward(self, x_2d):
            bs = x_2d.shape[0]
            x_seq = x_2d.unsqueeze(1).expand(bs, self.sl, -1)
            seg = self.seg.expand(bs)
            out = self.core(x_seq, seg)
            # GradientExplainer requires 2-D output (batch, n_outputs)
            return out.unsqueeze(-1) if out.dim() == 1 else out

    wrapper = _FlatSeqWrapper(model_obj, fixed_seg, seq_len).to(device)
    wrapper.eval()

    explainer = shap.GradientExplainer(wrapper, bg_2d)

    all_scaled = scaler.transform(df[features].fillna(0).values).astype(np.float32)
    all_t = torch.tensor(all_scaled, dtype=torch.float32).to(device)
    parts = []
    for i in range(0, len(all_t), batch_size):
        sv = explainer.shap_values(all_t[i : i + batch_size])
        if isinstance(sv, list):
            sv = sv[0]
        parts.append(sv if isinstance(sv, np.ndarray) else sv.detach().cpu().numpy())

    arr = np.vstack(parts).astype(np.float32)
    if arr.ndim == 3:          # GradientExplainer adds output dim → squeeze to (n, n_features)
        arr = arr.squeeze(-1)
    return arr, features


# ── Unified entry point ────────────────────────────────────────────────────────
def compute_shap(
    df: pd.DataFrame,
    features: list,
    model_obj,
    model_family: str,
    aux: dict,
    background_df: pd.DataFrame,
    model_type: str = "Regression",
) -> tuple:
    """
    Route to the appropriate SHAP function and return (shap_array, eff_features).

    shap_array   : np.ndarray of shape (n_rows, n_eff_features)
    eff_features : feature name list matching shap_array columns
    model_type   : "Regression" or "Classification" — controls which class SHAP is returned
    """
    if model_family == "Linear":
        return _shap_linear(df, features, aux, background_df)
    elif model_family == "Tree":
        return _shap_tree(df, features, model_obj, background_df, model_type=model_type)
    elif model_family == "Neural":
        return _shap_neural(df, features, model_obj, aux, background_df)
    else:
        raise ValueError(f"Unknown model_family: {model_family!r}")


# ── Aggregator ─────────────────────────────────────────────────────────────────
def aggregate_shap(
    shap_arr: np.ndarray,
    eff_features: list,
    df: pd.DataFrame,
    run: dict,
    group_vars: list,
    min_samples: int = 5,
) -> pd.DataFrame:
    """
    Aggregate |SHAP| at Global, Decile, and per-group_var level.

    Output schema (long format, mirrors model_performance.csv):
        Dataset, Model, Model_Domain, Model_Type,
        Group_Variable, Group_Value, Samples, Feature, Mean_Abs_SHAP
    """
    meta = {
        "Dataset": run["dataset"],
        "Model": run["model"],
        "Model_Domain": run["model_domain"],
        "Model_Type": run["model_type"],
    }
    rows = []

    def _emit(gvar: str, gval: str, mask: np.ndarray):
        n = int(mask.sum())
        if n < min_samples:
            return
        sub = np.abs(shap_arr[mask])
        for j, feat in enumerate(eff_features):
            rows.append(
                {
                    **meta,
                    "Group_Variable": gvar,
                    "Group_Value": gval,
                    "Samples": n,
                    "Feature": feat,
                    "Mean_Abs_SHAP": round(float(sub[:, j].mean()), 6),
                }
            )

    n = len(df)
    df = df.reset_index(drop=True)

    # ── Global ─────────────────────────────────────────────────────────────────
    _emit("Global", "Global", np.ones(n, dtype=bool))

    # ── Decile — rank by score_col (probability for Classification,
    #             same as pred_col for Regression) ───────────────────────────
    score_col  = run.get("score_col") or run["target_col"]
    if "Decile" not in df.columns:
        df = df.copy()
        df["Decile"] = pd.qcut(
            df[score_col].rank(method="first"), 10,
            labels=[f"D{i}" for i in range(1, 11)],
        )
    for dval, didx in df.groupby("Decile").groups.items():
        mask = np.zeros(n, dtype=bool)
        mask[didx] = True
        _emit("Decile", str(dval), mask)

    # ── Per group_var (e.g. Segment) ──────────────────────────────────────────
    for gvar in group_vars:
        if gvar not in df.columns:
            continue
        for gval, gidx in df.groupby(gvar).groups.items():
            mask = np.zeros(n, dtype=bool)
            mask[gidx] = True
            _emit(gvar, str(gval), mask)

    return pd.DataFrame(rows)
