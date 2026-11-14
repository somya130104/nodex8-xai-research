"""
Integer snapping for Manifold-LIME perturbations (ROADMAP.md Phase 3.3, step 4:
"round any genuinely integer-valued columns").

A Gaussian perturbation of a count or a binary flag produces values such as
2.7 trades or a promotion flag of 0.37 — not real clients or days. These
helpers build the `snap` callable ManifoldLIME accepts: it maps perturbed
points back to RAW units, rounds the integer-valued columns, and maps forward
again, so the model is only ever queried at plausible values.
"""
import numpy as np
import pandas as pd


def integer_columns(X_raw) -> np.ndarray:
    """Boolean mask of columns whose every value (in the normal population) is a whole number."""
    X = np.asarray(X_raw, dtype=np.float64)
    return np.all(np.mod(X, 1) == 0, axis=0)


def snap_via_preprocessor(pre, int_mask):
    """For TradeFraud: raw = pre.inverse_transform(Z); round; pre.transform(raw)."""
    int_mask = np.asarray(int_mask, dtype=bool)

    def snap(Z):
        raw = pre.inverse_transform(Z)
        raw[:, int_mask] = np.round(raw[:, int_mask])
        return pre.transform(pd.DataFrame(raw, columns=pre.feat_cols)).astype(np.float64)
    return snap


def snap_via_standard_scaler(scaler, int_mask, tile: int = 1):
    """
    For standardised-only spaces (DemandForecast): raw = z*scale + mean.
    tile > 1 handles flattened sequences, where the same per-timestep feature
    columns repeat `tile` times along the feature axis.
    """
    mask = np.tile(np.asarray(int_mask, dtype=bool), tile)
    mean = np.tile(scaler.mean_, tile)
    scale = np.tile(scaler.scale_, tile)

    def snap(Z):
        Z = np.asarray(Z, dtype=np.float64).copy()
        raw = Z * scale + mean
        raw[:, mask] = np.round(raw[:, mask])
        return (raw - mean) / scale
    return snap


def _stochastic_round(raw, mask, rng):
    """
    Round the masked columns to a neighbouring integer with probability equal to the
    fractional part (2.3 -> 3 with p=0.3, else 2). Unlike np.round this keeps the value
    integer yet lets a count of 0 become 1 some of the time, so Manifold-LIME can see
    low-count features (ARCHITECTURE.md, Phase 4 finding 6). The expectation of the
    rounded value equals the unrounded one.
    """
    v = raw[:, mask]
    lo = np.floor(v + 1e-9)                     # guards 2.9999999999 / -1e-12 from numerical noise
    frac = np.clip(v - lo, 0.0, 1.0)
    raw[:, mask] = lo + (rng.random_sample(v.shape) < frac)
    return raw


def snap_stochastic_via_preprocessor(pre, int_mask):
    """Stochastic-rounding counterpart of snap_via_preprocessor; callable is snap(Z, rng)."""
    int_mask = np.asarray(int_mask, dtype=bool)

    def snap(Z, rng):
        raw = _stochastic_round(pre.inverse_transform(Z), int_mask, rng)
        return pre.transform(pd.DataFrame(raw, columns=pre.feat_cols)).astype(np.float64)
    return snap


def snap_stochastic_via_standard_scaler(scaler, int_mask, tile: int = 1):
    """Stochastic-rounding counterpart of snap_via_standard_scaler; callable is snap(Z, rng)."""
    mask = np.tile(np.asarray(int_mask, dtype=bool), tile)
    mean = np.tile(scaler.mean_, tile)
    scale = np.tile(scaler.scale_, tile)

    def snap(Z, rng):
        raw = _stochastic_round(np.asarray(Z, dtype=np.float64) * scale + mean, mask, rng)
        return (raw - mean) / scale
    return snap
