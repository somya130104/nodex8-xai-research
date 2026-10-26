"""
Shared preprocessing for TradeFraud's autoencoder-family models (Phase 2.1,
see ARCHITECTURE.md Section 7 item 9 / ROADMAP.md Phase 2.1).

Why this exists: the original autoencoder fed raw standardized values of
heavily skewed financial amount columns straight into the network. A
handful of large-valued columns then dominated the mean-squared
reconstruction error, drowning out every other feature — which is very
likely why its AUC sat at ~0.53 (barely better than chance). 221 of 271
features were found to have |skew| > 2 on normal training rows.

The fix: a SIGNED log transform, `sign(x) * log1p(|x|)`, applied only to
columns that are actually skewed (not to everything — most calendar/ratio
features are already well-behaved and don't need it). A signed version is
needed (rather than plain log1p) because about 40 of the skewed columns
have a handful of small negative values — likely data corrections — that
plain log1p cannot handle (undefined below -1).

This module is a single, picklable, fit-once-reuse-everywhere object so
that:
  - the exact same transform is applied identically in training and in
    every evaluation / explanation script (it is saved to model_config.json
    and preprocessor.pkl so that is not left to chance), and
  - Phase 3's explainers (Adapted SHAP, Manifold-LIME, LRD) can load the
    SAME preprocessing and know exactly what the network's input space
    looks like.
"""
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler


def signed_log1p(x):
    """sign(x) * log1p(|x|) — behaves like log1p for x >= 0, and handles
    negative values (which plain log1p cannot, below -1) symmetrically."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def inverse_signed_log1p(x):
    """Exact inverse of signed_log1p — recovers the original value."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * (np.expm1(np.abs(x)))


class AutoencoderPreprocessor:
    """
    Fit once on NORMAL training rows only (never disputes — the autoencoder
    family is trained exclusively on normal rows, and so is this). Produces
    a clipped, scaled array ready for the network.

    Pipeline: signed_log1p(skewed columns) -> StandardScaler -> clip to
    [-clip_value, clip_value].
    """

    def __init__(self, skew_threshold: float = 2.0, clip_value: float = 5.0):
        self.skew_threshold = skew_threshold
        self.clip_value = clip_value
        self.feat_cols = None
        self.log_cols = None
        self.scaler = None

    def fit(self, X_normal: pd.DataFrame):
        self.feat_cols = list(X_normal.columns)
        skewness = X_normal.apply(lambda c: skew(c.values.astype(np.float64)))
        self.log_cols = [c for c in self.feat_cols if abs(skewness[c]) > self.skew_threshold]

        X_logged = X_normal.copy()
        for c in self.log_cols:
            X_logged[c] = signed_log1p(X_logged[c].values)

        self.scaler = StandardScaler()
        self.scaler.fit(X_logged[self.feat_cols].values)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("AutoencoderPreprocessor.fit() must be called before transform().")
        X = X[self.feat_cols].copy()
        for c in self.log_cols:
            X[c] = signed_log1p(X[c].values)
        X_scaled = self.scaler.transform(X.values).astype(np.float32)
        return np.clip(X_scaled, -self.clip_value, self.clip_value)

    def summary(self) -> dict:
        return {
            "skew_threshold": self.skew_threshold,
            "clip_value": self.clip_value,
            "n_features": len(self.feat_cols),
            "n_log_transformed": len(self.log_cols),
            "log_transformed_columns": self.log_cols,
        }
