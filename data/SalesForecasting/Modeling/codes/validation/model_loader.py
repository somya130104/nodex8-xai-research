"""
Generic model-artifact loader — the one piece of engine.py that used to
reach outside the validation package (into V3/src/shap_utils.py) to load a
model pkl. That made the package impossible to copy into another project
without also dragging along FraudPrediction-specific SHAP/XAI code and
custom neural architectures.

This loader only handles what's actually generic: a pickle that is either
  (a) a bare fitted estimator, or
  (b) a dict bundle {"model": estimator, "features": [...], ...extras}
which covers both FraudPrediction's Tree pkls and, e.g., a differently-named
project's `{"model":..., "label_encoder":..., "features": FEATURES}` bundle
(same convention, different project) — no project-specific classes involved.

Anything that needs a custom class definition to unpickle (a raw torch
state_dict, a bespoke nn.Module) is NOT something a generic loader can
reconstruct — that's inherent, not a gap to paper over. Those are surfaced
as `model_obj=None` with a clear reason, and the caller (engine.py) skips
them via a capability check rather than a hardcoded family name.
"""
from __future__ import annotations

import pickle


class ModelLoadError(ValueError):
    pass


def _infer_features(model_obj) -> list[str] | None:
    if hasattr(model_obj, "get_booster") and getattr(model_obj.get_booster(), "feature_names", None):
        return list(model_obj.get_booster().feature_names)
    if hasattr(model_obj, "feature_names_in_"):
        return list(model_obj.feature_names_in_)
    return None


def capability(model_obj) -> str | None:
    """'classification' if the object exposes predict_proba, 'regression' if
    it only exposes predict, else None (can't be used for re-inference)."""
    if model_obj is None:
        return None
    if hasattr(model_obj, "predict_proba"):
        return "classification"
    if hasattr(model_obj, "predict"):
        return "regression"
    return None


def load(model_pkl_path: str) -> tuple[object | None, list[str] | None, dict, str | None]:
    """Returns (model_obj, features, aux, note).

    model_obj is None (with `note` explaining why) when the pickle can't be
    used generically — the caller should skip that run rather than guess.
    """
    with open(model_pkl_path, "rb") as f:
        pkg = pickle.load(f)

    if isinstance(pkg, dict):
        model_obj = pkg.get("model") or pkg.get("estimator")
        if model_obj is None:
            return None, None, {}, (
                f"dict pickle at {model_pkl_path} has no 'model'/'estimator' key "
                f"(keys: {list(pkg.keys())}) — nothing generic to load"
            )
        features = pkg.get("features") or pkg.get("feature_names") or _infer_features(model_obj)
        if features is None:
            return None, None, {}, (
                f"could not determine a feature list for {type(model_obj).__name__} "
                f"in {model_pkl_path} — no 'features' key and none derivable from the model"
            )
        aux = {k: v for k, v in pkg.items() if k not in ("model", "estimator", "features", "feature_names")}
        if capability(model_obj) is None:
            return None, None, {}, f"{type(model_obj).__name__} exposes neither predict_proba nor predict"
        return model_obj, list(features), aux, None

    # bare object
    model_obj = pkg
    if capability(model_obj) is None:
        return None, None, {}, f"{type(model_obj).__name__} exposes neither predict_proba nor predict — not usable for re-inference"
    features = _infer_features(model_obj)
    if features is None:
        return None, None, {}, (
            f"bare {type(model_obj).__name__} at {model_pkl_path} has no detectable feature list "
            f"(no get_booster().feature_names / feature_names_in_). Re-save as "
            f"{{'model': model_obj, 'features': [...]}}."
        )
    return model_obj, features, {}, None
