"""
Baseline 1 of the Tri-Explainer: Adapted SHAP (gradient-based).

Standard KernelSHAP treats a model as an opaque oracle and needs thousands
of model evaluations per explained instance. Gradient-based members of the
SHAP family use the network's own gradients instead, so an explanation costs
a handful of forward/backward passes.

What is being explained ("the scalar"):
  An autoencoder produces no probability — only a reconstruction. The
  quantity attributed here is a SCALAR the caller supplies as a torch module
  mapping a batch (B, ...) -> (B,): for the autoencoder family that is the
  reconstruction error L(x) = sum_j (x_j - x_hat_j)^2 (see
  ReconstructionError below); for TabNet it is the class log-odds; for the
  LSTM it is the predicted demand. Attributions answer "which input features
  account for this scalar being what it is, relative to a reference input".

Methods (via captum):
  integrated_gradients      exact path integral from a reference to x;
                             satisfies completeness: sum_j attr_j = f(x) -
                             f(reference). The completeness residual is
                             returned so it can be checked, not assumed.
  ig_multi_baseline         IG averaged over a SET of references (e.g. k-means
                             centres) — an "expected gradients" style estimate.
  gradient_shap             GradientSHAP: expected gradients over a reference
                             DISTRIBUTION with optional input noise.

Deliberately NOT included: DeepLIFT. captum's DeepLIFT rules cover
nn.Module nonlinearities only; the reconstruction-error scalar contains a
functional squaring ((x - x_hat)**2) that DeepLIFT would treat linearly via
gradients, silently producing incorrect attributions. IG and GradientSHAP
are exact for any differentiable scalar, so those are used.
"""
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, GradientShap


class ReconstructionError(nn.Module):
    """
    Scalar-valued wrapper: reconstruction error of an autoencoder-family
    model, in the preprocessed space, as a torch module (B, p) -> (B,).
    kind='ae'  : x_hat = decode(encode(x))
    kind='vae' : x_hat = decode(mu(x))   (deterministic; no sampling)
    """
    def __init__(self, model, kind: str):
        super().__init__()
        assert kind in ("ae", "vae")
        self.model, self.kind = model, kind

    def forward(self, x):
        if self.kind == "ae":
            x_hat = self.model.decode(self.model.encode(x))
        else:
            mu, _ = self.model.encode(x)
            x_hat = self.model.decode(mu)
        return ((x - x_hat) ** 2).sum(dim=1)


def _as_tensor(a, like=None):
    t = torch.as_tensor(np.asarray(a), dtype=torch.float32)
    return t


class AdaptedSHAP:
    def __init__(self, scalar_fn: nn.Module):
        self.f = scalar_fn.eval()
        for p in self.f.parameters():
            p.requires_grad_(False)

    def integrated_gradients(self, X, baselines, n_steps: int = 50):
        """
        X: (n, ...) array/tensor. baselines: (1, ...) broadcast to all
        instances, or (n, ...) one reference per instance.
        Returns (attributions (n, ...) np.ndarray, completeness_delta (n,) np.ndarray)
        where completeness_delta = sum(attr) - (f(x) - f(reference)).
        """
        X = _as_tensor(X)
        B = _as_tensor(baselines)
        ig = IntegratedGradients(self.f)
        attr, delta = ig.attribute(X, baselines=B, n_steps=n_steps,
                                   return_convergence_delta=True)
        return attr.detach().numpy(), delta.detach().numpy()

    def ig_multi_baseline(self, X, baseline_set, n_steps: int = 50):
        """Average IG over each reference in baseline_set (k, ...)."""
        X = _as_tensor(X)
        S = _as_tensor(baseline_set)
        total = None
        deltas = []
        for k in range(len(S)):
            a, d = self.integrated_gradients(X, S[k:k + 1], n_steps=n_steps)
            total = a if total is None else total + a
            deltas.append(d)
        return total / len(S), np.mean(deltas, axis=0)

    def gradient_shap(self, X, baseline_dist, n_samples: int = 50,
                      stdevs: float = 0.0, seed: int = 0):
        """GradientSHAP against a reference distribution (m, ...) of rows."""
        X = _as_tensor(X)
        D = _as_tensor(baseline_dist)
        gs = GradientShap(self.f)
        # captum draws the interpolation coefficients from NUMPY's global RNG (and the
        # noise from torch's), so BOTH must be seeded for a reproducible result. The
        # caller's numpy RNG state is restored afterwards to avoid side effects.
        np_state = np.random.get_state()
        try:
            np.random.seed(seed)
            torch.manual_seed(seed)
            attr = gs.attribute(X, baselines=D, n_samples=n_samples, stdevs=stdevs)
        finally:
            np.random.set_state(np_state)
        return attr.detach().numpy()
