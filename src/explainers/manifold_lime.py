"""
Baseline 2 of the Tri-Explainer: Manifold-LIME.

The problem with standard LIME (proposal, Section 1, failure mode 2): it
explains one prediction by generating many slightly altered copies of the
instance, watching how the model responds, and fitting a simple linear model
to that local behaviour. The copies are made by jittering every feature
INDEPENDENTLY. Financial features are strongly correlated (a client's 30-day
trade volume cannot be far above their 365-day volume), so independent
jitter produces "clients" that could not exist, and the explanation is built
from the model's behaviour on impossible inputs.

Manifold-LIME changes only how the copies are generated and weighted:

  1. Estimate the covariance Sigma of the NORMAL population with
     Ledoit-Wolf shrinkage (a raw 271x271 covariance of heavily correlated
     features is near-singular; shrinkage makes it well-conditioned).
  2. Take its Cholesky factor L (Sigma = L L^T).
  3. Perturb AROUND THE INSTANCE BEING EXPLAINED:  z = x + s * L @ eps,
     eps ~ N(0, I). Correlated features then move together. Perturbing
     around x — not around the population mean — matters: a fraud case is
     itself an outlier, and centring on the mean (or discarding copies that
     lie "too far from the mean") would throw away exactly the neighbourhood
     that needs explaining.
  4. Clip each feature to the range seen in the normal population — widened,
     per instance, to include x's own value, so the point being explained is
     never moved (an out-of-time row with year=2013 when the reference ends in
     2012 would otherwise have every copy dragged to 2012, the kernel weights
     underflow to zero, and the explanation is of a different point) — then
     SNAP genuinely integer-valued columns (counts, binary flags, calendar
     fields) to whole numbers via the caller-supplied `snap` function —
     "promotion flag = 0.37" is not a real day. 96 of TradeFraud's 271 and
     31 of DemandForecast's 51 columns are integer-valued.
  5. Weight each copy by closeness to x measured with the MAHALANOBIS
     distance under Sigma, so copies that stay on-manifold count more.
  6. Fit a weighted ridge regression; report coefficient_j * (x_j - ref_j),
     a signed, SHAP-comparable contribution (ref = the neutral reference
     point, the origin of the preprocessed space by default).

Modes (so each realism constraint can be separated in an ablation):
  'manifold'          covariance-shaped noise + range clip + integer snapping  (the method)
  'manifold_noround'  covariance-shaped noise + range clip                    (ablation)
  'manifold_stochround'  as 'manifold' but integer columns are rounded STOCHASTICALLY
                      (Phase 5 candidate fix for the low-count blind spot; needs snap_stochastic)
  'independent'       vanilla LIME-style: independent noise + range clip       (comparison arm)

For a controlled comparison the SAME class also produces "vanilla" LIME-style
perturbations (mode='independent'): identical per-feature marginal spread
(same standard deviations), but independent across features, Euclidean
kernel with LIME's default width 0.75*sqrt(p). The only differences between
the two arms are the dependence structure and the kernel metric — exactly
what the proposal says Manifold-LIME changes.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


@dataclass
class LimeResult:
    attribution: np.ndarray      # (p,) signed contribution coef_j * (x_j - ref_j)
    coef: np.ndarray             # (p,) local linear coefficients (per unit of feature)
    r2: float                    # weighted R^2 of the local surrogate (fidelity)
    mean_weight: float
    frac_clipped: float          # share of perturbed entries that hit the realistic range limits


class ManifoldLIME:
    def __init__(self, black_box, U_reference, scale: float = 0.3, n_samples: int = 2000,
                 ridge_alpha: float = 1.0, kernel_width_factor: float = 1.0,
                 reference_point=None, snap=None, snap_stochastic=None):
        """
        black_box: callable (n, p) float array -> (n,) array — the scalar being
                   explained, evaluated in the same coordinate system as U_reference.
        U_reference: (N, p) NORMAL-population rows in that coordinate system.
        snap: optional callable (n, p) -> (n, p) that rounds the genuinely
              integer-valued columns (in RAW units) and returns coordinates in
              the same system as U_reference. Applied in mode 'manifold' only.
        """
        U = np.asarray(U_reference, dtype=np.float64)
        self.f = black_box
        self.p = U.shape[1]
        self.scale = scale
        self.n_samples = n_samples
        self.ridge_alpha = ridge_alpha
        self.kernel_width_factor = kernel_width_factor
        self.snap = snap
        self.snap_stochastic = snap_stochastic      # callable (Z, rng) -> Z, for mode 'manifold_stochround'

        lw = LedoitWolf().fit(U)
        self.shrinkage = float(lw.shrinkage_)
        self.cov = lw.covariance_
        self.prec = lw.precision_
        self.L = np.linalg.cholesky(self.cov)
        self.std = np.sqrt(np.diag(self.cov))
        self.lo, self.hi = U.min(axis=0), U.max(axis=0)
        self.ref = np.zeros(self.p) if reference_point is None else np.asarray(reference_point, float).ravel()

    # -- perturbation ---------------------------------------------------
    def perturb(self, x, mode: str, rng: np.random.RandomState, scale=None):
        s = self.scale if scale is None else scale
        x = np.asarray(x, dtype=np.float64).ravel()
        eps = rng.standard_normal((self.n_samples, self.p))
        if mode in ("manifold", "manifold_noround", "manifold_stochround"):
            delta = s * (eps @ self.L.T)
        elif mode == "independent":
            delta = s * eps * self.std
        else:
            raise ValueError("mode must be 'manifold', 'manifold_noround', 'manifold_stochround' or 'independent'")
        raw = x + delta
        Z = np.clip(raw, np.minimum(self.lo, x), np.maximum(self.hi, x))   # never clip x itself
        frac_clipped = float(np.mean(Z != raw))
        if mode == "manifold" and self.snap is not None:
            Z = np.asarray(self.snap(Z), dtype=np.float64)
        elif mode == "manifold_stochround":
            if self.snap_stochastic is None:
                raise ValueError("mode 'manifold_stochround' needs snap_stochastic")
            Z = np.asarray(self.snap_stochastic(Z, rng), dtype=np.float64)
        return Z, frac_clipped

    def weights(self, x, Z, mode: str, scale=None):
        s = self.scale if scale is None else scale
        d = Z - np.asarray(x, dtype=np.float64).ravel()
        if mode in ("manifold", "manifold_noround", "manifold_stochround"):
            d2 = ((d @ self.prec) * d).sum(axis=1)      # BLAS matmul (a 3-operand einsum is an unoptimised O(n p^2) loop)
            width = self.kernel_width_factor * s * np.sqrt(self.p)
        else:
            d2 = ((d / self.std) ** 2).sum(axis=1)
            width = 0.75 * np.sqrt(self.p)          # LIME's default kernel width
        return np.sqrt(np.exp(-d2 / width ** 2))    # LIME's kernel form

    # -- explanation ----------------------------------------------------
    def explain(self, x, mode: str = "manifold", seed: int = 0, scale=None, ridge_alpha=None) -> LimeResult:
        """scale / ridge_alpha override the constructor values for this call only (used by sensitivity sweeps)."""
        s = self.scale if scale is None else scale
        alpha = self.ridge_alpha if ridge_alpha is None else ridge_alpha
        x = np.asarray(x, dtype=np.float64).ravel()
        rng = np.random.RandomState(seed)
        Z, frac_clipped = self.perturb(x, mode, rng, scale=s)
        w = self.weights(x, Z, mode, scale=s)
        if not np.any(w > 0):
            raise ValueError("all kernel weights underflowed to zero — perturbations are far from x")
        y = np.asarray(self.f(Z.astype(np.float32)), dtype=np.float64).ravel()

        Xfit = (Z - x) / s                     # unit-spread inputs so alpha is meaningful
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(Xfit, y, sample_weight=w)
        coef = ridge.coef_ / s                 # back to "per unit of the feature"
        r2 = float(r2_score(y, ridge.predict(Xfit), sample_weight=w))
        return LimeResult(attribution=coef * (x - self.ref), coef=coef, r2=r2,
                          mean_weight=float(w.mean()), frac_clipped=frac_clipped)

    def explain_many(self, X, mode: str = "manifold", seed: int = 0):
        """Explain each row of X; returns (attributions (n,p), r2 (n,))."""
        X = np.asarray(X)
        attrs, r2s = [], []
        for i in range(len(X)):
            res = self.explain(X[i], mode=mode, seed=seed + i)
            attrs.append(res.attribution)
            r2s.append(res.r2)
        return np.array(attrs), np.array(r2s)


def torch_black_box(scalar_fn, batch_size: int = 4096):
    """Adapt a torch (B, p) -> (B,) module to the numpy callable ManifoldLIME expects."""
    import torch

    def f(Z):
        out = []
        with torch.no_grad():
            for i in range(0, len(Z), batch_size):
                out.append(scalar_fn(torch.as_tensor(Z[i:i + batch_size], dtype=torch.float32)).numpy())
        return np.concatenate(out)
    return f
