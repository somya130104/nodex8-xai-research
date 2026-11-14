"""
One uniform interface over every explainer variant, so every benchmark applies the SAME call:

    suite.explain(name, U, seed=0)  ->  (n, p) float64 attributions

for a model given as a torch scalar module f: (B, p) -> (B,) in a fixed coordinate system U.

Variants (a model gets only those that apply to it):
  AdaptedSHAP            Integrated Gradients averaged over 10 k-means references   (Phase 3 primary)
  AdaptedSHAP_zero       Integrated Gradients from the origin
  GradientSHAP           expected gradients over 100 real normal rows
  ManifoldLIME           covariance-shaped perturbations, range clip, deterministic integer rounding (primary)
  ManifoldLIME_stochround   ... with stochastic integer rounding (Phase 4 candidate fix)
  VanillaLIME            independent perturbations, Euclidean kernel (comparison arm)
  LRD_lam0 / LRD_lam0.5 / LRD_lam1   Latent Reconstruction Decomposition (autoencoder family only)
  NativeMask             TabNet's own attention-mask importance (TabNet only)
  KernelSHAP             standard model-agnostic SHAP, 10 k-means background (external reference)
Controls (not explainers; they show what "no skill" and "trivial skill" look like):
  Random                 a random ranking
  AbsValue               rank features by |value| in the preprocessed space (the 'largest z-score' rule)
"""
import numpy as np
import torch

from src.explainers.adapted_shap import AdaptedSHAP
from src.explainers.baselines import kmeans_baselines
from src.explainers.lrd import LRDExplainer
from src.explainers.manifold_lime import ManifoldLIME, torch_black_box

PRIMARY = ("AdaptedSHAP", "ManifoldLIME", "LRD_lam0.5", "NativeMask")
CONTROLS = ("Random", "AbsValue")
STOCHASTIC = {"GradientSHAP", "ManifoldLIME", "ManifoldLIME_stochround", "VanillaLIME", "KernelSHAP", "Random"}


class ExplainerSuite:
    def __init__(self, f_torch, U_normal, snap=None, snap_stochastic=None, lrd_model=None, lrd_kind=None,
                 native_fn=None, seed=42, ig_steps=50, kmeans_k=10, gs_baselines=100, lime_samples=2000,
                 lime_scale=0.3):
        self.f_torch = f_torch.eval()
        self.f_np = torch_black_box(f_torch)
        self.U_normal = np.asarray(U_normal, dtype=np.float32)
        self.p = self.U_normal.shape[1]
        self.seed, self.ig_steps = seed, ig_steps
        self.ash = AdaptedSHAP(f_torch)
        self.b_zero = np.zeros((1, self.p), dtype=np.float32)
        self.b_km = kmeans_baselines(self.U_normal, k=kmeans_k, seed=seed)
        rs = np.random.RandomState(seed)
        self.gs_base = self.U_normal[rs.choice(len(self.U_normal), min(gs_baselines, len(self.U_normal)), replace=False)]
        self.mlime = ManifoldLIME(self.f_np, self.U_normal, scale=lime_scale, n_samples=lime_samples,
                                  snap=snap, snap_stochastic=snap_stochastic)
        self.lrd_model, self.lrd_kind = lrd_model, lrd_kind
        self.native_fn = native_fn
        self.has_stoch = snap_stochastic is not None
        self._kernel = None

    # ------------------------------------------------------------------ catalogue
    @property
    def available(self):
        names = ["AdaptedSHAP", "AdaptedSHAP_zero", "GradientSHAP", "ManifoldLIME", "VanillaLIME"]
        if self.has_stoch:
            names.insert(4, "ManifoldLIME_stochround")
        if self.lrd_model is not None:
            names += ["LRD_lam0", "LRD_lam0.5", "LRD_lam1"]
        if self.native_fn is not None:
            names.append("NativeMask")
        return names

    @staticmethod
    def is_stochastic(name):
        return name in STOCHASTIC

    # ------------------------------------------------------------------ the one call
    def explain(self, name, U, seed=0, same_seed=False):
        """
        same_seed=False: LIME row i uses seed+i (independent draws; the normal use).
        same_seed=True : every row uses the same seed (needed by the adversarial search, where
        a changed explanation must come from the changed INPUT, not from a changed random draw).
        """
        U = np.ascontiguousarray(U, dtype=np.float32)
        n = len(U)
        if name == "AdaptedSHAP":
            return self.ash.ig_multi_baseline(U, self.b_km, self.ig_steps)[0].astype(np.float64)
        if name == "AdaptedSHAP_zero":
            return self.ash.integrated_gradients(U, self.b_zero, self.ig_steps)[0].astype(np.float64)
        if name == "GradientSHAP":
            return self.ash.gradient_shap(U, self.gs_base, n_samples=50, stdevs=0.0, seed=self.seed + seed).astype(np.float64)
        if name in ("ManifoldLIME", "ManifoldLIME_stochround", "VanillaLIME"):
            mode = {"ManifoldLIME": "manifold", "ManifoldLIME_stochround": "manifold_stochround", "VanillaLIME": "independent"}[name]
            return np.array([self.mlime.explain(U[i], mode=mode, seed=seed if same_seed else seed + i).attribution
                             for i in range(n)], dtype=np.float64)
        if name.startswith("LRD_lam"):
            lam = float(name[len("LRD_lam"):])
            return LRDExplainer(self.lrd_model, kind=self.lrd_kind, lam=lam).explain(torch.as_tensor(U)).astype(np.float64)
        if name == "NativeMask":
            return np.asarray(self.native_fn(U), dtype=np.float64)
        if name == "KernelSHAP":
            import shap
            if self._kernel is None:
                self._kernel = shap.KernelExplainer(self.f_np, self.b_km)
            np_state = np.random.get_state()
            try:
                np.random.seed(self.seed + seed)
                sv = self._kernel.shap_values(U, silent=True)
            finally:
                np.random.set_state(np_state)
            return np.asarray(sv[0] if isinstance(sv, list) else sv, dtype=np.float64).reshape(n, -1)
        if name == "Random":
            return np.random.RandomState(self.seed + seed).normal(size=U.shape)
        if name == "AbsValue":
            return np.abs(U).astype(np.float64)
        raise KeyError(name)
