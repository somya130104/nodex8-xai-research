"""
Phase 3 — the Tri-Explainer framework (see ROADMAP.md Phase 3 and
ARCHITECTURE.md Section 5):

  adapted_shap   Baseline 1: gradient-based SHAP-family attribution
                 (Integrated Gradients / GradientSHAP via captum) on a scalar
                 "thing to explain" such as reconstruction error.
  manifold_lime  Baseline 2: LIME whose perturbations follow the data's own
                 covariance structure instead of independent per-feature noise.
  lrd            The proprietary method: Latent Reconstruction Decomposition,
                 for autoencoder-family models.
  baselines      The four background/baseline references used by Adapted SHAP.
  validation     Faithfulness (feature-replacement) and agreement utilities.
"""
