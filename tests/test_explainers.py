"""
Tests for the Phase 3 explainers (src/explainers/*) and the supporting
preprocessing additions. Every test uses synthetic data with a KNOWN answer,
so a pass means the maths is right — not merely that the code runs.

Run from the repo root:   python3 -m pytest tests -q
"""
import sys
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, "/Users/somyabhadada/nodex8")

from src.autoencoder_models import Autoencoder, VAE
from src.autoencoder_preprocessing import AutoencoderPreprocessor
from src.explainers.adapted_shap import AdaptedSHAP, ReconstructionError
from src.explainers.baselines import kmeans_baselines, nearest_neighbor_baselines, raw_mean_baseline
from src.explainers.lrd import LRDExplainer
from src.explainers.manifold_lime import ManifoldLIME, torch_black_box
from src.explainers.snapping import integer_columns, snap_via_preprocessor, snap_via_standard_scaler
from src.explainers.validation import (
    replacement_effect, summarize_replacement, nearest_real_distance, pairwise_agreement,
)


# ---------------------------------------------------------------- helpers
class Quadratic(nn.Module):
    """f(x) = sum_j w_j x_j^2  — IG from the origin has the closed form w_j x_j^2."""
    def __init__(self, w):
        super().__init__()
        self.w = torch.tensor(w, dtype=torch.float32)

    def forward(self, x):
        return (self.w * x ** 2).sum(dim=1)


def make_ae(p=12, hidden=(8, 6, 3), seed=0):
    torch.manual_seed(seed)
    m = Autoencoder(p, list(hidden)).eval()
    # BatchNorm running stats are at their defaults after init; perturb them
    # so the eval-mode path is non-trivial.
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm1d):
            mod.running_mean.normal_(0, 0.1)
            mod.running_var.uniform_(0.5, 1.5)
    return m


def make_vae(p=12, seed=0):
    torch.manual_seed(seed)
    m = VAE(p, [8, 6], 3).eval()
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm1d):
            mod.running_mean.normal_(0, 0.1)
            mod.running_var.uniform_(0.5, 1.5)
    return m


# ----------------------------------------------------- Adapted SHAP / IG
def test_ig_matches_closed_form_and_is_complete():
    w = np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float32)
    x = np.array([[1.0, -2.0, 3.0, 0.5], [0.0, 1.0, -1.0, 2.0]], dtype=np.float32)
    attr, delta = AdaptedSHAP(Quadratic(w)).integrated_gradients(x, np.zeros((1, 4)), n_steps=200)
    np.testing.assert_allclose(attr, w * x ** 2, atol=1e-2)       # closed form
    assert np.abs(delta).max() < 1e-2                               # completeness residual


def test_ig_nonzero_baseline_is_complete():
    w = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    b = np.array([[0.5, -1.0, 2.0]], dtype=np.float32)
    f = Quadratic(w)
    attr, _ = AdaptedSHAP(f).integrated_gradients(x, b, n_steps=200)
    expected = float(f(torch.tensor(x)) - f(torch.tensor(b)))
    assert abs(attr.sum() - expected) < 1e-2


def test_ig_multi_baseline_is_mean_of_single_baselines():
    w = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    S = np.array([[0.0, 0, 0], [1.0, 1, 1], [-1.0, 0, 2]], dtype=np.float32)
    shap_ = AdaptedSHAP(Quadratic(w))
    multi, _ = shap_.ig_multi_baseline(x, S, n_steps=100)
    singles = np.mean([shap_.integrated_gradients(x, S[k:k + 1], n_steps=100)[0] for k in range(3)], axis=0)
    np.testing.assert_allclose(multi, singles, atol=1e-6)


def test_gradient_shap_runs_and_is_roughly_complete():
    w = np.array([1.0, 2.0, 0.5, 1.0], dtype=np.float32)
    f = Quadratic(w)
    x = np.array([[1.0, 2.0, -1.0, 0.5]], dtype=np.float32)
    D = np.random.RandomState(0).normal(0, 0.1, (50, 4)).astype(np.float32)
    attr = AdaptedSHAP(f).gradient_shap(x, D, n_samples=500, seed=0)
    assert attr.shape == (1, 4)
    expected = float(f(torch.tensor(x))) - float(f(torch.tensor(D)).mean())
    assert abs(attr.sum() - expected) < 0.3          # stochastic estimator: loose tolerance


def test_gradient_shap_is_reproducible_regardless_of_global_numpy_state():
    f = Quadratic(np.array([1.0, 2.0, 0.5, 1.0], dtype=np.float32))
    x = np.array([[1.0, 2.0, -1.0, 0.5]], dtype=np.float32)
    D = np.random.RandomState(0).normal(0, 0.1, (50, 4)).astype(np.float32)
    shap_ = AdaptedSHAP(f)
    np.random.seed(1)
    a = shap_.gradient_shap(x, D, n_samples=200, seed=7)
    np.random.seed(2)
    b = shap_.gradient_shap(x, D, n_samples=200, seed=7)
    np.testing.assert_allclose(a, b)
    state_before = np.random.get_state()[1].copy()
    shap_.gradient_shap(x, D, n_samples=10, seed=3)
    assert np.array_equal(state_before, np.random.get_state()[1])      # caller's RNG untouched


def test_reconstruction_error_wrapper_matches_manual():
    ae, vae = make_ae(), make_vae()
    X = torch.randn(5, 12)
    with torch.no_grad():
        manual_ae = ((X - ae(X)) ** 2).sum(1)
        mu, _ = vae.encode(X)
        manual_vae = ((X - vae.decode(mu)) ** 2).sum(1)
        np.testing.assert_allclose(ReconstructionError(ae, "ae")(X).numpy(), manual_ae.numpy(), rtol=1e-5)
        np.testing.assert_allclose(ReconstructionError(vae, "vae")(X).numpy(), manual_vae.numpy(), rtol=1e-5)


def test_ig_on_autoencoder_error_is_complete():
    ae = make_ae()
    f = ReconstructionError(ae, "ae")
    X = torch.randn(6, 12)
    attr, delta = AdaptedSHAP(f).integrated_gradients(X.numpy(), np.zeros((1, 12)), n_steps=300)
    assert np.abs(delta).max() < 5e-2


# -------------------------------------------------------------- baselines
def test_baselines_shapes_and_nn_is_actually_nearest():
    rng = np.random.RandomState(0)
    ref = rng.normal(size=(500, 6)).astype(np.float32)
    q = rng.normal(size=(7, 6)).astype(np.float32)
    km = kmeans_baselines(ref, k=4)
    assert km.shape == (4, 6)
    nn_b = nearest_neighbor_baselines(q, ref)
    assert nn_b.shape == (7, 6)
    brute = ref[np.argmin(((q[:, None, :] - ref[None]) ** 2).sum(-1), axis=1)]
    np.testing.assert_allclose(nn_b, brute, atol=1e-6)


def test_raw_mean_baseline_differs_from_origin_for_skewed_data():
    rng = np.random.RandomState(0)
    raw = pd.DataFrame({"a": rng.lognormal(0, 2, 5000), "b": rng.normal(0, 1, 5000)})
    pre = AutoencoderPreprocessor(skew_threshold=2.0).fit(raw)
    b = raw_mean_baseline(raw, pre)
    assert b.shape == (1, 2)
    assert abs(b[0, 0]) > 0.3          # the arithmetic mean of a heavy-tailed column is NOT the typical value


# ---------------------------------------------------------- preprocessing
def test_inverse_transform_roundtrips_unclipped_values():
    rng = np.random.RandomState(0)
    raw = pd.DataFrame({"a": rng.lognormal(0, 1.5, 3000),
                        "b": rng.normal(0, 1, 3000),
                        "c": rng.lognormal(0, 1.5, 3000) * rng.choice([1, -1], 3000, p=[0.97, 0.03])})
    pre = AutoencoderPreprocessor(skew_threshold=2.0, clip_value=50.0).fit(raw)   # clip loose -> invertible
    back = pre.inverse_transform(pre.transform(raw))
    np.testing.assert_allclose(back, raw.values, rtol=1e-3, atol=1e-3)


def test_torch_inverse_matches_numpy_and_has_finite_gradients():
    rng = np.random.RandomState(1)
    raw = pd.DataFrame({"a": rng.lognormal(0, 2, 2000), "b": rng.normal(0, 1, 2000) * 1e4})
    pre = AutoencoderPreprocessor(skew_threshold=2.0).fit(raw)
    U = pre.transform(raw.iloc[:20])
    inv = pre.torch_inverse_fn()
    out = inv(torch.tensor(U))
    np.testing.assert_allclose(out.numpy(), pre.inverse_transform(U), rtol=1e-3, atol=1e-3)
    u = torch.tensor(U, requires_grad=True)
    inv(u).sum().backward()
    assert torch.isfinite(u.grad).all()      # no inf/NaN from expm1 on the non-log column


# ------------------------------------------------------------------- LRD
@pytest.mark.parametrize("kind", ["ae", "vae"])
def test_lrd_completeness_nonnegativity_and_lambda_endpoints(kind):
    model = make_ae() if kind == "ae" else make_vae()
    X = torch.randn(9, 12) * 2
    exp = LRDExplainer(model, kind=kind)
    variants, loss = exp.explain_variants(X, lams=(0.0, 0.5, 1.0))
    for lam, A in variants.items():
        np.testing.assert_allclose(A.sum(axis=1), loss, rtol=1e-4)      # completeness for every blend
        assert (A >= -1e-9).all()                                        # magnitudes
    comp = exp.components(X)
    np.testing.assert_allclose(variants[1.0], comp["direct"].numpy(), rtol=1e-5)      # lam=1 == direct error
    np.testing.assert_allclose(variants[0.0], comp["latent_path"].numpy(), rtol=1e-5)  # lam=0 == pure latent path


def test_lrd_loss_matches_reconstruction_error_scalar():
    ae = make_ae()
    X = torch.randn(7, 12)
    _, loss = LRDExplainer(ae, "ae").explain_variants(X)
    np.testing.assert_allclose(loss, ReconstructionError(ae, "ae")(X).detach().numpy(), rtol=1e-5)


def test_lrd_encoder_jacobian_matches_finite_differences():
    ae = make_ae()
    x = torch.randn(1, 12)
    J = LRDExplainer(ae, "ae").components(x)["J"][0].numpy()          # (K, p)
    eps = 1e-3
    fd = np.zeros_like(J)
    with torch.no_grad():
        for j in range(12):
            xp, xm = x.clone(), x.clone()
            xp[0, j] += eps
            xm[0, j] -= eps
            fd[:, j] = ((ae.encode(xp) - ae.encode(xm)) / (2 * eps))[0].numpy()
    np.testing.assert_allclose(J, fd, atol=2e-3)


def test_lrd_decoder_gradient_matches_finite_differences():
    """Step A uses dL/dz with x held fixed — verify against finite differences."""
    ae = make_ae()
    x = torch.randn(1, 12)
    exp = LRDExplainer(ae, "ae")
    comp = exp.components(x)
    with torch.no_grad():
        z = ae.encode(x)
        eps = 1e-3
        g_fd = np.zeros(z.shape[1])
        for k in range(z.shape[1]):
            zp, zm = z.clone(), z.clone()
            zp[0, k] += eps
            zm[0, k] -= eps
            g_fd[k] = (((x - ae.decode(zp)) ** 2).sum() - ((x - ae.decode(zm)) ** 2).sum()).item() / (2 * eps)
    np.testing.assert_allclose(comp["c"][0].numpy(), (z[0].numpy() * g_fd), atol=5e-3)


def test_lrd_latent_path_ignores_features_the_encoder_cannot_see():
    """
    Hand-built autoencoder: the encoder reads only features 0 and 1 (zero
    weights for the rest). The latent path must therefore attribute nothing to
    features 2.., even when they carry large direct reconstruction error —
    which is exactly the difference between the latent path and the direct
    per-feature error (lam=1).
    """
    p = 6
    ae = Autoencoder(p, [4, 2]).eval()
    with torch.no_grad():
        for mod in ae.modules():
            if isinstance(mod, nn.BatchNorm1d):
                mod.running_mean.zero_(); mod.running_var.fill_(1.0)
        first = ae.encoder[0]              # Linear(p -> 4)
        first.weight.zero_(); first.bias.zero_()
        first.weight[0, 0] = 1.0           # unit 0 reads feature 0
        first.weight[1, 1] = 1.0           # unit 1 reads feature 1
    x = torch.tensor([[2.0, 1.5, 4.0, -3.0, 2.5, 3.5]])
    exp = LRDExplainer(ae, "ae")
    comp = exp.components(x)
    latent = comp["latent_path"][0].numpy()
    direct = comp["direct"][0].numpy()
    assert latent[2:].sum() < 1e-6                 # encoder cannot see them -> no latent-path credit
    assert direct[2:].sum() > 1.0                  # ...though they do carry real direct error
    np.testing.assert_allclose(latent.sum(), comp["loss"][0].item(), rtol=1e-4)   # all error credited to 0/1


def test_lrd_fallback_when_nothing_is_distributable():
    """x = 0 makes every |J*x| zero: latent path must fall back to direct error, keeping completeness."""
    ae = make_ae()
    x = torch.zeros(1, 12)
    comp = LRDExplainer(ae, "ae").components(x)
    np.testing.assert_allclose(comp["latent_path"].numpy(), comp["direct"].numpy(), rtol=1e-5)


def test_vae_sigma_weighting_changes_result_but_not_completeness():
    vae = make_vae()
    X = torch.randn(6, 12)
    plain = LRDExplainer(vae, "vae", sigma_weighting=False).explain(X, lam=0.0)
    wtd = LRDExplainer(vae, "vae", sigma_weighting=True).explain(X, lam=0.0)
    assert not np.allclose(plain, wtd)
    np.testing.assert_allclose(plain.sum(1), wtd.sum(1), rtol=1e-4)


# --------------------------------------------------------- Manifold-LIME
def correlated_data(n=6000, p=8, seed=0):
    rng = np.random.RandomState(seed)
    A = rng.normal(size=(p, p))
    cov = A @ A.T / p + 0.3 * np.eye(p)
    return rng.multivariate_normal(np.zeros(p), cov, size=n), cov


def test_manifold_perturbations_follow_data_covariance_and_independent_ones_do_not():
    U, cov = correlated_data()
    lime_ = ManifoldLIME(lambda Z: np.zeros(len(Z)), U, scale=0.5, n_samples=60000)
    x = np.zeros(U.shape[1])
    Zm, _ = lime_.perturb(x, "manifold", np.random.RandomState(0))
    Zi, _ = lime_.perturb(x, "independent", np.random.RandomState(0))
    est = np.cov(Zm.T) / 0.25
    corr = lambda C: C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    true_corr = corr(lime_.cov)
    assert np.abs(corr(est) - true_corr).max() < 0.05          # manifold noise reproduces the correlations
    off = ~np.eye(U.shape[1], dtype=bool)
    assert np.abs(corr(np.cov(Zi.T))[off]).max() < 0.05         # independent noise has none
    assert np.abs(true_corr[off]).max() > 0.3                   # ...and the data genuinely has strong ones


def test_manifold_lime_recovers_linear_black_box():
    U, _ = correlated_data()
    w = np.array([2.0, -1.0, 0.5, 0.0, 3.0, -2.0, 1.0, 0.25])
    f = lambda Z: Z @ w + 5.0
    lime_ = ManifoldLIME(f, U, scale=0.3, n_samples=3000, ridge_alpha=1e-3)
    x = np.array([1.0, 0.5, -1.0, 2.0, 0.3, -0.7, 1.1, 0.0])
    res = lime_.explain(x, "manifold", seed=0)
    np.testing.assert_allclose(res.coef, w, atol=0.05)
    np.testing.assert_allclose(res.attribution, w * x, atol=0.05)
    assert res.r2 > 0.999


def test_manifold_perturbations_are_closer_to_real_data_than_independent_ones():
    U, _ = correlated_data(n=4000, p=10, seed=3)
    lime_ = ManifoldLIME(lambda Z: np.zeros(len(Z)), U, scale=0.6, n_samples=800)
    x = U[0]
    Zm, _ = lime_.perturb(x, "manifold", np.random.RandomState(0))
    Zi, _ = lime_.perturb(x, "independent", np.random.RandomState(0))
    dm = nearest_real_distance(Zm, U).mean()
    di = nearest_real_distance(Zi, U).mean()
    assert dm < di


def test_manifold_lime_on_nonlinear_scalar_is_stable_across_seeds():
    U, _ = correlated_data(p=6)
    f = lambda Z: np.sin(Z[:, 0]) + Z[:, 1] ** 2 - 0.5 * Z[:, 2]
    lime_ = ManifoldLIME(f, U, scale=0.2, n_samples=2000)
    x = np.array([0.5, 1.0, -0.3, 0.1, 0.0, 0.2])
    a = lime_.explain(x, "manifold", seed=1).attribution
    b = lime_.explain(x, "manifold", seed=2).attribution
    assert np.corrcoef(a, b)[0, 1] > 0.95


def test_torch_black_box_adapter():
    f = Quadratic(np.array([1.0, 2.0], dtype=np.float32))
    bb = torch_black_box(f, batch_size=3)
    Z = np.random.RandomState(0).normal(size=(10, 2)).astype(np.float32)
    np.testing.assert_allclose(bb(Z), (np.array([1.0, 2.0]) * Z ** 2).sum(1), rtol=1e-5)


# ------------------------------------------------------------- validation
def test_replacement_effect_prefers_a_correct_ranking_over_a_reversed_one():
    rng = np.random.RandomState(0)
    n, p = 200, 20
    U = rng.normal(size=(n, p)) * 0.1
    U[:, :3] += rng.normal(size=(n, 3)) * 5               # three features carry the signal
    f = lambda M: (M ** 2).sum(axis=1)
    donors = rng.normal(size=(500, p)) * 0.1
    good = U ** 2                                         # perfect attribution of f
    bad = 1.0 / (good + 1e-6)                             # reversed ranking (ranking is by |attribution|, so -good would not reverse it)
    s_good = summarize_replacement(replacement_effect(f, U, good, donors, k=3))
    s_bad = summarize_replacement(replacement_effect(f, U, bad, donors, k=3))
    assert s_good["top_over_bottom_ratio"] > 10 * s_bad["top_over_bottom_ratio"]
    assert s_good["frac_instances_top_gt_bottom"] > 0.95
    assert s_good["mean_top_reduction"] > 0               # repairing flagged features lowers the error


def test_pairwise_agreement_identical_vs_unrelated():
    rng = np.random.RandomState(0)
    A = rng.normal(size=(30, 15))
    out = pairwise_agreement({"a": A, "b": A.copy(), "c": rng.normal(size=(30, 15))})
    assert out[("a", "b")] > 0.99
    assert out[("a", "c")] < 0.4


# ---------------------------------------------------------------- snapping
def test_integer_columns_detects_whole_number_columns():
    X = np.column_stack([np.array([0, 1, 5, 2.0]), np.array([0.5, 1, 2, 3]), np.array([1, 1, 1, 1.0])])
    assert integer_columns(X).tolist() == [True, False, True]


def test_snap_via_standard_scaler_makes_integer_columns_whole_in_raw_units():
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(0)
    raw = np.column_stack([rng.randint(0, 6, 500), rng.normal(10, 3, 500), rng.randint(0, 2, 500)]).astype(float)
    sc = StandardScaler().fit(raw)
    mask = integer_columns(raw)
    snap = snap_via_standard_scaler(sc, mask)
    Z = sc.transform(raw) + rng.normal(0, 0.4, raw.shape)
    back = snap(Z) * sc.scale_ + sc.mean_
    assert np.allclose(back[:, mask], np.round(back[:, mask]), atol=1e-9)        # whole numbers
    np.testing.assert_allclose(snap(Z)[:, ~mask], Z[:, ~mask])                  # continuous column untouched


def test_snap_via_standard_scaler_tiles_over_flattened_sequences():
    from sklearn.preprocessing import StandardScaler
    raw = np.column_stack([np.arange(20) % 4, np.linspace(0, 5, 20)]).astype(float)
    sc = StandardScaler().fit(raw)
    mask = integer_columns(raw)                                  # [True, False]
    snap = snap_via_standard_scaler(sc, mask, tile=3)            # 3 timesteps x 2 features
    Z = np.random.RandomState(0).normal(size=(5, 6))
    out = snap(Z) * np.tile(sc.scale_, 3) + np.tile(sc.mean_, 3)
    assert np.allclose(out[:, [0, 2, 4]], np.round(out[:, [0, 2, 4]]), atol=1e-9)


def test_snap_via_preprocessor_roundtrip_and_integrality():
    rng = np.random.RandomState(0)
    raw = pd.DataFrame({"count": rng.poisson(30, 4000).astype(float),
                        "amt": rng.lognormal(3, 1.5, 4000),
                        "flag": rng.randint(0, 2, 4000).astype(float)})
    pre = AutoencoderPreprocessor(skew_threshold=2.0, clip_value=50.0).fit(raw)
    mask = integer_columns(raw.values)
    snap = snap_via_preprocessor(pre, mask)
    U = pre.transform(raw.iloc[:200])
    np.testing.assert_allclose(snap(U), U, atol=1e-3)                            # idempotent on real data
    noisy = U + rng.normal(0, 0.3, U.shape)
    back = pre.inverse_transform(snap(noisy))
    assert np.allclose(back[:, mask], np.round(back[:, mask]), atol=1e-3)      # integer columns whole in RAW units


def test_manifold_lime_snap_only_applies_in_manifold_mode():
    U, _ = correlated_data(n=3000, p=5, seed=1)
    snap = lambda Z: np.round(Z)
    lime_ = ManifoldLIME(lambda Z: np.zeros(len(Z)), U, scale=0.5, n_samples=300, snap=snap)
    x = np.zeros(5)
    Zm, _ = lime_.perturb(x, "manifold", np.random.RandomState(0))
    Zn, _ = lime_.perturb(x, "manifold_noround", np.random.RandomState(0))
    Zi, _ = lime_.perturb(x, "independent", np.random.RandomState(0))
    assert np.allclose(Zm, np.round(Zm))
    assert not np.allclose(Zn, np.round(Zn))
    assert not np.allclose(Zi, np.round(Zi))


def test_manifold_lime_scale_and_alpha_overrides_take_effect_without_mutating_the_object():
    U, _ = correlated_data(p=5)
    lime_ = ManifoldLIME(lambda Z: (Z ** 2).sum(1), U, scale=0.3, n_samples=400, ridge_alpha=1.0)
    x = np.array([0.5, 1.0, -0.3, 0.1, 0.2])
    a = lime_.explain(x, "manifold", seed=0).attribution
    b = lime_.explain(x, "manifold", seed=0, scale=0.9).attribution
    c = lime_.explain(x, "manifold", seed=0, ridge_alpha=1e4).attribution
    assert not np.allclose(a, b) and not np.allclose(a, c)
    np.testing.assert_allclose(a, lime_.explain(x, "manifold", seed=0).attribution)   # object unchanged
    assert lime_.scale == 0.3 and lime_.ridge_alpha == 1.0
    assert np.abs(c).sum() < np.abs(a).sum()                                           # heavy ridge shrinks


def test_replacement_effect_zero_mode_repairs_flagged_features():
    rng = np.random.RandomState(0)
    n, p = 150, 12
    U = rng.normal(size=(n, p)) * 0.1
    U[:, :2] += rng.normal(size=(n, 2)) * 5
    f = lambda M: (M ** 2).sum(axis=1)
    res = replacement_effect(f, U, U ** 2, donors=None, k=2)
    assert res["top_reduction"].mean() > 0.9 * (U[:, :2] ** 2).sum(1).mean()      # zeroing the signal features removes ~all of f
    assert np.mean(res["top_effect"] > res["bottom_effect"]) > 0.95


def test_manifold_lime_never_clips_the_instance_itself():
    """Regression: an instance outside the reference range (e.g. year=2013 vs a 2012
    reference) used to have every perturbation clipped back into range, so the
    kernel weights underflowed to zero. Perturbations must stay centred on x."""
    rng = np.random.RandomState(0)
    U = rng.normal(size=(500, 4))
    ml = ManifoldLIME(lambda Z: Z @ np.array([1.0, -2.0, 0.5, 3.0]), U, scale=0.3, n_samples=800)
    x = U[0].copy(); x[3] = U[:, 3].max() + 4.0          # far outside the reference range
    Z, _ = ml.perturb(x, "manifold", np.random.RandomState(1))
    assert abs(np.median(Z[:, 3]) - x[3]) < 0.1             # centred on x, not dragged into range
    assert Z[:, 3].max() <= x[3] + 1e-12                    # still clipped on the far side
    res = ml.explain(x, "manifold", seed=0)
    assert res.r2 > 0.9 and res.mean_weight > 0
    # an in-range instance is unaffected by the widening
    x_in = U[1]
    Z_old = np.clip(x_in + 0.3 * (np.random.RandomState(2).standard_normal((800, 4)) @ ml.L.T), ml.lo, ml.hi)
    Z_new, _ = ml.perturb(x_in, "manifold", np.random.RandomState(2))
    assert np.allclose(Z_old, Z_new)
