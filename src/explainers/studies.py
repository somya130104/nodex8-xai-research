"""
One reusable driver for the Phase 3 studies, so every model (autoencoder,
VAE, TabNet, LSTM) is put through IDENTICAL procedures and the results are
directly comparable.

Everything operates on a flat 2-D coordinate system (n_instances x n_columns).
A sequence model is handled by flattening (T timesteps x F features) into
T*F columns and letting its scalar function reshape internally; per-feature
aggregation over time is done by the caller afterwards.

run_study() produces, for one model:
  attributions            all explainer variants (wide format)
  background_sensitivity  how much Adapted SHAP's answer depends on the reference
  manifold_lime_diagnostics  fidelity / plausibility / stability, 3 perturbation modes
  faithfulness            feature-replacement check for every variant + a random baseline
  latency                 single-instance timings (+ a KernelSHAP reference)
  agreement preview       placeholder weighted tau among the three PRIMARY explainers
  summary json            the small file that is tracked in git

PRIMARY variants are fixed in advance (not chosen from the results):
    Adapted SHAP  = ig_kmeans        Manifold-LIME = lime_manifold        LRD = lrd_lam0.5
"""
import itertools
import json
import os
import time
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch

from src.eval_utils import baseline_instance_tau
from src.explainers.adapted_shap import AdaptedSHAP
from src.explainers.baselines import kmeans_baselines, nearest_neighbor_baselines
from src.explainers.lrd import LRDExplainer
from src.explainers.manifold_lime import ManifoldLIME, torch_black_box
from src.explainers.validation import (
    replacement_effect, summarize_replacement, pairwise_agreement, nearest_real_distance,
)

PRIMARY = {"AdaptedSHAP": "ig_kmeans", "ManifoldLIME": "lime_manifold", "LRD": "lrd_lam0.5"}


@dataclass
class StudyConfig:
    seed: int = 42
    ig_steps: int = 50
    kmeans_k: int = 10
    gs_baselines: int = 100
    lime_scale: float = 0.3
    lime_samples: int = 2000
    diag_per_class: int = 15
    stability_seeds: int = 5
    latency_n: int = 200
    kernelshap_n: int = 5
    faith_k: int = 5
    n_donors: int = 5000
    topk_jaccard: int = 10
    lime_diag_samples: int = 1000
    run_sweeps: bool = True
    ridge_alpha_default: float = 1.0
    sweep_scales: tuple = (0.1, 0.3, 1.0)
    sweep_alphas: tuple = (1.0, 10.0, 100.0)
    sweep_instances: int = 12
    sweep_samples: int = 500
    kernelshap_big_bg: int = 100
    kernelshap_big_n: int = 3


def jaccard_topk(a, b, k):
    sa, sb = set(np.argsort(-np.abs(a))[:k]), set(np.argsort(-np.abs(b))[:k])
    return len(sa & sb) / len(sa | sb)


def latency_ms(per_instance, n, n_warmup=3):
    for i in range(n_warmup):
        per_instance(i)
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        per_instance(i)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)
    return dict(n=int(n), p50_ms=float(np.percentile(ts, 50)), p95_ms=float(np.percentile(ts, 95)),
                mean_ms=float(ts.mean()), max_ms=float(ts.max()))


def _groups(y_inst):
    return (("all", np.ones(len(y_inst), bool)), ("positives", y_inst == 1), ("negatives", y_inst == 0))


def run_study(*, model_name, f_torch, U_inst, y_inst, ids, U_normal, columns, out_dir,
              raw_mean_baseline, snap=None, lrd_model=None, lrd_kind=None,
              extra_attrs=None, extra_latency=None, baseline_note=None,
              cfg: StudyConfig = None, run_kernelshap=True):
    """
    f_torch : torch module (B, D) -> (B,): the scalar being explained, in the flat coordinate system.
    U_inst  : (n, D) instances.   U_normal: (N, D) reference population (normal rows / training rows).
    columns : list of D column names.
    raw_mean_baseline : (1, D) the arithmetic-mean-in-raw-units reference mapped into this coordinate system.
    snap    : optional integer-snapping callable for Manifold-LIME.
    lrd_model/lrd_kind : autoencoder-family model + 'ae'|'vae' to enable LRD (None -> LRD not applicable).
    baseline_note : free text stored in the summary describing how the `ig_rawmean` reference was defined for this model.
    extra_attrs : {variant: (explainer_family, (n, D) array)} extra signals to carry along (e.g. TabNet's native masks).
    """
    cfg = cfg or StudyConfig()
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()
    n, D = U_inst.shape
    U_inst = U_inst.astype(np.float32)
    U_normal = U_normal.astype(np.float32)
    f_np = torch_black_box(f_torch)
    with torch.no_grad():
        f_vals = f_torch(torch.tensor(U_inst)).numpy()
    pd.DataFrame({"Instance_ID": ids, "True_Label": y_inst, "Scalar_Value": f_vals}).to_csv(
        os.path.join(out_dir, f"instances_{model_name}.csv"), index=False)

    attrs, meta = {}, {}
    summary = {"scalar_mean_by_group": {g: float(f_vals[m].mean()) for g, m in _groups(y_inst) if m.any()},
               "model": model_name, "n_instances": int(n), "n_columns": int(D),
               "n_positive": int(y_inst.sum()), "primary_variants": dict(PRIMARY),
               "rawmean_baseline_note": baseline_note}

    # ---------------------------------------------------------------- Adapted SHAP
    print("[1/3] Adapted SHAP (Integrated Gradients / GradientSHAP)...")
    ash = AdaptedSHAP(f_torch)
    b_zero = np.zeros((1, D), dtype=np.float32)
    b_raw = np.asarray(raw_mean_baseline, dtype=np.float32).reshape(1, D)
    b_km = kmeans_baselines(U_normal, k=cfg.kmeans_k, seed=cfg.seed)
    b_nn = nearest_neighbor_baselines(U_inst, U_normal, seed=cfg.seed)
    ig_delta = {}
    attrs["ig_zeros"], ig_delta["ig_zeros"] = ash.integrated_gradients(U_inst, b_zero, cfg.ig_steps)
    attrs["ig_rawmean"], ig_delta["ig_rawmean"] = ash.integrated_gradients(U_inst, b_raw, cfg.ig_steps)
    attrs["ig_kmeans"], ig_delta["ig_kmeans"] = ash.ig_multi_baseline(U_inst, b_km, cfg.ig_steps)
    attrs["ig_nn"], ig_delta["ig_nn"] = ash.integrated_gradients(U_inst, b_nn, cfg.ig_steps)
    gs_base = U_normal[np.random.RandomState(cfg.seed).choice(len(U_normal), min(cfg.gs_baselines, len(U_normal)), replace=False)]
    attrs["gradient_shap"] = ash.gradient_shap(U_inst, gs_base, n_samples=50, stdevs=0.0, seed=cfg.seed)
    for v in ("ig_zeros", "ig_rawmean", "ig_kmeans", "ig_nn", "gradient_shap"):
        meta[v] = "AdaptedSHAP"
    scale_ref = float(np.abs(f_vals).mean())
    summary["ig_completeness"] = {v: {"mean_abs_residual": float(np.abs(d).mean()),
                                      "mean_abs_residual_over_mean_scalar": float(np.abs(d).mean() / (scale_ref + 1e-12))}
                                  for v, d in ig_delta.items()}
    summary["baseline_geometry"] = {"max_abs_mean_of_reference": float(np.abs(U_normal.mean(0)).max()),
                                    "l2_zeros_to_rawmean": float(np.linalg.norm(b_raw))}
    print("   IG completeness residual / mean scalar: " + ", ".join(
        f"{v}={d['mean_abs_residual_over_mean_scalar']:.3f}" for v, d in summary["ig_completeness"].items()))

    # ---------------------------------------------------------------- Manifold-LIME
    print("[2/3] Manifold-LIME (fitting Ledoit-Wolf covariance)...")
    mlime = ManifoldLIME(f_np, U_normal, scale=cfg.lime_scale, n_samples=cfg.lime_samples, snap=snap)
    summary["ledoit_wolf_shrinkage"] = mlime.shrinkage
    print(f"   Ledoit-Wolf shrinkage = {mlime.shrinkage:.4f}  |  snapping {'ON' if snap is not None else 'OFF'}")
    r2 = {}
    attrs["lime_manifold"], r2["manifold"] = mlime.explain_many(U_inst, "manifold", seed=0)
    attrs["lime_manifold_noround"], r2["manifold_noround"] = mlime.explain_many(U_inst, "manifold_noround", seed=0)
    attrs["lime_vanilla"], r2["independent"] = mlime.explain_many(U_inst, "independent", seed=0)
    for v in ("lime_manifold", "lime_manifold_noround", "lime_vanilla"):
        meta[v] = "ManifoldLIME"

    # ---------------------------------------------------------------- LRD
    if lrd_model is not None:
        print("[3/3] LRD (Latent Reconstruction Decomposition)...")
        lrd = LRDExplainer(lrd_model, kind=lrd_kind, lam=0.5)
        lrd_vars, lrd_loss = lrd.explain_variants(U_inst, lams=(0.0, 0.5, 1.0))
        for lam, A in lrd_vars.items():
            attrs[f"lrd_lam{lam}"] = A
            meta[f"lrd_lam{lam}"] = "LRD"
        if lrd_kind == "vae":
            attrs["lrd_lam0.5_sigma"] = LRDExplainer(lrd_model, kind="vae", lam=0.5, sigma_weighting=True).explain(U_inst)
            meta["lrd_lam0.5_sigma"] = "LRD"
        summary["lrd_completeness_max_rel_error"] = float(np.max(np.abs(lrd_vars[0.5].sum(1) - lrd_loss) / (lrd_loss + 1e-12)))
        print(f"   LRD completeness: max relative |sum(attr) - L| = {summary['lrd_completeness_max_rel_error']:.2e}")
    else:
        print("[3/3] LRD not applicable to this model (no autoencoder bottleneck) — skipped.")

    for v, (fam, A) in (extra_attrs or {}).items():
        attrs[v] = A
        meta[v] = fam

    # ---------------------------------------------------------------- save attributions
    blocks = []
    for name, A in attrs.items():
        b = pd.DataFrame(A, columns=columns)
        b.insert(0, "Variant", name)
        b.insert(0, "Explainer", meta[name])
        b.insert(0, "True_Label", y_inst)
        b.insert(0, "Instance_ID", ids)
        b.insert(0, "Model", model_name)
        blocks.append(b)
    pd.concat(blocks, ignore_index=True).to_csv(os.path.join(out_dir, f"attributions_{model_name}.csv"), index=False)

    # ---------------------------------------------------------------- background sensitivity
    print("Background-sensitivity study (Adapted SHAP)...")
    bs_variants = ["ig_zeros", "ig_rawmean", "ig_kmeans", "ig_nn", "gradient_shap"]
    rows = []
    for a, b in itertools.combinations(bs_variants, 2):
        taus = np.array([baseline_instance_tau(attrs[a][i], attrs[b][i]) for i in range(n)], dtype=float)
        jac = np.array([jaccard_topk(attrs[a][i], attrs[b][i], cfg.topk_jaccard) for i in range(n)])
        rows.append({"variant_a": a, "variant_b": b, "mean_tau_all": np.nanmean(taus),
                     "mean_tau_positives": np.nanmean(taus[y_inst == 1]) if (y_inst == 1).any() else np.nan,
                     "mean_tau_negatives": np.nanmean(taus[y_inst == 0]) if (y_inst == 0).any() else np.nan,
                     f"mean_top{cfg.topk_jaccard}_jaccard": jac.mean()})
    bs = pd.DataFrame(rows)
    bs.to_csv(os.path.join(out_dir, f"background_sensitivity_{model_name}.csv"), index=False)
    print(bs.round(3).to_string(index=False))
    summary["background_sensitivity"] = bs.round(4).to_dict(orient="records")

    # ---------------------------------------------------------------- Manifold-LIME diagnostics
    print("Manifold-LIME diagnostics (fidelity / plausibility / stability)...")
    rng = np.random.RandomState(cfg.seed)
    pos = np.where(y_inst == 1)[0]
    neg = np.where(y_inst == 0)[0]
    if len(pos) > 0:
        d_idx = np.concatenate([pos[:cfg.diag_per_class], neg[:cfg.diag_per_class]])
    else:
        d_idx = rng.choice(n, min(2 * cfg.diag_per_class, n), replace=False)
    mode_names = {"manifold": "manifold", "manifold_noround": "manifold_noround", "independent": "vanilla"}
    diag = []
    for mode, name in mode_names.items():
        diag += [{"metric": "local_fidelity_r2_mean", "mode": name, "group": "all", "value": float(r2[mode].mean())},
                 {"metric": "local_fidelity_r2_median", "mode": name, "group": "all", "value": float(np.median(r2[mode]))}]
    n_save = mlime.n_samples
    mlime.n_samples = cfg.lime_diag_samples
    nn_d = {m: [] for m in mode_names}
    clip_f = {m: [] for m in mode_names}
    for i in d_idx:
        for mode in mode_names:
            Z, fc = mlime.perturb(U_inst[i], mode, np.random.RandomState(100 + int(i)))
            nn_d[mode].append(nearest_real_distance(Z.astype(np.float32), U_normal, seed=cfg.seed).mean())
            clip_f[mode].append(fc)
    mlime.n_samples = n_save
    # Reference: how far do REAL normal rows sit from OTHER real normal rows? (a disjoint
    # half is used as the search set so a row cannot match itself)
    half = len(U_normal) // 2
    real_ref = nearest_real_distance(U_normal[:200], U_normal[half:], seed=cfg.seed).mean()
    for mode, name in mode_names.items():
        diag += [{"metric": "mean_nn_distance_of_perturbations_to_real_rows", "mode": name, "group": "diag_subset", "value": float(np.mean(nn_d[mode]))},
                 {"metric": "frac_perturbed_entries_clipped_to_range", "mode": name, "group": "diag_subset", "value": float(np.mean(clip_f[mode]))}]
    diag.append({"metric": "mean_nn_distance_of_REAL_rows_to_other_real_rows", "mode": "reference", "group": "reference", "value": float(real_ref)})
    stab = {m: [] for m in mode_names}
    for i in d_idx:
        for mode in mode_names:
            vs = [mlime.explain(U_inst[i], mode, seed=1000 + s).attribution for s in range(cfg.stability_seeds)]
            t = [baseline_instance_tau(vs[a], vs[b]) for a, b in itertools.combinations(range(cfg.stability_seeds), 2)]
            stab[mode].append(np.nanmean(t))
    for mode, name in mode_names.items():
        diag.append({"metric": f"stability_mean_pairwise_tau_over_{cfg.stability_seeds}_seeds", "mode": name,
                     "group": "diag_subset", "value": float(np.mean(stab[mode]))})
    dg = pd.DataFrame(diag)
    dg.to_csv(os.path.join(out_dir, f"manifold_lime_diagnostics_{model_name}.csv"), index=False)
    print(dg.round(4).to_string(index=False))
    summary["manifold_lime_diagnostics"] = dg.round(5).to_dict(orient="records")

    # ---------------------------------------------------------------- sensitivity sweeps
    if cfg.run_sweeps:
        print("Manifold-LIME sensitivity sweeps (perturbation scale and ridge strength)...")
        sw_idx = np.concatenate([pos[:cfg.sweep_instances // 2], neg[:cfg.sweep_instances // 2]]) if len(pos) > 0 \
            else d_idx[:cfg.sweep_instances]
        sweep = []

        def one_setting(scale, alpha, with_plausibility):
            n_s = mlime.n_samples
            mlime.n_samples = cfg.sweep_samples
            for mode, name in mode_names.items():
                nn_vals, stab_vals, r2_vals = [], [], []
                for i in sw_idx:
                    res0 = mlime.explain(U_inst[i], mode, seed=1000, scale=scale, ridge_alpha=alpha)
                    vs = [res0.attribution] + [mlime.explain(U_inst[i], mode, seed=1000 + s, scale=scale, ridge_alpha=alpha).attribution
                                               for s in range(1, cfg.stability_seeds)]
                    stab_vals.append(np.nanmean([baseline_instance_tau(vs[a], vs[b])
                                                 for a, b in itertools.combinations(range(len(vs)), 2)]))
                    r2_vals.append(res0.r2)
                    if with_plausibility:
                        Z, _ = mlime.perturb(U_inst[i], mode, np.random.RandomState(100 + int(i)), scale=scale)
                        nn_vals.append(nearest_real_distance(Z.astype(np.float32), U_normal, seed=cfg.seed).mean())
                sweep.append({"scale": scale, "ridge_alpha": alpha, "mode": name,
                              "mean_stability_tau": float(np.mean(stab_vals)), "mean_fidelity_r2": float(np.mean(r2_vals)),
                              "mean_nn_distance_to_real_rows": float(np.mean(nn_vals)) if with_plausibility else np.nan})
            mlime.n_samples = n_s

        for sc in cfg.sweep_scales:
            one_setting(sc, cfg.ridge_alpha_default, True)
        for al in cfg.sweep_alphas:
            if al != cfg.ridge_alpha_default:
                one_setting(cfg.lime_scale, al, False)
        sw = pd.DataFrame(sweep)
        sw.to_csv(os.path.join(out_dir, f"manifold_lime_sweeps_{model_name}.csv"), index=False)
        print(sw.round(3).to_string(index=False))
        summary["manifold_lime_sweeps"] = sw.round(4).to_dict(orient="records")

    # ---------------------------------------------------------------- faithfulness
    print(f"Faithfulness (overwrite top-{cfg.faith_k} / bottom-{cfg.faith_k} / random-{cfg.faith_k} ranked columns): "
          "'donor' = values of a random real normal row; 'zero' = the typical (normal-mean) value, i.e. a repair test")
    donors = U_normal[np.random.RandomState(cfg.seed + 1).choice(len(U_normal), min(cfg.n_donors, len(U_normal)), replace=False)]
    fv = dict(attrs)
    fv["RANDOM_ranking"] = np.random.RandomState(cfg.seed + 2).normal(size=attrs["ig_kmeans"].shape)
    frows = []
    for repl_name, repl_donors in (("donor", donors), ("zero", None)):
        for name, A in fv.items():
            res = replacement_effect(f_np, U_inst, A, repl_donors, k=cfg.faith_k, seed=cfg.seed)
            for gname, mask in _groups(y_inst):
                if not mask.any():
                    continue
                row = summarize_replacement({k_: v[mask] for k_, v in res.items()})
                row.update({"variant": name, "explainer": meta.get(name, "reference"), "group": gname, "replacement": repl_name})
                frows.append(row)
    fa = pd.DataFrame(frows)
    fa.to_csv(os.path.join(out_dir, f"faithfulness_{model_name}.csv"), index=False)
    summary["faithfulness_all"] = {}
    for repl_name in ("donor", "zero"):
        show = fa[(fa.group == "all") & (fa.replacement == repl_name)][
            ["variant", "mean_top_effect", "mean_bottom_effect", "top_over_bottom_ratio",
             "top_over_random_ratio", "frac_instances_top_gt_bottom", "mean_top_reduction"]]
        print(f"--- replacement = {repl_name} ---")
        print(show.round(3).to_string(index=False))
        summary["faithfulness_all"][repl_name] = show.round(4).to_dict(orient="records")

    # ---------------------------------------------------------------- agreement preview
    prim = {k_: attrs[v] for k_, v in PRIMARY.items() if v in attrs}
    if "TabNetNative" in {meta[v] for v in attrs}:
        nat = [v for v in attrs if meta[v] == "TabNetNative"][0]
        prim["TabNetNative"] = attrs[nat]
    agree = {}
    for gname, mask in _groups(y_inst):
        if not mask.any():
            continue
        for (a, b), v in pairwise_agreement(prim, rows=np.where(mask)[0]).items():
            agree[f"{a}_vs_{b}__{gname}"] = v
    summary["agreement_preview_placeholder_tau"] = agree
    print("Pairwise agreement among PRIMARY explainers (placeholder weighted tau; NOT the CLI):")
    for k_, v in agree.items():
        print(f"   {k_:60s} {v:6.3f}")

    # ---------------------------------------------------------------- latency
    print("Latency (single instance, CPU)...")
    n_lat = min(cfg.latency_n, n)
    Ut = torch.tensor(U_inst)
    lat = {}
    lat["ig_zeros (50 steps)"] = latency_ms(lambda i: ash.integrated_gradients(U_inst[i:i + 1], b_zero, cfg.ig_steps), n_lat)
    lat["ig_kmeans (10 baselines x 50 steps)"] = latency_ms(lambda i: ash.ig_multi_baseline(U_inst[i:i + 1], b_km, cfg.ig_steps), n_lat)
    lat["gradient_shap (100 baselines, 50 samples)"] = latency_ms(
        lambda i: ash.gradient_shap(U_inst[i:i + 1], gs_base, n_samples=50, stdevs=0.0, seed=cfg.seed), n_lat)
    lat[f"manifold_lime ({cfg.lime_samples} samples)"] = latency_ms(lambda i: mlime.explain(U_inst[i], "manifold", seed=i), n_lat)
    lat[f"vanilla_lime ({cfg.lime_samples} samples)"] = latency_ms(lambda i: mlime.explain(U_inst[i], "independent", seed=i), n_lat)
    if lrd_model is not None:
        lrd_t = LRDExplainer(lrd_model, kind=lrd_kind, lam=0.5)
        lat["LRD (decoder grad + encoder Jacobian)"] = latency_ms(lambda i: lrd_t.explain(Ut[i:i + 1]), n_lat)
    for name, fn in (extra_latency or {}).items():
        lat[name] = latency_ms(fn, min(n_lat, 50))
    if run_kernelshap:
        import shap
        ker = shap.KernelExplainer(f_np, b_km)
        ks = []
        for i in range(cfg.kernelshap_n):
            t0 = time.perf_counter()
            ker.shap_values(U_inst[i:i + 1], silent=True)
            ks.append((time.perf_counter() - t0) * 1000.0)
        lat["KernelSHAP reference (10 k-means background, default nsamples)"] = dict(
            n=cfg.kernelshap_n, p50_ms=float(np.median(ks)), p95_ms=float(np.max(ks)), mean_ms=float(np.mean(ks)), max_ms=float(np.max(ks)))
        bg_big = U_normal[np.random.RandomState(cfg.seed + 3).choice(len(U_normal), min(cfg.kernelshap_big_bg, len(U_normal)), replace=False)]
        ker_big = shap.KernelExplainer(f_np, bg_big)
        ks2 = []
        for i in range(cfg.kernelshap_big_n):
            t0 = time.perf_counter()
            ker_big.shap_values(U_inst[i:i + 1], silent=True)
            ks2.append((time.perf_counter() - t0) * 1000.0)
        lat[f"KernelSHAP reference ({cfg.kernelshap_big_bg}-row background, default nsamples)"] = dict(
            n=cfg.kernelshap_big_n, p50_ms=float(np.median(ks2)), p95_ms=float(np.max(ks2)), mean_ms=float(np.mean(ks2)), max_ms=float(np.max(ks2)))
    lt = pd.DataFrame([{"explainer": k_, **v} for k_, v in lat.items()])
    lt.to_csv(os.path.join(out_dir, f"latency_{model_name}.csv"), index=False)
    print(lt.round(2).to_string(index=False))
    summary["latency_ms"] = lt.round(3).to_dict(orient="records")

    summary["runtime_seconds"] = round(time.time() - t_start, 1)
    with open(os.path.join(out_dir, f"tri_explainer_{model_name}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"✅ {model_name} done in {summary['runtime_seconds']}s -> {out_dir}")
    return summary, attrs, meta


def primary_attributions(*, f_torch, U_inst, U_normal, snap=None, lrd_model=None, lrd_kind=None,
                         cfg: StudyConfig = None):
    """
    Only the three PRIMARY explainers, built EXACTLY as run_study builds them (same
    k-means references, IG steps, Manifold-LIME scale / sample count / snapping /
    per-row seeds, LRD lambda). Used by Phase 4 to explain new instances (e.g. the
    out-of-time months) without re-running the whole study.
    Returns ({"AdaptedSHAP": (n, D), "ManifoldLIME": (n, D)[, "LRD": (n, D)]}, scalar (n,)).
    """
    cfg = cfg or StudyConfig()
    U_inst = U_inst.astype(np.float32)
    U_normal = U_normal.astype(np.float32)
    with torch.no_grad():
        f_vals = f_torch(torch.tensor(U_inst)).numpy()
    out = {}
    b_km = kmeans_baselines(U_normal, k=cfg.kmeans_k, seed=cfg.seed)
    out["AdaptedSHAP"], _ = AdaptedSHAP(f_torch).ig_multi_baseline(U_inst, b_km, cfg.ig_steps)
    mlime = ManifoldLIME(torch_black_box(f_torch), U_normal, scale=cfg.lime_scale,
                         n_samples=cfg.lime_samples, snap=snap)
    out["ManifoldLIME"], _ = mlime.explain_many(U_inst, "manifold", seed=0)
    if lrd_model is not None:
        out["LRD"] = LRDExplainer(lrd_model, kind=lrd_kind, lam=0.5).explain(U_inst)
    return out, f_vals
