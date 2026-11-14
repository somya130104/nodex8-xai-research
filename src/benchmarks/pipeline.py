"""
The benchmark stages, shared by the TradeFraud and DemandForecast runner scripts.

A ModelBundle describes one model to benchmark; each run_* function applies one benchmark stage to it and writes
  <out_dir>/<stage>_<model>.csv                  the full table (git-ignored like other CSVs)
  <out_dir>/benchmark_<stage>_<model>_summary.json   the same rows plus settings (tracked in git)
See src/benchmarks/__init__.py for the decision rules these stages implement.
"""
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from src.benchmarks import runtime as rt
from src.benchmarks.faithfulness import deletion_insertion, summarize
from src.benchmarks.robustness import adversarial_search, noise_stability
from src.benchmarks.suite import CONTROLS, ExplainerSuite
from src.benchmarks.synthetic import bootstrap_ci, corrupt_rows, make_planted_task, precision_at_k

# Phase 3 stored-attribution variant -> suite name
STORED = {"ig_kmeans": "AdaptedSHAP", "ig_zeros": "AdaptedSHAP_zero", "gradient_shap": "GradientSHAP",
          "lime_manifold": "ManifoldLIME", "lime_vanilla": "VanillaLIME", "lrd_lam0.0": "LRD_lam0",
          "lrd_lam0.5": "LRD_lam0.5", "lrd_lam1.0": "LRD_lam1", "tabnet_native_mask": "NativeMask"}


@dataclass
class ModelBundle:
    name: str                                   # e.g. "Model4_Autoencoder"
    kind: str                                   # 'ae' | 'vae' | 'tabnet' | 'lstm'
    suite: ExplainerSuite
    U_inst: np.ndarray                          # the Phase 3 explained instances (preprocessed coordinates)
    attrs_stored: Dict[str, np.ndarray]         # Phase 3 attributions, keyed by suite name
    U_normal: np.ndarray                        # normal reference population
    K: int                                      # deletion/insertion depth
    U_runtime: np.ndarray                       # pool of instances for the latency runs
    sigma: np.ndarray                           # per-feature std of the normal population
    run_kernelshap: bool = True
    gt_rows: Optional[np.ndarray] = None        # raw normal rows for the corruption test (autoencoder family only)
    gt_transform: Optional[Callable] = None     # raw -> preprocessed coordinates
    n_runtime: int = 1000
    n_kernel_runtime: int = 30
    n_mem: int = 30
    n_stab: int = 60
    n_adv: int = 15
    adv_iters: int = 30
    adv_cands: int = 4
    extra: dict = field(default_factory=dict)


def _save(out_dir, stage, model, df, meta=None):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{stage}_{model}.csv"), index=False)
    payload = {"stage": stage, "model": model, "settings": meta or {}, "rows": json.loads(df.round(5).to_json(orient="records"))}
    with open(os.path.join(out_dir, f"benchmark_{stage}_{model}_summary.json"), "w") as fh:
        json.dump(payload, fh, indent=1)


# ============================================================================ 5.1 faithfulness
def run_faithfulness(b: ModelBundle, out_dir, seed=42, n_donors=5000):
    t0 = time.time()
    print(f"[{b.name}] 5.1 faithfulness (K={b.K}) ...")
    S = b.suite
    variants = dict(b.attrs_stored)
    for v in (["ManifoldLIME_stochround"] if S.has_stoch else []) + (["KernelSHAP"] if b.run_kernelshap else []):
        variants[v] = S.explain(v, b.U_inst, seed=0)
    for c in CONTROLS:
        variants[c] = S.explain(c, b.U_inst, seed=0)
    rs = np.random.RandomState(seed + 1)
    donors = b.U_normal[rs.choice(len(b.U_normal), len(b.U_inst), replace=len(b.U_normal) < len(b.U_inst))]
    bases = {"zero": np.zeros_like(b.U_inst), "donor": donors}
    rows, curves = [], {}
    for op, base in bases.items():
        for v, A in variants.items():
            res = deletion_insertion(S.f_np, b.U_inst, A, b.K, base, seed=seed)
            rows.append({"model": b.name, "variant": v, "operator": op, "n": len(b.U_inst), "K": b.K, **summarize(res)})
            curves[f"{v}|{op}|del"], curves[f"{v}|{op}|ins"] = res["curve_del"], res["curve_ins"]
            curves[f"{v}|{op}|rand_del"], curves[f"{v}|{op}|rand_ins"] = res["curve_rand_del"], res["curve_rand_ins"]
    df = pd.DataFrame(rows)
    _save(out_dir, "faithfulness", b.name, df, {"K": b.K, "operators": ["zero", "donor"], "random_rankings_per_instance": 5})
    np.savez(os.path.join(out_dir, f"faithfulness_curves_{b.name}.npz"), **curves)
    print(df[df.operator == "zero"][["variant", "del_over_random", "ins_over_random"]].round(2).to_string(index=False), f"\n   ({time.time() - t0:.0f}s)")
    return df


# ============================================================================ 5.2a corruption ground truth
def run_groundtruth(b: ModelBundle, out_dir, n_rows=500, severities=(10, 100), seed=123):
    t0 = time.time()
    S = b.suite
    p = b.U_inst.shape[1]
    rs = np.random.RandomState(seed)
    pool = b.gt_rows
    sel = rs.choice(len(pool), n_rows, replace=False)
    X = np.asarray(pool)[sel].astype(np.float64)
    U_clean = b.gt_transform(X)
    perm = np.random.RandomState(seed + 1).permutation(n_rows)
    half = {"tuning": perm[: n_rows // 2], "evaluation": perm[n_rows // 2:]}
    names = S.available + (["KernelSHAP"] if b.run_kernelshap else []) + list(CONTROLS)
    L_clean = S.f_np(U_clean)
    rows, chance = [], 3.0 / p
    for sev in severities:
        Xc, truth = corrupt_rows(X, sev, seed=seed + sev)
        Uc = b.gt_transform(Xc)
        L_c = S.f_np(Uc)
        shift = np.abs(Uc - U_clean)[np.arange(n_rows)[:, None], truth]
        print(f"[{b.name}] 5.2 corruption x{sev}: error {L_clean.mean():.1f} -> {L_c.mean():.1f} (rises in {100 * (L_c > L_clean).mean():.0f}% of rows); "
              f"median shift of corrupted features {np.median(shift):.2f} sigma; chance precision@3 = {chance:.4f}")
        for v in names:
            A = S.explain(v, Uc, seed=0)
            _, per_row = precision_at_k(A, truth, 3, seed=seed)
            row = {"model": b.name, "variant": v, "severity": sev, "chance": chance,
                   "error_clean_mean": float(L_clean.mean()), "error_corrupt_mean": float(L_c.mean()),
                   "median_shift_sigma": float(np.median(shift))}
            for h, idx in half.items():
                row[f"precision_at_3_{h}"] = float(per_row[idx].mean())
            lo, hi = bootstrap_ci(per_row[half["evaluation"]])
            row.update({"precision_at_3_eval_ci_lo": lo, "precision_at_3_eval_ci_hi": hi, "precision_at_3_all": float(per_row.mean()),
                        "x_over_chance_eval": float(per_row[half["evaluation"]].mean() / chance)})
            rows.append(row)
        sub = pd.DataFrame([r for r in rows if r["severity"] == sev])
        print(sub[["variant", "precision_at_3_tuning", "precision_at_3_evaluation", "x_over_chance_eval"]].round(3).to_string(index=False))
    df = pd.DataFrame(rows)
    # --- decision rules R1 / R2 (see src/benchmarks/__init__.py), computed from the TUNING half at 10x only
    tun = df[df.severity == 10].set_index("variant").precision_at_3_tuning
    lam = {0.0: tun.get("LRD_lam0"), 0.5: tun.get("LRD_lam0.5"), 1.0: tun.get("LRD_lam1")}
    decisions = {}
    if all(v is not None for v in lam.values()):
        best = max(lam, key=lam.get)
        decisions["R1_lambda_best_on_tuning"] = best
        decisions["R1_lambda_chosen"] = 0.5 if lam[best] - lam[0.5] <= 0.02 else best
        decisions["R1_tuning_precision"] = lam
    if "ManifoldLIME_stochround" in tun.index:
        decisions["R2_stochround_minus_deterministic_tuning"] = float(tun["ManifoldLIME_stochround"] - tun["ManifoldLIME"])
        decisions["R2_passes_threshold_0.03"] = bool(decisions["R2_stochround_minus_deterministic_tuning"] >= 0.03)
    decisions["R3_variants_within_2x_chance_eval"] = sorted(set(
        df[(df.x_over_chance_eval < 2) & (~df.variant.isin(["Random"]))].variant))
    _save(out_dir, "groundtruth", b.name, df, {"n_rows": n_rows, "severities": list(severities), "decisions": decisions})
    print(f"   decisions: {decisions}   ({time.time() - t0:.0f}s)")
    return df, decisions


# ============================================================================ 5.2b planted-signal
def run_planted(name, U_normal, snap, snap_stochastic, out_dir, reps=3, n_eval=200, seed=0, run_kernelshap=True):
    t0 = time.time()
    p = U_normal.shape[1]
    chance = 3.0 / p
    rows = []
    for r in range(reps):
        task = make_planted_task(U_normal, seed=seed + r)
        suite = ExplainerSuite(task["model"], U_normal, snap=snap, snap_stochastic=snap_stochastic)
        ev = task["U_hold"][np.random.RandomState(seed + 50 + r).choice(len(task["U_hold"]), n_eval, replace=False)]
        truth = np.tile(task["truth"], (n_eval, 1))
        print(f"[{name}] 5.2b planted rep {r}: truth={[int(x) for x in task['truth']]}  R2={task['r2_holdout']:.3f}  planted importance share={task['planted_importance_share']:.2f}")
        for v in suite.available + (["KernelSHAP"] if run_kernelshap else []) + list(CONTROLS):
            A = suite.explain(v, ev, seed=0)
            m, per_row = precision_at_k(A, truth, 3, seed=seed)
            rows.append({"dataset": name, "rep": r, "variant": v, "precision_at_3": m, "chance": chance, "x_over_chance": m / chance,
                         "model_r2": task["r2_holdout"], "planted_importance_share": task["planted_importance_share"],
                         "twin_corr_threshold": task["twin_corr_threshold"], "n_candidates": task["n_candidates"]})
    df = pd.DataFrame(rows)
    agg = df.groupby("variant").agg(precision_at_3=("precision_at_3", "mean"), min_over_reps=("precision_at_3", "min"),
                                    max_over_reps=("precision_at_3", "max")).reset_index()
    agg["chance"] = chance
    agg["model_r2_mean"] = df.model_r2.mean()
    agg["planted_importance_share_mean"] = df.planted_importance_share.mean()
    agg["twin_corr_threshold"] = float(df.twin_corr_threshold.iloc[0])
    agg.insert(0, "dataset", name)
    _save(out_dir, "planted", name, agg, {"reps": reps, "n_eval": n_eval})
    df.to_csv(os.path.join(out_dir, f"planted_by_rep_{name}.csv"), index=False)
    print(agg.round(3).to_string(index=False), f"\n   ({time.time() - t0:.0f}s)")
    return agg


# ============================================================================ 5.3 robustness
def choose_eps_in(f_np, U, eps_f, grid=(0.05, 0.02, 0.01), seed=0, n_inst=30, n_prop=40):
    """Rule R6: largest eps_in whose proposals satisfy the OUTPUT constraint for >= 20% of draws (median instance)."""
    rs = np.random.RandomState(seed)
    idx = rs.choice(len(U), min(n_inst, len(U)), replace=False)
    for eps in grid:
        shares = []
        for i in idx:
            x = U[i:i + 1]
            d = np.clip(0.5 * eps * rs.normal(size=(n_prop, U.shape[1])), -eps, eps).astype(np.float32)
            shares.append(np.mean(np.abs(f_np(x + d) - f_np(x)[0]) <= eps_f))
        if np.median(shares) >= 0.2:
            return eps, float(np.median(shares))
    return grid[-1], float(np.median(shares))


def run_robustness(b: ModelBundle, out_dir, seed=0):
    t0 = time.time()
    S = b.suite
    rs = np.random.RandomState(seed)
    f_inst = S.f_np(b.U_inst)
    eps_f = 0.02 * float(f_inst.std())
    eps_in, feas = choose_eps_in(S.f_np, b.U_inst, eps_f, seed=seed)
    print(f"[{b.name}] 5.3 robustness: eps_f={eps_f:.4g} (2% of output std), eps_in={eps_in} (median feasible share {feas:.2f})")
    stab_names = [v for v in ["AdaptedSHAP", "AdaptedSHAP_zero", "GradientSHAP", "ManifoldLIME", "ManifoldLIME_stochround", "VanillaLIME",
                              "LRD_lam0", "LRD_lam0.5", "LRD_lam1", "NativeMask"] if v in S.available]
    adv_names = [v for v in ["AdaptedSHAP", "ManifoldLIME", "VanillaLIME", "LRD_lam0.5", "NativeMask"] if v in S.available]
    si = rs.choice(len(b.U_inst), min(b.n_stab, len(b.U_inst)), replace=False)
    ai = rs.choice(len(b.U_inst), min(b.n_adv, len(b.U_inst)), replace=False)
    rows, adv_conf = [], {}
    for v in sorted(set(stab_names + adv_names), key=lambda s: (stab_names + adv_names).index(s)):
        row = {"model": b.name, "variant": v}
        if v in stab_names:
            ns = noise_stability(S, v, b.U_inst[si], b.sigma, rel=0.01, seed=seed)
            row.update({"stability_tau_noise": float(np.nanmean(ns["tau_noise"])), "stability_tau_control": float(np.nanmean(ns["tau_control"])),
                        "stability_n": len(si)})
            row["stability_share_of_ceiling"] = row["stability_tau_noise"] / max(row["stability_tau_control"], 1e-9)
        if v in adv_names:
            ad = adversarial_search(S, v, b.U_inst[ai], eps_in=eps_in, eps_f=eps_f, iters=b.adv_iters, cands=b.adv_cands, seed=seed)
            adv_conf[v] = ad["tau_confirmed"]
            row.update({"adv_tau_search": float(np.nanmean(ad["tau_search"])), "adv_tau_confirmed": float(np.nanmean(ad["tau_confirmed"])),
                        "adv_tau_random_control": float(np.nanmean(ad["tau_random"])), "adv_top1_changed": float(np.nanmean(ad["top1_changed"])),
                        "adv_feasible_share": float(np.nanmean(ad["feasible_share"])), "adv_n": len(ai), "adv_eps_in": eps_in, "adv_eps_f": eps_f})
        rows.append(row)
        print("   " + "  ".join(f"{k}={v_:.3f}" if isinstance(v_, float) else f"{k}={v_}" for k, v_ in row.items() if k not in ("model",)))
    df = pd.DataFrame(rows)
    meta = {"eps_in": eps_in, "eps_f": eps_f, "feasible_share_at_chosen_eps": feas, "noise_rel": 0.01, "iters": b.adv_iters, "cands": b.adv_cands}
    # R4: Manifold-LIME vs vanilla LIME, paired by instance
    if "ManifoldLIME" in adv_conf and "VanillaLIME" in adv_conf:
        d = adv_conf["ManifoldLIME"] - adv_conf["VanillaLIME"]
        d = d[np.isfinite(d)]
        lo, hi = bootstrap_ci(d)
        meta["R4_confirmed_tau_manifold_minus_vanilla"] = {"mean": float(d.mean()), "ci95": [lo, hi], "n": int(len(d))}
        print(f"   R4 (higher tau = harder to fool): Manifold-LIME minus vanilla LIME = {d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  (n={len(d)})")
    _save(out_dir, "robustness", b.name, df, meta)
    print(f"   ({time.time() - t0:.0f}s)")
    return df


# ============================================================================ 5.4 runtime and memory
def run_runtime(b: ModelBundle, out_dir, seed=0):
    t0 = time.time()
    S = b.suite
    rs = np.random.RandomState(seed)
    pool = b.U_runtime
    names = S.available + (["KernelSHAP"] if b.run_kernelshap else [])
    rows = []
    for v in names:
        n = b.n_kernel_runtime if v == "KernelSHAP" else b.n_runtime
        idx = rs.choice(len(pool), n, replace=len(pool) < n)
        rows_u = pool[idx]
        call = lambda i, v=v, rows_u=rows_u: S.explain(v, rows_u[i:i + 1], seed=i)
        lat = rt.latency(call, n)
        rows.append({"model": b.name, "explainer": v, **lat})
        print(f"[{b.name}] 5.4 {v:26s} n={lat['n']:5d}  p50={lat['p50_ms']:9.2f} ms  p95={lat['p95_ms']:9.2f} ms")
    df = pd.DataFrame(rows)
    df["meets_LRD_target_50ms"] = np.where(df.explainer.str.startswith("LRD"), df.p95_ms < 50, np.nan)
    df["meets_100ms_p50"] = df.p50_ms < 100
    df["meets_100ms_p95"] = df.p95_ms < 100
    _save(out_dir, "runtime", b.name, df, {"n": b.n_runtime, "n_kernelshap": b.n_kernel_runtime,
                                           "hardware_note": "single process, CPU, nothing else running"})
    print(f"   ({time.time() - t0:.0f}s)")
    return df


# ============================================================================ 5.4 memory (fresh process per explainer)
def memory_probe(b: ModelBundle, explainer, seed=0):
    """Run INSIDE a fresh child process: peak resident-memory growth for n single-instance explanations."""
    S = b.suite
    n = min(b.n_kernel_runtime if explainer == "KernelSHAP" else b.n_mem, len(b.U_runtime))
    rows_u = b.U_runtime[np.random.RandomState(seed).choice(len(b.U_runtime), n, replace=False)]
    res = rt.rss_probe(lambda i: S.explain(explainer, rows_u[i:i + 1], seed=i), n)
    res.update({"model": b.name, "explainer": explainer})
    print("MEMPROBE " + json.dumps(res))


def run_memory(model, explainers, child_cmd, out_dir):
    """Orchestrate one fresh child process per explainer; each prints a MEMPROBE json line."""
    import subprocess
    rows = []
    for v in explainers:
        t0 = time.time()
        out = subprocess.run(child_cmd + ["--memory-child", v], capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("MEMPROBE ")]
        if not line:
            raise RuntimeError(f"memory probe failed for {model}/{v}:\n{out.stderr[-2000:]}")
        r = json.loads(line[-1][len("MEMPROBE "):])
        rows.append(r)
        print(f"[{model}] 5.4 memory {v:26s} peak resident-memory growth = {r['peak_rss_growth_mb']:7.1f} MB over {r['n']} explanations  ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    _save(out_dir, "memory", model, df, {"method": "fresh process per explainer; peak RSS growth, 2 ms sampling"})
    return df
