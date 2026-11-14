"""
5.5  Assemble the final benchmark matrix from the stage outputs (the "Comprehensive Benchmark Evaluation Suite").

Reads every benchmark_<stage>_<model>_summary.json written by src/benchmarks/pipeline.py in both datasets' benchmark
folders and writes, to data/Benchmark_Suite/:
  benchmark_matrix.csv          one row per (model, explainer): faithfulness, ground truth, stability, adversarial, latency, memory
  decisions.json                the pre-registered decision rules R1-R4 evaluated on the results
  BENCHMARK_MATRIX.md           the matrix and the decision readout as markdown (the written analysis lives in ARCHITECTURE.md)
  plots/*.png                   deletion/insertion curves, ground-truth precision, latency

Run:  python3 -m src.benchmarks.assemble
"""
import glob
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/somyabhadada/nodex8/data/"
DIRS = {"TradeFraud": ROOT + "TradeFraud/XAI_Outputs/benchmarks/", "DemandForecast": ROOT + "DemandForecast/XAI_Outputs/benchmarks/"}
OUT = ROOT + "Benchmark_Suite/"
MODELS = [("Model4_Autoencoder", "TradeFraud", "Autoencoder"), ("Model9_VAE", "TradeFraud", "VAE"),
          ("Model7_TabNet", "TradeFraud", "TabNet (fraud)"), ("DemandForecast_TabNet", "DemandForecast", "TabNet (demand)"),
          ("DemandForecast_LSTM", "DemandForecast", "LSTM (demand)")]
NICE = {"AdaptedSHAP": "Adapted SHAP (IG, 10 refs)", "AdaptedSHAP_zero": "Adapted SHAP (IG, zero ref)", "GradientSHAP": "GradientSHAP",
        "ManifoldLIME": "Manifold-LIME", "ManifoldLIME_stochround": "Manifold-LIME (stochastic rounding)", "VanillaLIME": "Vanilla LIME",
        "LRD_lam0": "LRD (lambda 0)", "LRD_lam0.5": "LRD (lambda 0.5)", "LRD_lam1": "LRD (lambda 1)", "NativeMask": "TabNet native mask",
        "KernelSHAP": "KernelSHAP (reference)", "Random": "Random ranking (control)", "AbsValue": "Largest |z| (control)"}
ORDER = list(NICE)
TEAL, AMBER, RED, GRAY, INK, MINT = "#0B6E69", "#F2A541", "#D1495B", "#6B7787", "#1B2430", "#3FA796"


def load(stage, model, ds):
    f = f"{DIRS[ds]}benchmark_{stage}_{model}_summary.json"
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    return pd.DataFrame(j["rows"]), j.get("settings", {})


def build():
    rows = []
    decisions = {"groundtruth": {}, "robustness_R4": {}}
    for model, ds, nice in MODELS:
        base = {}
        faith = load("faithfulness", model, ds)
        gt = load("groundtruth", model, ds)
        rob = load("robustness", model, ds)
        run = load("runtime", model, ds)
        mem = load("memory", model, ds)
        planted = load("planted", ds, ds)
        names = set()
        for t in (faith, gt, rob, run, mem):
            if t is not None:
                names |= set(t[0].variant if "variant" in t[0] else t[0].explainer)
        for v in sorted(names, key=lambda s: ORDER.index(s) if s in ORDER else 99):
            r = {"model": nice, "model_key": model, "explainer": v, "explainer_name": NICE.get(v, v)}
            if faith is not None:
                f = faith[0][faith[0].variant == v]
                for op in ("zero", "donor"):
                    x = f[f.operator == op]
                    if len(x):
                        r[f"deletion_x_random_{op}"] = float(x.del_over_random.iloc[0])
                        r[f"insertion_x_random_{op}"] = float(x.ins_over_random.iloc[0])
                        r[f"deletion_win_rate_{op}"] = float(x.del_win_rate.iloc[0])
            if gt is not None:
                g = gt[0][gt[0].variant == v]
                for sev in (10, 100):
                    x = g[g.severity == sev]
                    if len(x):
                        r[f"groundtruth_p3_x{sev}_eval"] = float(x.precision_at_3_evaluation.iloc[0])
                        r[f"groundtruth_p3_x{sev}_ci"] = f"[{x.precision_at_3_eval_ci_lo.iloc[0]:.2f}, {x.precision_at_3_eval_ci_hi.iloc[0]:.2f}]"
                        r[f"groundtruth_p3_x{sev}_tuning"] = float(x.precision_at_3_tuning.iloc[0])
                        r["groundtruth_chance"] = float(x.chance.iloc[0])
            if planted is not None:
                x = planted[0][planted[0].variant == v]
                if len(x):
                    r["planted_p3_mean"] = float(x.precision_at_3.iloc[0])
                    r["planted_p3_min"] = float(x.min_over_reps.iloc[0])
                    r["planted_p3_max"] = float(x.max_over_reps.iloc[0])
                    r["planted_chance"] = float(x.chance.iloc[0])
            if rob is not None:
                x = rob[0][rob[0].variant == v]
                if len(x):
                    for c in ("stability_tau_noise", "stability_tau_control", "adv_tau_confirmed", "adv_tau_random_control", "adv_top1_changed", "adv_feasible_share"):
                        if c in x and pd.notna(x[c].iloc[0]):
                            r[c] = float(x[c].iloc[0])
            if run is not None:
                x = run[0][run[0].explainer == v]
                if len(x):
                    r["latency_p50_ms"], r["latency_p95_ms"], r["latency_n"] = float(x.p50_ms.iloc[0]), float(x.p95_ms.iloc[0]), int(x.n.iloc[0])
                    r["meets_100ms_p95"] = bool(x.p95_ms.iloc[0] < 100)
                    if v.startswith("LRD"):
                        r["meets_LRD_50ms_p95"] = bool(x.p95_ms.iloc[0] < 50)
            if mem is not None:
                x = mem[0][mem[0].explainer == v]
                if len(x):
                    r["peak_memory_growth_mb"] = float(x.peak_rss_growth_mb.iloc[0])
            rows.append(r)
        if gt is not None:
            decisions["groundtruth"][model] = gt[1].get("decisions", {})
        if rob is not None and "R4_confirmed_tau_manifold_minus_vanilla" in rob[1]:
            decisions["robustness_R4"][model] = rob[1]["R4_confirmed_tau_manifold_minus_vanilla"]
    df = pd.DataFrame(rows)

    # ---- global decisions R1 / R2 across AE and VAE
    g = decisions["groundtruth"]
    ae, vae = g.get("Model4_Autoencoder", {}), g.get("Model9_VAE", {})
    if "R1_lambda_chosen" in ae and "R1_lambda_chosen" in vae:
        same = ae["R1_lambda_chosen"] == vae["R1_lambda_chosen"]
        decisions["R1_final"] = {"autoencoder": ae["R1_lambda_chosen"], "vae": vae["R1_lambda_chosen"],
                                 "project_lambda": ae["R1_lambda_chosen"] if same else 0.5,
                                 "rule": "same choice on both -> adopt it; otherwise keep 0.5 and report the disagreement"}
    if "R2_passes_threshold_0.03" in ae and "R2_passes_threshold_0.03" in vae:
        both = bool(ae["R2_passes_threshold_0.03"] and vae["R2_passes_threshold_0.03"])
        decisions["R2_final"] = {"autoencoder_gain": ae["R2_stochround_minus_deterministic_tuning"], "vae_gain": vae["R2_stochround_minus_deterministic_tuning"],
                                 "recommend_adopting_stochastic_rounding": both,
                                 "rule": "recommend only if tuning-half precision@3 improves by >= 0.03 on BOTH autoencoder and VAE"}
    return df, decisions


def plots(df):
    os.makedirs(OUT + "plots", exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "figure.dpi": 130, "axes.edgecolor": "#C9D1DB"})
    # --- deletion / insertion curves
    shown = ["AdaptedSHAP", "ManifoldLIME", "LRD_lam0.5", "NativeMask", "Random", "AbsValue"]
    colors = {"AdaptedSHAP": TEAL, "ManifoldLIME": AMBER, "LRD_lam0.5": RED, "NativeMask": MINT, "Random": GRAY, "AbsValue": INK}
    for op in ("zero", "donor"):
        fig, axs = plt.subplots(2, len(MODELS), figsize=(3.6 * len(MODELS), 6), squeeze=False)
        for j, (model, ds, nice) in enumerate(MODELS):
            f = f"{DIRS[ds]}faithfulness_curves_{model}.npz"
            if not os.path.exists(f):
                continue
            z = np.load(f)
            for i, kind in enumerate(("del", "ins")):
                ax = axs[i, j]
                for v in shown:
                    k = f"{v}|{op}|{kind}"
                    if k in z:
                        ax.plot(range(1, len(z[k]) + 1), z[k], color=colors[v], lw=2 if v in ("Random", "AbsValue") else 1.8,
                                ls="--" if v in ("Random", "AbsValue") else "-", label=NICE[v].replace(" (control)", ""))
                ax.set_title(f"{nice}: {'deletion' if kind == 'del' else 'insertion'}", fontsize=10, loc="left", fontweight="bold")
                ax.set_xlabel("features changed (most important first)", fontsize=8)
                ax.set_ylabel("mean |output change|", fontsize=8)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(color="#E3E8EE")
        axs[0, 0].legend(frameon=False, fontsize=7)
        fig.suptitle(f"Deletion / insertion curves, reference = {'typical client (zeros)' if op == 'zero' else 'random real client (donor)'}; higher = more faithful", x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT + f"plots/deletion_insertion_{op}.png")
        plt.close(fig)
    # --- ground-truth precision@3
    parts = []
    for model, label in (("Autoencoder", "Autoencoder\n(10x)"), ("VAE", "VAE\n(10x)")):
        d = df[df.model == model]
        parts.append((label, d.set_index("explainer")["groundtruth_p3_x10_eval"].dropna() if "groundtruth_p3_x10_eval" in d else pd.Series(dtype=float)))
    for model, label in (("TabNet (fraud)", "Planted MLP\n(fraud space)"), ("TabNet (demand)", "Planted MLP\n(demand space)")):
        d = df[df.model == model]
        parts.append((label, d.set_index("explainer")["planted_p3_mean"].dropna() if "planted_p3_mean" in d else pd.Series(dtype=float)))
    names = [v for v in ORDER if any(v in s.index for _, s in parts)]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    w = 0.2
    cols = [TEAL, MINT, AMBER, RED]
    for k, (label, s) in enumerate(parts):
        ax.bar(np.arange(len(names)) + (k - 1.5) * w, [s.get(v, np.nan) for v in names], w, label=label.replace("\n", " "), color=cols[k])
    ax.set_xticks(range(len(names)), [NICE[v].replace(" (control)", "").replace(" (reference)", "") for v in names], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("precision@3", fontsize=10)
    ax.set_title("Ground truth: do the explainers find the features that are known to be responsible?", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E3E8EE")
    fig.tight_layout()
    fig.savefig(OUT + "plots/groundtruth_precision.png")
    plt.close(fig)
    # --- latency
    fig, axs = plt.subplots(1, len(MODELS), figsize=(3.8 * len(MODELS), 4.6), sharex=False)
    for ax, (model, ds, nice) in zip(np.atleast_1d(axs), MODELS):
        d = df[(df.model == nice)].dropna(subset=["latency_p50_ms"]) if "latency_p50_ms" in df else pd.DataFrame()
        if not len(d):
            continue
        d = d.sort_values("latency_p50_ms")
        ax.barh(d.explainer_name, d.latency_p50_ms, color=TEAL)
        ax.errorbar(d.latency_p50_ms, range(len(d)), xerr=[np.zeros(len(d)), (d.latency_p95_ms - d.latency_p50_ms).clip(lower=0)], fmt="none", ecolor=INK, capsize=2, lw=1)
        ax.set_xscale("log")
        ax.axvline(100, color=RED, ls="--", lw=1)
        ax.axvline(50, color=AMBER, ls="--", lw=1)
        ax.set_title(nice, loc="left", fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("ms per instance (median; bar = p95)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Latency (red: 100 ms system target, amber: 50 ms LRD target)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT + "plots/latency.png")
    plt.close(fig)


def fmt(x, nd=2):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x))


def markdown(df, decisions):
    L = ["# Benchmark matrix (Phase 5)", "",
         "Generated by `python3 -m src.benchmarks.assemble` from the stage outputs. The written analysis is in ARCHITECTURE.md; the decision rules these",
         "results were judged by are fixed in `src/benchmarks/__init__.py` (committed before any result existed).", "",
         "**How to read the columns.** Faithfulness: output change when the explainer's top features are deleted / inserted, as a multiple of a random ranking",
         "(1.0 = no better than random) under two references (typical client = *zero*, random real client = *donor*). Ground truth: precision@3 on 250 held-out",
         "evaluation rows (95% bootstrap range); chance = 3/p. *Planted* = precision@3 on a planted-signal MLP with known drivers (dataset-level, 3 replicates).",
         "Stability: weighted tau between the explanation before and after 1% noise (control = same input, new random seed). Adversarial: lowest weighted tau",
         "found by search under an output-preserving perturbation, re-checked under a new seed (higher = harder to fool; control = random perturbation of the same size).", ""]
    for model, ds, nice in MODELS:
        d = df[df.model == nice]
        if d.empty:
            continue
        L += [f"## {nice}", ""]
        cols = [("explainer_name", "Explainer")]
        for c, h in (("deletion_x_random_zero", "Deletion ×rand (zero)"), ("deletion_x_random_donor", "Deletion ×rand (donor)"),
                     ("insertion_x_random_zero", "Insertion ×rand (zero)"), ("insertion_x_random_donor", "Insertion ×rand (donor)"),
                     ("groundtruth_p3_x10_eval", "P@3 corrupt ×10"), ("groundtruth_p3_x100_eval", "P@3 corrupt ×100"), ("planted_p3_mean", "P@3 planted"),
                     ("stability_tau_noise", "Stability τ (noise)"), ("stability_tau_control", "τ ceiling"),
                     ("adv_tau_confirmed", "Adversarial τ"), ("adv_tau_random_control", "τ random ctrl"),
                     ("latency_p50_ms", "p50 ms"), ("latency_p95_ms", "p95 ms"), ("peak_memory_growth_mb", "Mem MB")):
            if c in d and d[c].notna().any():
                cols.append((c, h))
        t = d[[c for c, _ in cols]].copy()
        t.columns = [h for _, h in cols]
        for c in t.columns[1:]:
            t[c] = t[c].map(lambda x: fmt(x, 1 if c in ("p50 ms", "p95 ms", "Mem MB") else 2))
        L += ["| " + " | ".join(t.columns) + " |", "|" + "|".join("---" for _ in t.columns) + "|"]
        L += ["| " + " | ".join(map(str, r)) + " |" for r in t.itertuples(index=False)]
        chance = d["groundtruth_chance"].dropna()
        pc = d["planted_chance"].dropna()
        notes = []
        if len(chance):
            notes.append(f"ground-truth chance = {chance.iloc[0]:.4f}")
        if len(pc):
            notes.append(f"planted chance = {pc.iloc[0]:.4f}")
        L += ["", "(" + "; ".join(notes) + ")" if notes else "", ""]
    L += ["## Pre-registered decisions", "", "```", json.dumps(decisions, indent=2, default=float), "```", ""]
    ph = DIRS["TradeFraud"] + "benchmark_posthoc_manifold_lime_summary.json"
    if os.path.exists(ph):
        L += ["## Post-hoc diagnostic (NOT pre-registered): why is Manifold-LIME below vanilla LIME on the corruption ground truth?", "",
              "Autoencoder, corruption x10, all 500 rows, precision@3 with 95% range; one factor changed at a time.", "",
              "| perturbation mode | scale | precision@3 | 95% range |", "|---|---|---|---|"]
        for r in json.load(open(ph))["rows"]:
            L.append(f"| {r['mode']} | {r['scale']} | {r['precision_at_3']:.3f} | [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] |")
        L.append("")
    cg = DIRS["TradeFraud"] + "benchmark_cli_vs_groundtruth_summary.json"
    if os.path.exists(cg):
        L += ["## Does a low CLI mean a wrong explanation? (additional analysis; hypothesis fixed before running)", "",
              "Corruption x10, 500 clients per model; CLI of the three primary explainers vs whether the consensus top-3 contains the corrupted features.", "",
              "| model | consensus precision@3 | Spearman(CLI, precision) | by CLI quartile (low→high) | RED (n) | AMBER (n) | GREEN (n) |", "|---|---|---|---|---|---|---|"]
        for m, r in json.load(open(cg))["models"].items():
            q = " / ".join(f"{v:.2f}" for v in r["consensus_precision_by_CLI_quartile"].values())
            f, n = r["consensus_precision_by_flag"], r["flag_counts"]
            L.append(f"| {m} | {r['consensus_precision_overall']:.3f} | {r['spearman_CLI_vs_consensus_precision']:+.3f} | {q} | "
                     f"{f['RED']:.2f} ({n['RED']}) | {f['AMBER']:.2f} ({n['AMBER']}) | {f['GREEN']:.2f} ({n['GREEN']}) |")
        L.append("")
    return "\n".join(L)


def main():
    os.makedirs(OUT, exist_ok=True)
    df, decisions = build()
    df.to_csv(OUT + "benchmark_matrix.csv", index=False)
    json.dump(decisions, open(OUT + "decisions.json", "w"), indent=2, default=float)
    open(OUT + "BENCHMARK_MATRIX.md", "w").write(markdown(df, decisions))
    plots(df)
    print(f"wrote {len(df)} rows -> {OUT}")


if __name__ == "__main__":
    main()
