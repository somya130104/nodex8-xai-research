"""
Phase 4 — the Consensus Logic Index (CLI) and uncertainty gate on TradeFraud.
See ROADMAP.md Phase 4, src/consensus.py (the math) and src/consensus_analysis.py.

CLI(x) is computed for ONE model, ONE client, across that model's OWN explainers:
  Model4_Autoencoder, Model9_VAE : Adapted SHAP, Manifold-LIME, LRD   (the proposal's Tri-Explainer, K=3)
  Model7_TabNet                  : Adapted SHAP, Manifold-LIME, TabNet's native attention mask
                                   (LRD has no bottleneck to use on TabNet; the native mask takes
                                   its place, so this is a CLI VARIANT, labelled as such)
Each explainer is its PRIMARY variant, fixed in Phase 3 before any results were seen.

Data
  Calibration : the Phase 3 Validation-OOS clients (51 disputes + 200 random normal clients).
                Attributions are re-used from Phase 3, not recomputed.
  Out-of-time : internal_oot (Aug-Sep 2025) and the Oct / Nov / Dec 2025 months — per period,
                every unique dispute + 250 random normal clients (seed 42), explained here with
                the same primary explainers (cached in consensus/oot_cache/).

Gate calibration (fixed BEFORE looking at any out-of-time CLI):
  RED   (route to human review)  CLI < 10th percentile of CLI over the 200 random normal
                                 Validation clients of that model.
  AMBER (show with a caveat)     CLI < 25th percentile.  (A convention, not a research result.)
  Disputes are NOT used to calibrate: they are a 20% over-sample of a 0.2% population and
  would move the quantiles; the effect of including them is reported.

Outputs: XAI_Outputs/consensus/
  cli_instances_<model>.csv   one row per client and period: CLI, pairwise taus, flag, note
  cli_tables_<model>.csv      period / segment / decile / label tables (long format)
  cli_sanity_<model>.txt      the roadmap's high-vs-low CLI check
  model_rashomon_index.csv    cross-MODEL agreement (NOT the CLI)
  consensus_cli_summary.json  everything needed for ARCHITECTURE.md (tracked in git)
"""
import os
import sys
import json
import pickle
import time
import warnings
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append("/Users/somyabhadada/nodex8")
from run_tri_explainer_tabnet import TabNetLogOdds
from src.eval_utils import dedupe_rows
from src.autoencoder_models import Autoencoder, VAE
from src.explainers.adapted_shap import ReconstructionError
from src.explainers.manifold_lime import torch_black_box
from src.explainers.snapping import integer_columns, snap_via_preprocessor
from src.explainers.studies import primary_attributions, StudyConfig
from src.consensus import calibrate_thresholds, disputed_features, model_rashomon_index, kendalls_w
from src.consensus_analysis import cli_frame, group_table, faithfulness_link, sanity_cases

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/Modeling_Data_Enriched/"
MODEL_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/Model_Outputs/"
XAI_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/XAI_Outputs/"
P3_DIR = os.path.join(XAI_DIR, "tri_explainer")
OUT_DIR = os.path.join(XAI_DIR, "consensus")
CACHE_DIR = os.path.join(OUT_DIR, "oot_cache")
EXCLUDE_COLS = ["Client", "Client_ID", "Date", "Dispute", "Segment", "Activeness_Segment"]
SEED = 42
N_NORMAL_VAL = 200
N_NORMAL_OOT = 250
PERIODS = [("internal_oot", "internal_oot.csv"), ("oot_2025-10", "oot_202510.csv"),
           ("oot_2025-11", "oot_202511.csv"), ("oot_2025-12", "oot_202512.csv")]
PERIOD_ORDER = ["validation"] + [p for p, _ in PERIODS]

VARIANT = {"AdaptedSHAP": "ig_kmeans", "ManifoldLIME": "lime_manifold", "LRD": "lrd_lam0.5",
           "TabNetNative": "tabnet_native_mask"}
MODELS = {
    "Model4_Autoencoder": dict(kind="ae", explainers=("AdaptedSHAP", "ManifoldLIME", "LRD"), unsigned=("LRD",),
                               cli_label="Tri-Explainer CLI (K=3)"),
    "Model9_VAE":         dict(kind="vae", explainers=("AdaptedSHAP", "ManifoldLIME", "LRD"), unsigned=("LRD",),
                               cli_label="Tri-Explainer CLI (K=3)"),
    "Model7_TabNet":      dict(kind="tabnet", explainers=("AdaptedSHAP", "ManifoldLIME", "TabNetNative"),
                               unsigned=("TabNetNative",),
                               cli_label="CLI variant (K=3; TabNet attention mask in place of LRD)"),
}


def load_split(fname):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = dedupe_rows(df, feat_cols).reset_index(drop=True)
    return df, df[feat_cols].fillna(0), (df["Dispute"] == "D").astype(int).values, feat_cols


def select(y, n_normal, seed=SEED):
    """Every dispute + n_normal random normal clients — the Phase 3 selection rule."""
    rng = np.random.RandomState(seed)
    return np.concatenate([np.where(y == 1)[0], rng.choice(np.where(y == 0)[0], n_normal, replace=False)])


def build_model(name, p):
    """-> (scalar torch module, lrd model or None, lrd kind, native-mask fn or None)."""
    kind = MODELS[name]["kind"]
    if kind == "ae":
        m = Autoencoder(p, [128, 64, 32])
        m.load_state_dict(torch.load(os.path.join(MODEL_DIR, name, "model4_ae_state.pt")), strict=True)
        return ReconstructionError(m.eval(), "ae"), m, "ae", None
    if kind == "vae":
        m = VAE(p, [256, 128], 32)
        m.load_state_dict(torch.load(os.path.join(MODEL_DIR, name, "model9_vae_state.pt")), strict=True)
        return ReconstructionError(m.eval(), "vae"), m, "vae", None
    pre = pickle.load(open(os.path.join(MODEL_DIR, "Model4_Autoencoder", "preprocessor.pkl"), "rb"))
    clf = TabNetClassifier()
    clf.load_model(os.path.join(MODEL_DIR, name, "tabnet_model.zip.zip"))
    net = clf.network.eval()
    for prm in net.parameters():
        prm.requires_grad_(False)
    return TabNetLogOdds(net, pre.torch_inverse_fn()), None, None, lambda X_raw: clf.explain(X_raw)[0]


def load_phase3(name, ids):
    att = pd.read_csv(os.path.join(P3_DIR, f"attributions_{name}.csv"))
    feat = [c for c in att.columns if c not in ("Model", "Instance_ID", "True_Label", "Explainer", "Variant")]
    out = {}
    for e in MODELS[name]["explainers"]:
        a = att[att.Variant == VARIANT[e]].set_index("Instance_ID")
        out[e] = a.loc[ids, feat].values.astype(np.float64)
    return out, feat


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.time()
    print("Loading data...")
    _, X_tr, y_tr, feat_cols = load_split("train.csv")
    X_tr_normal = X_tr[y_tr == 0]
    pre = pickle.load(open(os.path.join(MODEL_DIR, "Model4_Autoencoder", "preprocessor.pkl"), "rb"))
    U_normal = pre.transform(X_tr_normal).astype(np.float32)
    snap = snap_via_preprocessor(pre, integer_columns(X_tr_normal.values))
    p = len(feat_cols)

    df_va, X_va, y_va, _ = load_split("val_oos.csv")
    val_idx = select(y_va, N_NORMAL_VAL)
    periods = {"validation": (df_va.iloc[val_idx], X_va.iloc[val_idx], y_va[val_idx])}
    for per, fname in PERIODS:
        d, X, y, fc = load_split(fname)
        assert fc == feat_cols
        idx = select(y, N_NORMAL_OOT)
        periods[per] = (d.iloc[idx], X.iloc[idx], y[idx])
        print(f"  {per:13s}: {int(y[idx].sum())} unique disputes + {N_NORMAL_OOT} random normal clients")

    summary = {"gate_definition": {"red": "CLI < 10th percentile of CLI on the 200 random normal Validation clients",
                                   "amber": "CLI < 25th percentile (convention)",
                                   "undefined_CLI": "treated as RED"},
               "tau": "weighted Kendall tau, hyperbolic additive weights (Vigna 2015), ranking by |attribution|",
               "models": {}}
    all_val_attrs = {}

    for name, spec in MODELS.items():
        print("\n" + "=" * 72 + f"\n{name}: {spec['cli_label']}\n" + "=" * 72)
        f_torch, lrd_model, lrd_kind, native_fn = build_model(name, p)
        f_np = torch_black_box(f_torch)
        expl = spec["explainers"]

        # ------------------------------------------------ calibration set = Phase 3 attributions
        d_va, Xv, yv = periods["validation"]
        ids_va = d_va["Client_ID"].values
        inst = pd.read_csv(os.path.join(P3_DIR, f"instances_{name}.csv"))
        assert list(inst["Instance_ID"]) == list(ids_va), "Phase 3 instance order differs"
        val_attrs, feat = load_phase3(name, ids_va)
        assert feat == feat_cols
        all_val_attrs[name] = val_attrs

        # reproduction check: the Phase 4 helper must rebuild Phase 3's primary explainers exactly
        U_chk = pre.transform(Xv.iloc[:6]).astype(np.float32)
        chk, _ = primary_attributions(f_torch=f_torch, U_inst=U_chk, U_normal=U_normal, snap=snap,
                                      lrd_model=lrd_model, lrd_kind=lrd_kind, cfg=StudyConfig(seed=SEED))
        repro = {e: float(np.max(np.abs(chk[e] - val_attrs[e][:6])) / (np.abs(val_attrs[e][:6]).max() + 1e-12))
                 for e in chk}
        print("  reproduction of Phase 3 attributions (max rel. diff): " +
              ", ".join(f"{e}={v:.1e}" for e, v in repro.items()))

        # ------------------------------------------------ out-of-time explanations (cached)
        frames_attrs = {"validation": val_attrs}
        U_by, scalar_by = {"validation": pre.transform(Xv).astype(np.float32)}, {}
        with torch.no_grad():
            scalar_by["validation"] = f_torch(torch.tensor(U_by["validation"])).numpy()
        for per, _ in PERIODS:
            d, X, y = periods[per]
            cp = os.path.join(CACHE_DIR, f"{name}_{per}.npz")
            U = pre.transform(X).astype(np.float32)
            if os.path.exists(cp):
                z = np.load(cp, allow_pickle=True)
                assert list(z["ids"]) == list(d["Client_ID"].values)
                A = {e: z[e] for e in expl}
            else:
                ts = time.time()
                A, _ = primary_attributions(f_torch=f_torch, U_inst=U, U_normal=U_normal, snap=snap,
                                            lrd_model=lrd_model, lrd_kind=lrd_kind, cfg=StudyConfig(seed=SEED))
                if native_fn is not None:
                    A["TabNetNative"] = native_fn(X.values.astype(np.float32))
                np.savez(cp, ids=d["Client_ID"].values, **A)
                print(f"  explained {per}: {len(U)} clients in {time.time() - ts:.0f}s")
            frames_attrs[per] = {e: np.asarray(A[e], dtype=np.float64) for e in expl}
            U_by[per] = U
            with torch.no_grad():
                scalar_by[per] = f_torch(torch.tensor(U)).numpy()

        # ------------------------------------------------ calibrate on validation normals ONLY
        val_scores = cli_frame(val_attrs, ids_va, unsigned=spec["unsigned"])
        normals = yv == 0
        th = calibrate_thresholds(val_scores["CLI"].values[normals],
                                  how="10th/25th percentile of CLI on the 200 random normal Validation clients")
        th_all = calibrate_thresholds(val_scores["CLI"].values, how="incl. disputes (sensitivity only)")
        print(f"  gate: RED if CLI < {th.red:.3f};  AMBER if CLI < {th.amber:.3f}  "
              f"(including disputes would give {th_all.red:.3f} / {th_all.amber:.3f})")

        # ------------------------------------------------ per-instance CLI everywhere
        frames = []
        for per in PERIOD_ORDER:
            d, X, y = periods[per]
            fr = cli_frame(frames_attrs[per], d["Client_ID"].values, thresholds=th,
                           feature_names=feat_cols, unsigned=spec["unsigned"])
            fr.insert(1, "Period", per)
            fr.insert(2, "True_Label", y)
            fr.insert(3, "Scalar_Value", scalar_by[per])
            fr["Activeness_Segment"] = d["Activeness_Segment"].values
            fr["Segment"] = d["Segment"].values
            frames.append(fr)
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(os.path.join(OUT_DIR, f"cli_instances_{name}.csv"), index=False)
        norm = df[df.True_Label == 0].copy()
        oot_norm = norm[norm.Period != "validation"].copy()

        # ------------------------------------------------ tables
        t_period = group_table(norm, "Period")
        t_period["order"] = t_period["Period"].map(PERIOD_ORDER.index)
        t_period = t_period.sort_values("order").drop(columns="order")
        t_label = group_table(df.assign(Label=np.where(df.True_Label == 1, "dispute", "normal")), "Label")
        t_seg = group_table(norm, "Activeness_Segment")
        # risk deciles: the model's own score, deciles over the pooled OUT-OF-TIME random normals
        oot_norm["Risk_Decile"] = pd.qcut(oot_norm["Scalar_Value"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
        t_dec = group_table(oot_norm, "Risk_Decile")
        rho_risk, p_risk = spearmanr(oot_norm["Scalar_Value"], oot_norm["CLI"], nan_policy="omit")
        tables = []
        for k, t in (("period", t_period), ("label", t_label), ("activeness_segment", t_seg), ("risk_decile", t_dec)):
            t = t.rename(columns={t.columns[0]: "group"})
            t.insert(0, "table", k)
            tables.append(t)
        pd.concat(tables, ignore_index=True).to_csv(os.path.join(OUT_DIR, f"cli_tables_{name}.csv"), index=False)
        print("\n  CLI by period (normal clients):\n" + t_period.round(3).to_string(index=False))
        print("\n  CLI disputes vs normals (all periods):\n" + t_label.round(3).to_string(index=False))
        print("\n  CLI by activeness segment (normal clients, all periods):\n" + t_seg.round(3).to_string(index=False))
        print("\n  CLI by risk decile (out-of-time normal clients; 10 = highest model score):\n"
              + t_dec.round(3).to_string(index=False))
        print(f"  Spearman(model score, CLI) = {rho_risk:.3f} (p={p_risk:.2g})")

        # ------------------------------------------------ divergence analysis on RED cases
        red = df[df.Flag == "RED"]
        disputed = {}
        for _, r in red.iterrows():
            per = r["Period"]
            i = int(np.where(periods[per][0]["Client_ID"].values == r["Instance_ID"])[0][0])
            a, b = r["Weakest_Pair"].split(" vs ")
            for f_, _ in disputed_features({a: frames_attrs[per][a][i], b: frames_attrs[per][b][i]}, (a, b),
                                           feat_cols, top_k=5):
                disputed[f_] = disputed.get(f_, 0) + 1
        top_disputed = sorted(disputed.items(), key=lambda kv: -kv[1])[:10]
        weakest_counts = red["Weakest_Pair"].value_counts().to_dict()
        weakest_all = df["Weakest_Pair"].value_counts().to_dict()
        print(f"\n  RED cases: {len(red)}; weakest pair among RED: {weakest_counts}")
        print("  most often disputed features in RED cases: " + ", ".join(f"{f} ({c})" for f, c in top_disputed))

        # ------------------------------------------------ does the gate mean anything? (indicative)
        flink = {}
        U_all = np.vstack([U_by[per] for per in PERIOD_ORDER])
        A_all = {e: np.vstack([frames_attrs[per][e] for per in PERIOD_ORDER]) for e in expl}
        donors = U_normal[np.random.RandomState(SEED).choice(len(U_normal), 5000, replace=False)]
        for op, dn in (("zero", None), ("donor", donors)):
            _, flink[op] = faithfulness_link(df["CLI"].values, A_all, f_np, U_all, donors=dn, seed=SEED)
        print("  faithfulness link (consensus top-5 replacement / random-5):")
        for op, s in flink.items():
            print(f"     {op:5s}: Spearman(CLI, ratio)={s['spearman_CLI_vs_consensus_top_over_random']:+.3f}  "
                  f"median ratio lowest-CLI quartile={s['median_ratio_lowest_CLI_quartile']:.2f}  "
                  f"vs rest={s['median_ratio_upper_three_quartiles']:.2f}")

        # ------------------------------------------------ roadmap sanity check
        txt, sanity = sanity_cases(df[df.Period == "validation"], val_attrs, feat_cols, ids_va)
        print(txt)
        with open(os.path.join(OUT_DIR, f"cli_sanity_{name}.txt"), "w") as fh:
            fh.write(f"{name}: {spec['cli_label']}\n{txt}\n")

        # ------------------------------------------------ weighting sensitivity
        rho_w, _ = spearmanr(df["CLI"], df["CLI_unweighted"], nan_policy="omit")
        pair_cols = [c for c in df.columns if c.startswith("tau_")]

        summary["models"][name] = {
            "cli_label": spec["cli_label"], "explainers": list(expl),
            "explainer_variants": {e: VARIANT[e] for e in expl},
            "reproduction_max_rel_diff_vs_phase3": repro,
            "thresholds": {"red": th.red, "amber": th.amber, "how": th.how,
                           "if_disputes_included": {"red": th_all.red, "amber": th_all.amber}},
            "n_instances": {per: int((df.Period == per).sum()) for per in PERIOD_ORDER},
            "n_disputes": {per: int(((df.Period == per) & (df.True_Label == 1)).sum()) for per in PERIOD_ORDER},
            "cli_distribution_validation_normals": {q: float(val_scores["CLI"][normals].quantile(float(q)))
                                                    for q in ("0.05", "0.1", "0.25", "0.5", "0.75", "0.9", "0.95")},
            "mean_pairwise_tau_all": {c: float(df[c].mean()) for c in pair_cols},
            "mean_companions_all": {"CLI_unweighted": float(df["CLI_unweighted"].mean()),
                                    "top5_jaccard": float(df["top5_jaccard"].mean()),
                                    "top5_sign_agreement": float(df["top5_sign_agreement"].mean())},
            "spearman_weighted_vs_unweighted_CLI": float(rho_w),
            "n_undefined_CLI": int(df["CLI"].isna().sum()),
            "tables": {"period": t_period.round(4).to_dict("records"),
                       "label": t_label.round(4).to_dict("records"),
                       "activeness_segment": t_seg.round(4).to_dict("records"),
                       "risk_decile": t_dec.round(4).to_dict("records")},
            "spearman_risk_score_vs_CLI_oot_normals": {"rho": float(rho_risk), "p": float(p_risk)},
            "frac_RED_disputes_all_periods": float((df[df.True_Label == 1].Flag == "RED").mean()),
            "red_cases": {"n": int(len(red)), "weakest_pair_counts": weakest_counts,
                          "weakest_pair_counts_all_instances": weakest_all,
                          "most_disputed_features": top_disputed},
            "faithfulness_link": flink,
            "sanity_check": sanity.round(4).to_dict("records"),
        }

    # ---------------------------------------------------------------- model_rashomon_index (NOT the CLI)
    print("\n" + "=" * 72 + "\nmodel_rashomon_index — do DIFFERENT MODELS rely on the same features?\n" + "=" * 72)
    ids_va = periods["validation"][0]["Client_ID"].values
    yv = periods["validation"][2]
    rash_rows, rash = [], {}
    for e in ("AdaptedSHAP", "ManifoldLIME"):
        per_inst = [model_rashomon_index({m: all_val_attrs[m][e][i] for m in MODELS}) for i in range(len(ids_va))]
        for i, r in enumerate(per_inst):
            row = {"Explainer": e, "Instance_ID": ids_va[i], "True_Label": yv[i], "index": r["index"]}
            row.update({f"tau_{a}_vs_{b}": t for (a, b), t in r["pairwise"].items()})
            rash_rows.append(row)
        rr = pd.DataFrame([x for x in rash_rows if x["Explainer"] == e])
        rash[f"instance_level_{e}"] = {c: float(rr[c].mean()) for c in rr.columns if c == "index" or c.startswith("tau_")}
        print(f"  instance level ({e}, 251 validation clients): mean index={rash[f'instance_level_{e}']['index']:.3f}  "
              + ", ".join(f"{c[4:]}={v:.3f}" for c, v in rash[f"instance_level_{e}"].items() if c.startswith("tau_")))
    pd.DataFrame(rash_rows).to_csv(os.path.join(OUT_DIR, "model_rashomon_index.csv"), index=False)
    g = pd.read_csv(os.path.join(XAI_DIR, "shap_global.csv")).pivot(index="Model", columns="Feature", values="Mean_Abs_SHAP")
    g = g.fillna(0.0)
    gr = model_rashomon_index({m: g.loc[m].values for m in g.index})
    rash["global_mean_abs_shap"] = {"models": list(g.index), "n_features": int(g.shape[1]),
                                    "index": gr["index"], "kendalls_w": kendalls_w(g.values),
                                    "pairwise": {f"{a}_vs_{b}": t for (a, b), t in gr["pairwise"].items()}}
    print(f"  global (Phase 1 mean |SHAP|, {len(g)} models): mean weighted tau={gr['index']:.3f}, "
          f"Kendall's W={rash['global_mean_abs_shap']['kendalls_w']:.3f}")
    summary["model_rashomon_index"] = rash
    summary["runtime_s"] = round(time.time() - t0, 1)
    json.dump(summary, open(os.path.join(OUT_DIR, "consensus_cli_summary.json"), "w"), indent=2, default=str)
    print(f"\nDone in {time.time() - t0:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
