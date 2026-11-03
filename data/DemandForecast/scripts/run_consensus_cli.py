"""
Phase 4 — the Consensus Logic Index (CLI) and uncertainty gate on DemandForecast.
See ROADMAP.md Phase 4, src/consensus.py and data/TradeFraud/scripts/run_consensus_cli.py
(the TradeFraud run, which is the main one: only the autoencoders have all three
explainers of the proposal).

  TabNet : Adapted SHAP, Manifold-LIME, TabNet's native attention mask -> CLI variant (K=3)
  LSTM   : Adapted SHAP, Manifold-LIME only                            -> two-explainer agreement (K=2;
           with one pair, "CLI" is just that pair's weighted tau — reported, but it is not a Tri-Explainer CLI)
           Computed on the 28 x 51 = 1,428 flattened columns the explainers actually produce; the
           51-feature (time-summed) version is reported alongside.
Primary variants as fixed in Phase 3.

Data
  Calibration : the Phase 3 Out-of-Sample rows (250 TabNet / first 120 for the LSTM; all are a
                random sample, so all are used). Attributions re-used from Phase 3.
  Out-of-time : the three OOT months (Dec 2012, Jan 2013, Feb 2013) — 150 random rows per month
                for TabNet, the first 60 of those for the LSTM (cost), explained here.
Gate: RED < 10th percentile, AMBER < 25th percentile of the calibration CLI (fixed before OOT).
Segments: BRAND and day type (Holiday > Promotion > Weekend > Weekday). "Deciles" are deciles
of the model's own prediction (a regression model has no risk score).
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
from pytorch_tabnet.tab_model import TabNetRegressor
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append("/Users/somyabhadada/nodex8")
import train_lstm as TL
from cohort_utils import assign_day_type
from run_tri_explainer_tabnet_lstm import TabNetValue, LSTMScore
from src.explainers.manifold_lime import torch_black_box
from src.explainers.snapping import integer_columns, snap_via_standard_scaler
from src.explainers.studies import primary_attributions, StudyConfig
from src.consensus import calibrate_thresholds, disputed_features
from src.consensus_analysis import cli_frame, group_table, faithfulness_link, sanity_cases

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/samples/"
MODEL_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/Model_Outputs/"
P3_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/XAI_Outputs/tri_explainer/"
OUT_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/XAI_Outputs/consensus/"
CACHE_DIR = os.path.join(OUT_DIR, "oot_cache")
SEED = 42
N_TABNET, N_LSTM = 250, 120
N_OOT_TABNET, N_OOT_LSTM = 150, 60
OOT = [("oot_2012-12", "oot_01_201212.csv"), ("oot_2013-01", "oot_02_201301.csv"), ("oot_2013-02", "oot_03_201302.csv")]
PERIOD_ORDER = ["validation_oos"] + [p for p, _ in OOT]
T, F_ = TL.SEQ_LEN, 51
VARIANT = {"AdaptedSHAP": "ig_kmeans", "ManifoldLIME": "lime_manifold", "TabNetNative": "tabnet_native_mask"}


def load_p3(fname, variants, ids, cols):
    att = pd.read_csv(os.path.join(P3_DIR, fname))
    return {e: att[att.Variant == VARIANT[e]].set_index("Instance_ID").loc[ids, cols].values.astype(np.float64)
            for e in variants}


def analyse(name, label, attrs_by_period, meta_by_period, scalar_by_period, U_by_period, f_np,
            feat_names, unsigned, summary, extra_level=None):
    """Same pipeline as TradeFraud: calibrate on period 0, flag everything, build the tables."""
    cal = PERIOD_ORDER[0]
    ids_cal = meta_by_period[cal]["RowNum"].values
    th = calibrate_thresholds(cli_frame(attrs_by_period[cal], ids_cal, unsigned=unsigned)["CLI"].values,
                              how="10th/25th percentile of CLI on the Phase 3 Out-of-Sample rows")
    print(f"  gate: RED if CLI < {th.red:.3f};  AMBER if CLI < {th.amber:.3f}")
    frames = []
    for per in PERIOD_ORDER:
        m = meta_by_period[per]
        fr = cli_frame(attrs_by_period[per], m["RowNum"].values, thresholds=th, feature_names=feat_names,
                       unsigned=unsigned)
        fr.insert(1, "Period", per)
        fr.insert(2, "Prediction", scalar_by_period[per])
        fr["BRAND"] = m["BRAND"].values
        fr["DayType"] = m["DayType"].values
        frames.append(fr)
    df = pd.concat(frames, ignore_index=True)
    if extra_level is not None:                     # LSTM: also the time-summed 51-feature agreement
        df["CLI_feature_level"] = np.concatenate(
            [cli_frame(extra_level[per], meta_by_period[per]["RowNum"].values, unsigned=unsigned)["CLI"].values
             for per in PERIOD_ORDER])
    df.to_csv(os.path.join(OUT_DIR, f"cli_instances_{name}.csv"), index=False)

    t_period = group_table(df, "Period")
    t_period = t_period.assign(o=t_period["Period"].map(PERIOD_ORDER.index)).sort_values("o").drop(columns="o")
    t_brand = group_table(df, "BRAND")
    t_day = group_table(df, "DayType")
    oot = df[df.Period != cal].copy()
    oot["Prediction_Decile"] = pd.qcut(oot["Prediction"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    t_dec = group_table(oot, "Prediction_Decile")
    rho, pv = spearmanr(oot["Prediction"], oot["CLI"], nan_policy="omit")
    tables = []
    for k, t in (("period", t_period), ("brand", t_brand), ("day_type", t_day), ("prediction_decile", t_dec)):
        t = t.rename(columns={t.columns[0]: "group"})
        t.insert(0, "table", k)
        tables.append(t)
    pd.concat(tables, ignore_index=True).to_csv(os.path.join(OUT_DIR, f"cli_tables_{name}.csv"), index=False)
    print("\n  CLI by period:\n" + t_period.round(3).to_string(index=False))
    print("\n  CLI by day type:\n" + t_day.round(3).to_string(index=False))
    print("\n  CLI by brand (n >= 15 shown):\n" + t_brand[t_brand.n >= 15].round(3).to_string(index=False))
    print("\n  CLI by prediction decile (out-of-time):\n" + t_dec.round(3).to_string(index=False))
    print(f"  Spearman(prediction, CLI) = {rho:.3f} (p={pv:.2g})")

    red = df[df.Flag == "RED"]
    disputed = {}
    for _, r in red.iterrows():
        per = r["Period"]
        i = int(np.where(meta_by_period[per]["RowNum"].values == r["Instance_ID"])[0][0])
        a, b = r["Weakest_Pair"].split(" vs ")
        for f_, _ in disputed_features({a: attrs_by_period[per][a][i], b: attrs_by_period[per][b][i]},
                                       (a, b), feat_names, top_k=5):
            disputed[f_] = disputed.get(f_, 0) + 1
    top_disputed = sorted(disputed.items(), key=lambda kv: -kv[1])[:10]
    print(f"\n  RED cases: {len(red)}; weakest pair among RED: {red['Weakest_Pair'].value_counts().to_dict()}")
    print("  most often disputed features in RED cases: " + ", ".join(f"{f} ({c})" for f, c in top_disputed))

    U_all = np.vstack([U_by_period[p] for p in PERIOD_ORDER])
    A_all = {e: np.vstack([attrs_by_period[p][e] for p in PERIOD_ORDER]) for e in attrs_by_period[cal]}
    _, flink = faithfulness_link(df["CLI"].values, A_all, f_np, U_all, donors=None, seed=SEED)
    print(f"  faithfulness link (zero operator): Spearman(CLI, ratio)="
          f"{flink['spearman_CLI_vs_consensus_top_over_random']:+.3f}; median ratio lowest-CLI quartile="
          f"{flink['median_ratio_lowest_CLI_quartile']:.2f} vs rest={flink['median_ratio_upper_three_quartiles']:.2f}")

    txt, sanity = sanity_cases(df[df.Period == cal], attrs_by_period[cal], feat_names, ids_cal)
    print(txt)
    with open(os.path.join(OUT_DIR, f"cli_sanity_{name}.txt"), "w") as fh:
        fh.write(f"{name}: {label}\n{txt}\n")

    pair_cols = [c for c in df.columns if c.startswith("tau_")]
    s = {"cli_label": label, "explainers": list(attrs_by_period[cal]),
         "thresholds": {"red": th.red, "amber": th.amber, "how": th.how},
         "n_instances": {p: int((df.Period == p).sum()) for p in PERIOD_ORDER},
         "mean_pairwise_tau_all": {c: float(df[c].mean()) for c in pair_cols},
         "mean_companions_all": {"CLI_unweighted": float(df["CLI_unweighted"].mean()),
                                 "top5_jaccard": float(df["top5_jaccard"].mean()),
                                 "top5_sign_agreement": float(df["top5_sign_agreement"].mean())},
         "n_undefined_CLI": int(df["CLI"].isna().sum()),
         "tables": {"period": t_period.round(4).to_dict("records"), "day_type": t_day.round(4).to_dict("records"),
                    "brand_n_ge_15": t_brand[t_brand.n >= 15].round(4).to_dict("records"),
                    "prediction_decile": t_dec.round(4).to_dict("records")},
         "spearman_prediction_vs_CLI_oot": {"rho": float(rho), "p": float(pv)},
         "red_cases": {"n": int(len(red)), "weakest_pair_counts": red["Weakest_Pair"].value_counts().to_dict(),
                       "most_disputed_features": top_disputed},
         "faithfulness_link_zero_operator": flink,
         "sanity_check": sanity.round(4).to_dict("records")}
    if extra_level is not None:
        rho_fl, _ = spearmanr(df["CLI"], df["CLI_feature_level"], nan_policy="omit")
        s["feature_level"] = {"mean_CLI": float(df["CLI_feature_level"].mean()),
                              "spearman_with_flattened_CLI": float(rho_fl)}
        print(f"  feature-level (time-summed) agreement: mean={df['CLI_feature_level'].mean():.3f}; "
              f"Spearman with flattened = {rho_fl:.3f}")
    summary["models"][name] = s


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.time()
    scaler = pickle.load(open(os.path.join(MODEL_DIR, "LSTM", "scaler.pkl"), "rb"))
    feat_cols = json.load(open(os.path.join(MODEL_DIR, "LSTM", "features.json")))
    dev = TL.clean_categoricals(pd.read_csv(os.path.join(DATA_DIR, "dev.csv"), parse_dates=["Date"]))
    X_dev = dev[feat_cols].fillna(0).values.astype(np.float64)
    U_dev = scaler.transform(X_dev).astype(np.float32)
    int_mask = integer_columns(X_dev)

    # ------------------------------------------------ rows: Phase 3 OOS selection + OOT months
    raw = {"validation_oos": pd.read_csv(os.path.join(DATA_DIR, "oos.csv"), parse_dates=["Date"])}
    idx = {"validation_oos": np.random.RandomState(SEED).choice(len(raw["validation_oos"]), N_TABNET, replace=False)}
    for per, fname in OOT:
        raw[per] = pd.read_csv(os.path.join(DATA_DIR, fname), parse_dates=["Date"])
        idx[per] = np.random.RandomState(SEED).choice(len(raw[per]), N_OOT_TABNET, replace=False)
    rows, meta, Xr, U = {}, {}, {}, {}
    for per in PERIOD_ORDER:
        clean = TL.clean_categoricals(raw[per])
        rows[per] = clean.iloc[idx[per]].copy()
        meta[per] = pd.DataFrame({"RowNum": rows[per]["RowNum"].values, "BRAND": raw[per].iloc[idx[per]]["BRAND"].values,
                                  "DayType": assign_day_type(raw[per].iloc[idx[per]]).values})
        Xr[per] = rows[per][feat_cols].fillna(0).values.astype(np.float32)
        U[per] = scaler.transform(Xr[per].astype(np.float64)).astype(np.float32)
    p3_ids = pd.read_csv(os.path.join(P3_DIR, "instances_DemandForecast_TabNet.csv"))["Instance_ID"].values
    assert list(p3_ids) == list(meta["validation_oos"]["RowNum"]), "Phase 3 instance order differs"

    summary = {"tau": "weighted Kendall tau, hyperbolic additive weights (Vigna 2015), ranking by |attribution|",
               "gate_definition": {"red": "CLI < 10th percentile on the Phase 3 OOS rows", "amber": "< 25th percentile",
                                   "undefined_CLI": "treated as RED"}, "models": {}}

    # ================================================ TabNet
    print("=" * 72 + "\nDemandForecast TabNet: CLI variant (K=3; native mask in place of LRD)\n" + "=" * 72)
    reg = TabNetRegressor()
    reg.load_model(os.path.join(MODEL_DIR, "TabNet", "tabnet_model.zip.zip"))
    net = reg.network.eval()
    for prm in net.parameters():
        prm.requires_grad_(False)
    f_tab = TabNetValue(net, scaler.mean_, scaler.scale_)
    snap = snap_via_standard_scaler(scaler, int_mask)
    expl = ("AdaptedSHAP", "ManifoldLIME", "TabNetNative")
    A_tab = {"validation_oos": load_p3("attributions_DemandForecast_TabNet.csv", expl, p3_ids, feat_cols)}
    chk, _ = primary_attributions(f_torch=f_tab, U_inst=U["validation_oos"][:6], U_normal=U_dev, snap=snap,
                                  cfg=StudyConfig(seed=SEED))
    repro_tab = {e: float(np.max(np.abs(chk[e] - A_tab["validation_oos"][e][:6])) /
                          (np.abs(A_tab["validation_oos"][e][:6]).max() + 1e-12)) for e in chk}
    print("  reproduction of Phase 3 (max rel. diff): " + ", ".join(f"{e}={v:.1e}" for e, v in repro_tab.items()))
    for per, _ in OOT:
        cp = os.path.join(CACHE_DIR, f"TabNet_{per}.npz")
        if os.path.exists(cp):
            z = np.load(cp)
            A = {e: z[e] for e in expl}
        else:
            ts = time.time()
            A, _ = primary_attributions(f_torch=f_tab, U_inst=U[per], U_normal=U_dev, snap=snap, cfg=StudyConfig(seed=SEED))
            A["TabNetNative"] = reg.explain(Xr[per])[0]
            np.savez(cp, **A)
            print(f"  explained {per}: {len(U[per])} rows in {time.time() - ts:.0f}s")
        A_tab[per] = {e: np.asarray(A[e], dtype=np.float64) for e in expl}
    with torch.no_grad():
        sc_tab = {per: f_tab(torch.tensor(U[per])).numpy() for per in PERIOD_ORDER}
    analyse("DemandForecast_TabNet", "CLI variant (K=3; TabNet attention mask in place of LRD)",
            A_tab, meta, sc_tab, U, torch_black_box(f_tab), feat_cols, ("TabNetNative",), summary)
    summary["models"]["DemandForecast_TabNet"]["reproduction_max_rel_diff_vs_phase3"] = repro_tab

    # ================================================ LSTM
    print("\n" + "=" * 72 + "\nDemandForecast LSTM: two-explainer agreement (K=2)\n" + "=" * 72)
    full = TL.clean_categoricals(pd.read_csv(TL.FULL_DATA_PATH, parse_dates=["Date"]))
    min_date = full["Date"].min()
    full["_day_idx"] = (full["Date"] - min_date).dt.days
    lookup = TL.build_day_lookup(full, feat_cols, scaler)
    dev["_day_idx"] = (dev["Date"] - min_date).dt.days
    X_dev_seq, _ = TL.build_sequences(dev, lookup, F_)
    U_seq_ref = X_dev_seq.reshape(len(X_dev_seq), -1)
    lstm = TL.DemandLSTM(input_dim=F_, hidden_dim=64, num_layers=2, dropout=0.2)
    lstm.load_state_dict(torch.load(os.path.join(MODEL_DIR, "LSTM", "best_lstm.pt")), strict=True)
    lstm.eval()
    for prm in lstm.parameters():
        prm.requires_grad_(False)
    f_lstm = LSTMScore(lstm, T, F_)
    cols = [f"{feat_cols[k]}@t-{T - 1 - j}" for j in range(T) for k in range(F_)]
    n_l = {"validation_oos": N_LSTM, **{per: N_OOT_LSTM for per, _ in OOT}}
    U_l, meta_l = {}, {}
    for per in PERIOD_ORDER:
        sub = rows[per].iloc[:n_l[per]].copy()
        sub["_day_idx"] = (sub["Date"] - min_date).dt.days
        Xs, _ = TL.build_sequences(sub, lookup, F_)
        U_l[per] = Xs.reshape(len(Xs), -1).astype(np.float32)
        meta_l[per] = meta[per].iloc[:n_l[per]].reset_index(drop=True)
    expl_l = ("AdaptedSHAP", "ManifoldLIME")
    A_l = {"validation_oos": load_p3("attributions_DemandForecast_LSTM.csv", expl_l, p3_ids[:N_LSTM], cols)}
    cfg_l = StudyConfig(seed=SEED, lime_samples=1500)
    for per, _ in OOT:
        cp = os.path.join(CACHE_DIR, f"LSTM_{per}.npz")
        if os.path.exists(cp):
            z = np.load(cp)
            A = {e: z[e] for e in expl_l}
        else:
            ts = time.time()
            A, _ = primary_attributions(f_torch=f_lstm, U_inst=U_l[per], U_normal=U_seq_ref,
                                        snap=snap_via_standard_scaler(scaler, int_mask, tile=T), cfg=cfg_l)
            np.savez(cp, **A)
            print(f"  explained {per}: {len(U_l[per])} rows in {time.time() - ts:.0f}s")
        A_l[per] = {e: np.asarray(A[e], dtype=np.float64) for e in expl_l}
    feat_level = {per: {e: a.reshape(len(a), T, F_).sum(axis=1) for e, a in A_l[per].items()} for per in PERIOD_ORDER}
    with torch.no_grad():
        sc_l = {per: f_lstm(torch.tensor(U_l[per])).numpy() for per in PERIOD_ORDER}
    analyse("DemandForecast_LSTM", "two-explainer agreement (K=2; not a Tri-Explainer CLI)",
            A_l, meta_l, sc_l, U_l, torch_black_box(f_lstm), cols, (), summary, extra_level=feat_level)

    summary["runtime_s"] = round(time.time() - t0, 1)
    json.dump(summary, open(os.path.join(OUT_DIR, "consensus_cli_summary.json"), "w"), indent=2, default=str)
    print(f"\nDone in {time.time() - t0:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
