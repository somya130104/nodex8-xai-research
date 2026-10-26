"""
Sanity viewer for the Tri-Explainer outputs (ROADMAP.md Phase 3, "how to know
Phase 3 is done").

Usage:   python3 data/TradeFraud/scripts/show_tri_explainer_case.py [Model4_Autoencoder|Model9_VAE|Model7_TabNet]

Two parts:

1. A QUANTITATIVE check. For every explainer variant: across all explained
   clients, how unusual are its top-5 features? "Unusual" = mean |z| of the
   feature's value in the preprocessed space (0 = typical normal client;
   larger = further from typical). An explainer pointing at genuinely
   anomalous features should score well above the all-feature average; a
   random ranking scores about the all-feature average by construction.
   NOTE this check favours attributions proportional to x (IG from the origin,
   LRD, LIME contribution), because those are zero for features at their
   typical value; it is a sanity check, not a faithfulness benchmark.

2. Three worked EXAMPLES (the highest-scalar dispute, a mid-scalar dispute, a
   typical normal client): the top-5 features chosen by each primary explainer
   side by side, each with the client's raw value and where it falls among the
   normal training population (percentile).
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

sys.path.append("/Users/somyabhadada/nodex8")
from src.eval_utils import dedupe_rows

warnings.filterwarnings("ignore")

DATA_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/Modeling_Data_Enriched/"
MODEL_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/Model_Outputs/"
OUT_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/XAI_Outputs/tri_explainer/"
EXCLUDE_COLS = ["Client", "Client_ID", "Date", "Dispute", "Segment", "Activeness_Segment"]
META = ["Model", "Instance_ID", "True_Label", "Explainer", "Variant"]
PRIMARY = ["ig_kmeans", "lime_manifold", "lrd_lam0.5", "tabnet_native_mask"]
K = 5


def load_split(fname):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    feat = [c for c in df.columns if c not in EXCLUDE_COLS]
    for c in feat:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = dedupe_rows(df, feat)
    return df, df[feat].fillna(0), feat


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Model4_Autoencoder"
    att = pd.read_csv(os.path.join(OUT_DIR, f"attributions_{model}.csv"))
    inst = pd.read_csv(os.path.join(OUT_DIR, f"instances_{model}.csv"))
    feat = [c for c in att.columns if c not in META]

    # normal training rows (after the same de-duplication) for percentile context
    full_tr, X_tr_all, _ = load_split("train.csv")
    X_norm = X_tr_all[(full_tr["Dispute"] == "N").values]
    df_va, X_va, _ = load_split("val_oos.csv")
    X_inst = X_va.set_index(df_va["Client_ID"].values).loc[inst["Instance_ID"].values, feat]

    pre = pickle.load(open(os.path.join(MODEL_DIR, "Model4_Autoencoder", "preprocessor.pkl"), "rb"))
    U = pd.DataFrame(pre.transform(X_inst), columns=feat, index=inst["Instance_ID"].values)
    all_mean_abs_z = float(np.abs(U.values).mean())

    # ----------------------------------------------------------- part 1
    print(f"\n=== {model}: how unusual are each explainer's top-{K} features? ===")
    print(f"(mean |z| over ALL features of the explained clients = {all_mean_abs_z:.3f}; random ranking ~ this)\n")
    rows = []
    for v in att["Variant"].unique():
        A = att[att.Variant == v].set_index("Instance_ID").loc[inst["Instance_ID"].values, feat].values
        top = np.argsort(-np.abs(A), axis=1)[:, :K]
        z = np.abs(U.values)[np.arange(len(A))[:, None], top].mean()
        rows.append({"variant": v, "explainer": att[att.Variant == v]["Explainer"].iloc[0],
                     "mean_abs_z_of_top5": round(float(z), 3), "ratio_vs_all_features": round(float(z) / all_mean_abs_z, 2)})
    print(pd.DataFrame(rows).sort_values("mean_abs_z_of_top5", ascending=False).to_string(index=False))

    # ----------------------------------------------------------- part 2
    d = inst[inst.True_Label == 1].sort_values("Scalar_Value", ascending=False)
    n = inst[inst.True_Label == 0].sort_values("Scalar_Value")
    picks = [("highest-scalar DISPUTE", d.iloc[0]), ("mid-scalar DISPUTE", d.iloc[len(d) // 2]),
             ("typical NORMAL client", n.iloc[len(n) // 2])]
    variants = [v for v in PRIMARY if v in set(att.Variant)]
    for title, row in picks:
        cid = row["Instance_ID"]
        print(f"\n--- {title}: {cid}  (label={int(row['True_Label'])}, scalar={row['Scalar_Value']:.2f}) ---")
        for v in variants:
            a = att[(att.Variant == v) & (att.Instance_ID == cid)][feat].iloc[0].values
            top = np.argsort(-np.abs(a))[:K]
            print(f"  {v}:")
            for j in top:
                f = feat[j]
                raw = X_inst.loc[cid, f]
                pct = float((X_norm[f].values <= raw).mean() * 100)
                print(f"     {f:32s} attr={a[j]:+9.3f}   raw value={raw:12.2f}   (higher than {pct:5.1f}% of normal clients)")


if __name__ == "__main__":
    main()
