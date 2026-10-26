"""
Phase 3.5 — Adapted SHAP + Manifold-LIME on TradeFraud's TabNet (Model 7).
See ROADMAP.md Phase 3.5 and src/explainers/studies.py.

LRD is NOT applied: TabNet has no autoencoder bottleneck to back-project
through. (Adapting LRD to TabNet's attention masks or last hidden layer is the
roadmap's stretch goal and is not attempted here.) Instead TabNet's own
sparsemax attention-mask importance is carried along as a fourth, NATIVE
signal — it is the model's built-in feature selection, not an explainer in the
proposal's sense, and is compared against the two gradient/perturbation-based
explainers as a reference point.

What is explained: the dispute log-odds  logit_1 - logit_0  (pre-softmax).

Coordinate system: TabNet was trained on RAW feature values, but heavy-tailed
raw features make "realistic neighbours" ill-defined. All explainers therefore
work in the same preprocessed space used for the autoencoders
(signed-log1p -> standardise -> clip to [-5, 5]); the function explained is
TabNet evaluated at the INVERSE of that transform, so gradients flow back
through the preprocessing. Because the clip step is not invertible, this
function equals the real TabNet only where no value was clipped; the script
measures and reports how closely the two agree rather than assuming it.

Instances: the same 251 Validation-OOS clients as the autoencoder run
(51 disputes + 200 normal, seed 42).
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier

sys.path.append("/Users/somyabhadada/nodex8")
from src.eval_utils import dedupe_rows
from src.explainers.baselines import raw_mean_baseline
from src.explainers.snapping import integer_columns, snap_via_preprocessor
from src.explainers.studies import run_study, StudyConfig

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/Modeling_Data_Enriched/"
MODEL_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/Model_Outputs/"
OUT_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/XAI_Outputs/tri_explainer/"
EXCLUDE_COLS = ["Client", "Client_ID", "Date", "Dispute", "Segment", "Activeness_Segment"]
SEED = 42
N_NORMAL_EXPLAIN = 200


class TabNetLogOdds(nn.Module):
    """u (preprocessed coordinates) -> inverse preprocessing -> TabNet -> logit_1 - logit_0."""
    def __init__(self, network, inverse_fn):
        super().__init__()
        self.net, self.inv = network, inverse_fn

    def forward(self, u):
        logits, _ = self.net(self.inv(u))
        return logits[:, 1] - logits[:, 0]


def load_split(fname):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = dedupe_rows(df, feat_cols)
    return df, df[feat_cols].fillna(0), (df["Dispute"] == "D").astype(int).values, feat_cols


def main():
    print("Loading data...")
    _, X_tr, y_tr, feat_cols = load_split("train.csv")
    X_tr_normal = X_tr[y_tr == 0]
    df_va, X_va, y_va, _ = load_split("val_oos.csv")

    rng = np.random.RandomState(SEED)
    disp_idx = np.where(y_va == 1)[0]
    norm_idx = rng.choice(np.where(y_va == 0)[0], N_NORMAL_EXPLAIN, replace=False)
    inst_idx = np.concatenate([disp_idx, norm_idx])
    y_inst = y_va[inst_idx]
    ids = df_va["Client_ID"].values[inst_idx]
    X_inst_raw = X_va.iloc[inst_idx].values.astype(np.float32)

    pre = pickle.load(open(os.path.join(MODEL_DIR, "Model4_Autoencoder", "preprocessor.pkl"), "rb"))
    clf = TabNetClassifier()
    clf.load_model(os.path.join(MODEL_DIR, "Model7_TabNet", "tabnet_model.zip.zip"))
    net = clf.network.eval()
    for prm in net.parameters():
        prm.requires_grad_(False)
    f = TabNetLogOdds(net, pre.torch_inverse_fn())

    U_normal = pre.transform(X_tr_normal)
    U_inst = pre.transform(X_va.iloc[inst_idx])

    # --- how faithful is "TabNet at inverse(preprocess(x))" to the real TabNet at x? -------------
    with torch.no_grad():
        real_logits, _ = net(torch.tensor(X_inst_raw))
        real_lo = (real_logits[:, 1] - real_logits[:, 0]).numpy()
        at_u = f(torch.tensor(U_inst)).numpy()
    corr = float(np.corrcoef(real_lo, at_u)[0, 1])
    mad = float(np.mean(np.abs(real_lo - at_u)))
    print(f"Explained function vs real TabNet on the 251 clients: corr={corr:.4f}, "
          f"mean |log-odds difference|={mad:.4f} (real log-odds std={real_lo.std():.3f})")

    # --- TabNet's native sparsemax-mask importance (4th, reference signal) -------------------------
    native, _ = clf.explain(X_inst_raw)

    run_study(
        model_name="Model7_TabNet", f_torch=f, U_inst=U_inst, y_inst=y_inst, ids=ids,
        U_normal=U_normal, columns=feat_cols, out_dir=OUT_DIR,
        raw_mean_baseline=raw_mean_baseline(X_tr_normal, pre),
        snap=snap_via_preprocessor(pre, integer_columns(X_tr_normal.values)),
        lrd_model=None, cfg=StudyConfig(seed=SEED, kernelshap_n=3),
        extra_attrs={"tabnet_native_mask": ("TabNetNative", native.astype(np.float32))},
        extra_latency={"TabNet native mask (clf.explain)": lambda i: clf.explain(X_inst_raw[i:i + 1])},
        baseline_note="arithmetic mean of normal rows in RAW units, mapped through the preprocessing",
    )

    # persist the representation-fidelity numbers alongside the study summary
    import json
    sp = os.path.join(OUT_DIR, "tri_explainer_Model7_TabNet_summary.json")
    s = json.load(open(sp))
    s["explained_function_vs_real_tabnet"] = {"pearson_corr": corr, "mean_abs_logodds_diff": mad,
                                              "real_logodds_std": float(real_lo.std())}
    json.dump(s, open(sp, "w"), indent=2)


if __name__ == "__main__":
    main()
