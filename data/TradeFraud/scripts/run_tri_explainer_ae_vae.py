"""
Phase 3 — the Tri-Explainer on TradeFraud's autoencoder (Model 4) and VAE
(Model 9).  See ROADMAP.md Phase 3, src/explainers/* and src/explainers/studies.py.

What is explained: the model's reconstruction error L(x) = sum_j (x_j - x_hat_j)^2
in the preprocessed space (VAE: decoded from z = mu) — i.e. "which features
account for this client looking anomalous to the model?"

Instances: all 51 unique disputes in Validation OOS + 200 randomly drawn normal
clients (seed 42) = 251 clients, the SAME clients for every TradeFraud model, so
Phase 4 can compare explainers client by client.

PRIMARY variants (fixed in advance, not chosen from results):
    Adapted SHAP  = Integrated Gradients averaged over 10 k-means baselines
    Manifold-LIME = manifold perturbations (Ledoit-Wolf covariance, scale 0.3,
                    range clip, integer snapping)
    LRD           = lam 0.5 (equal blend of latent path and direct error)

Outputs: XAI_Outputs/tri_explainer/  (see studies.run_study)
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import torch

sys.path.append("/Users/somyabhadada/nodex8")
from src.eval_utils import dedupe_rows
from src.autoencoder_models import Autoencoder, VAE
from src.explainers.adapted_shap import ReconstructionError
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

MODELS = [
    ("Model4_Autoencoder", "ae",  lambda p: Autoencoder(p, [128, 64, 32]), "model4_ae_state.pt"),
    ("Model9_VAE",         "vae", lambda p: VAE(p, [256, 128], 32),        "model9_vae_state.pt"),
]


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
    p = len(feat_cols)

    rng = np.random.RandomState(SEED)
    disp_idx = np.where(y_va == 1)[0]
    norm_idx = rng.choice(np.where(y_va == 0)[0], N_NORMAL_EXPLAIN, replace=False)
    inst_idx = np.concatenate([disp_idx, norm_idx])
    y_inst = y_va[inst_idx]
    ids = df_va["Client_ID"].values[inst_idx]
    print(f"Explaining {len(inst_idx)} clients: {int(y_inst.sum())} disputes + {int((y_inst == 0).sum())} normal")

    int_mask = integer_columns(X_tr_normal.values)
    print(f"Integer-valued columns (snapped in Manifold-LIME): {int(int_mask.sum())} / {p}")

    for folder, kind, ctor, state_file in MODELS:
        print("\n" + "=" * 72 + f"\n{folder} ({kind.upper()})\n" + "=" * 72)
        pre = pickle.load(open(os.path.join(MODEL_DIR, folder, "preprocessor.pkl"), "rb"))
        model = ctor(p)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, folder, state_file)), strict=True)
        model.eval()

        U_normal = pre.transform(X_tr_normal)
        U_inst = pre.transform(X_va.iloc[inst_idx])
        run_study(
            model_name=folder, f_torch=ReconstructionError(model, kind),
            U_inst=U_inst, y_inst=y_inst, ids=ids, U_normal=U_normal, columns=feat_cols,
            out_dir=OUT_DIR, raw_mean_baseline=raw_mean_baseline(X_tr_normal, pre),
            snap=snap_via_preprocessor(pre, int_mask),
            lrd_model=model, lrd_kind=kind, cfg=StudyConfig(seed=SEED),
        )


if __name__ == "__main__":
    main()
