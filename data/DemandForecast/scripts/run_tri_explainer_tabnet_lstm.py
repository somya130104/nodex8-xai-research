"""
Phase 3.5 — Adapted SHAP + Manifold-LIME on DemandForecast's TabNet and the
real-sequence LSTM.  See ROADMAP.md Phase 3.5 and src/explainers/studies.py.

LRD is NOT applied to either model (no autoencoder bottleneck). TabNet's own
sparsemax-mask importance is carried along as a native reference signal.

What is explained
  TabNet : the predicted demand (its regression output).
  LSTM   : the PRE-ReLU score of its output head. The deployed model clamps
           predictions at zero with a final ReLU; explaining the clamped value
           would make every explanation exactly zero for any client the model
           predicts <= 0 (zero gradient). The pre-ReLU score is the quantity the
           ReLU is applied to, and equals the model's prediction wherever that is
           positive.

Coordinate system: standardised features (the LSTM's saved StandardScaler,
fit on DEV only — also used for TabNet so both live in one space). For the
LSTM the 28-day window is FLATTENED to 28 x 51 = 1,428 columns
("<feature>@t-<k>"), explained there, and then also aggregated back to the 51
features (summing signed attributions over time) for feature-level ranking.

Instances: 250 randomly drawn Out-of-Sample rows (seed 42) for TabNet; the first
120 of those for the LSTM (Manifold-LIME on 1,428 dimensions is ~25x costlier
per instance — a compute-driven scale-down, not a methodological one).

Note: no "realistic-neighbour" claim is made for the binary/calendar flags
beyond integer snapping — many DemandForecast columns are discrete.
"""
import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append("/Users/somyabhadada/nodex8")
import train_lstm as TL                                    # DemandLSTM + real-sequence builders
from cohort_utils import assign_day_type
from src.explainers.snapping import integer_columns, snap_via_standard_scaler
from src.explainers.studies import run_study, StudyConfig
from src.explainers.validation import pairwise_agreement

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/samples/"
MODEL_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/Model_Outputs/"
OUT_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/XAI_Outputs/tri_explainer/"
SEED = 42
N_TABNET, N_LSTM = 250, 120
T, F_ = TL.SEQ_LEN, 51


class TabNetValue(nn.Module):
    """u (standardised) -> raw -> TabNet regression output."""
    def __init__(self, network, mean, scale):
        super().__init__()
        self.net = network
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, u):
        out, _ = self.net(u * self.scale + self.mean)
        return out[:, 0]


class LSTMScore(nn.Module):
    """Flattened (B, T*F) standardised window -> LSTM -> PRE-ReLU head output."""
    def __init__(self, model, t, f):
        super().__init__()
        self.m, self.t, self.f = model, t, f

    def forward(self, u):
        out, _ = self.m.lstm(u.view(-1, self.t, self.f))
        return self.m.fc(out[:, -1, :]).squeeze(-1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scaler = pickle.load(open(os.path.join(MODEL_DIR, "LSTM", "scaler.pkl"), "rb"))
    feat_cols = json.load(open(os.path.join(MODEL_DIR, "LSTM", "features.json")))
    assert len(feat_cols) == F_

    dev_raw = pd.read_csv(os.path.join(DATA_DIR, "dev.csv"), parse_dates=["Date"])
    oos_raw = pd.read_csv(os.path.join(DATA_DIR, "oos.csv"), parse_dates=["Date"])
    dev = TL.clean_categoricals(dev_raw)
    oos = TL.clean_categoricals(oos_raw)
    X_dev = dev[feat_cols].fillna(0).values.astype(np.float64)
    U_dev = scaler.transform(X_dev).astype(np.float32)
    int_mask = integer_columns(X_dev)
    print(f"Integer-valued columns (snapped in Manifold-LIME): {int(int_mask.sum())} / {F_}")

    rng = np.random.RandomState(SEED)
    inst_idx = rng.choice(len(oos), N_TABNET, replace=False)
    oos_inst = oos.iloc[inst_idx]
    X_inst_raw = oos_inst[feat_cols].fillna(0).values.astype(np.float32)
    U_inst = scaler.transform(X_inst_raw.astype(np.float64)).astype(np.float32)
    ids = oos_inst["RowNum"].values
    y_inst = np.zeros(len(inst_idx), dtype=int)           # regression: no class label

    # ---------------------------------------------------------------- TabNet
    print("\n" + "=" * 72 + "\nDemandForecast TabNet\n" + "=" * 72)
    reg = TabNetRegressor()
    reg.load_model(os.path.join(MODEL_DIR, "TabNet", "tabnet_model.zip.zip"))
    net = reg.network.eval()
    for prm in net.parameters():
        prm.requires_grad_(False)
    f_tab = TabNetValue(net, scaler.mean_, scaler.scale_)
    with torch.no_grad():
        at_u = f_tab(torch.tensor(U_inst)).numpy()
    real = reg.predict(X_inst_raw).ravel()
    print(f"Explained function vs real TabNet predictions: max |diff| = {np.abs(real - at_u).max():.2e}")
    native, _ = reg.explain(X_inst_raw)

    run_study(
        model_name="DemandForecast_TabNet", f_torch=f_tab, U_inst=U_inst, y_inst=y_inst, ids=ids,
        U_normal=U_dev, columns=feat_cols, out_dir=OUT_DIR,
        raw_mean_baseline=np.median(U_dev, axis=0, keepdims=True),
        snap=snap_via_standard_scaler(scaler, int_mask), lrd_model=None,
        cfg=StudyConfig(seed=SEED, kernelshap_n=3),
        extra_attrs={"tabnet_native_mask": ("TabNetNative", native.astype(np.float32))},
        extra_latency={"TabNet native mask (reg.explain)": lambda i: reg.explain(X_inst_raw[i:i + 1])},
        baseline_note="DemandForecast has no log transform, so the raw mean IS the origin (identical to the "
                      "zeros baseline). The training MEDIAN, in standardised units, is used here instead so "
                      "this slot is a genuinely different 'typical client' reference.",
    )

    # cohort metadata for later per-segment analysis (Phase 4)
    meta = pd.DataFrame({"RowNum": ids, "BRAND": oos_raw.iloc[inst_idx]["BRAND"].values,
                         "DayType": assign_day_type(oos_raw.iloc[inst_idx]).values,
                         "Actual_Quantity": oos_inst["QUANTITY"].values, "TabNet_Pred": real})
    meta.to_csv(os.path.join(OUT_DIR, "instances_meta_DemandForecast.csv"), index=False)

    # ---------------------------------------------------------------- LSTM
    print("\n" + "=" * 72 + "\nDemandForecast LSTM (real 28-day sequences)\n" + "=" * 72)
    print("Building 28-day sequences (full history lookup)...")
    full = TL.clean_categoricals(pd.read_csv(TL.FULL_DATA_PATH, parse_dates=["Date"]))
    min_date = full["Date"].min()
    full["_day_idx"] = (full["Date"] - min_date).dt.days
    lookup = TL.build_day_lookup(full, feat_cols, scaler)
    dev["_day_idx"] = (dev["Date"] - min_date).dt.days
    X_dev_seq, _ = TL.build_sequences(dev, lookup, F_)
    sub = oos_inst.iloc[:N_LSTM].copy()
    sub["_day_idx"] = (sub["Date"] - min_date).dt.days
    X_seq, _ = TL.build_sequences(sub, lookup, F_)
    U_seq_ref = X_dev_seq.reshape(len(X_dev_seq), -1)
    U_seq_inst = X_seq.reshape(len(X_seq), -1)
    cols = [f"{feat_cols[k]}@t-{T - 1 - j}" for j in range(T) for k in range(F_)]

    lstm = TL.DemandLSTM(input_dim=F_, hidden_dim=64, num_layers=2, dropout=0.2)
    lstm.load_state_dict(torch.load(os.path.join(MODEL_DIR, "LSTM", "best_lstm.pt")), strict=True)
    lstm.eval()
    for prm in lstm.parameters():
        prm.requires_grad_(False)
    f_lstm = LSTMScore(lstm, T, F_)
    with torch.no_grad():
        score = f_lstm(torch.tensor(U_seq_inst)).numpy()
        deployed = lstm(torch.tensor(X_seq)).numpy()
    print(f"relu(pre-ReLU score) vs deployed LSTM prediction: max |diff| = {np.abs(np.maximum(score, 0) - deployed).max():.2e}; "
          f"{int((score <= 0).sum())}/{len(score)} instances have a clamped (<=0) prediction")

    summary, attrs, meta_l = run_study(
        model_name="DemandForecast_LSTM", f_torch=f_lstm, U_inst=U_seq_inst, y_inst=np.zeros(len(U_seq_inst), dtype=int),
        ids=ids[:N_LSTM], U_normal=U_seq_ref, columns=cols, out_dir=OUT_DIR,
        raw_mean_baseline=np.median(U_seq_ref, axis=0, keepdims=True),
        snap=snap_via_standard_scaler(scaler, int_mask, tile=T), lrd_model=None,
        cfg=StudyConfig(seed=SEED, lime_samples=1500, latency_n=30, diag_per_class=10,
                        stability_seeds=4, kernelshap_n=2, kernelshap_big_n=1, n_donors=3000,
                        run_sweeps=False),          # sweeps are too costly at 1,428 dimensions
        baseline_note="standardised-space median (see TabNet note); window flattened to 28 x 51 columns",
    )

    # ---- aggregate the flattened (time x feature) attributions back to 51 features / 28 timesteps
    n = len(U_seq_inst)
    feat_level = {v: A.reshape(n, T, F_).sum(axis=1) for v, A in attrs.items()}
    blocks = []
    for v, A in feat_level.items():
        b = pd.DataFrame(A, columns=feat_cols)
        b.insert(0, "Variant", v)
        b.insert(0, "Explainer", meta_l[v])
        b.insert(0, "Instance_ID", ids[:N_LSTM])
        blocks.append(b)
    pd.concat(blocks, ignore_index=True).to_csv(os.path.join(OUT_DIR, "attributions_DemandForecast_LSTM_featurelevel.csv"), index=False)

    prim = {"AdaptedSHAP": feat_level["ig_kmeans"], "ManifoldLIME": feat_level["lime_manifold"]}
    agree = {f"{a}_vs_{b}__feature_level": v for (a, b), v in pairwise_agreement(prim).items()}
    time_profile = {v: np.abs(attrs[v].reshape(n, T, F_)).sum(axis=2).mean(axis=0).round(5).tolist()
                    for v in ("ig_kmeans", "lime_manifold")}
    sp = os.path.join(OUT_DIR, "tri_explainer_DemandForecast_LSTM_summary.json")
    s = json.load(open(sp))
    s["agreement_preview_feature_level_placeholder_tau"] = agree
    s["mean_abs_attribution_by_timestep_t-27_to_t-0"] = time_profile
    s["clamped_prediction_fraction"] = float((score <= 0).mean())
    json.dump(s, open(sp, "w"), indent=2)
    print("\nFeature-level (time-aggregated) agreement preview:", {k: round(v, 3) for k, v in agree.items()})
    print("Mean |IG attribution| by timestep (t-27 ... t-0):", np.round(time_profile["ig_kmeans"], 3))


if __name__ == "__main__":
    main()
