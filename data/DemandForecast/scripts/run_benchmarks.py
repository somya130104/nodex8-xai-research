"""
Phase 5 — benchmark suite on DemandForecast (TabNet, LSTM).  See src/benchmarks/ and ROADMAP.md Phase 5.

Same stages as the TradeFraud runner (faithfulness, planted, robustness, runtime, memory). The corruption ground-truth
test (5.2a) needs an autoencoder, which DemandForecast does not have, so ground truth here comes from the
planted-signal test only. LRD does not apply (no bottleneck); TabNet's native attention mask is carried as a 4th signal.
The LSTM is explained on its 1,428 flattened (28 days x 51 features) columns, with a scaled-down budget (fewer
instances and search iterations) because every explanation is ~25x more expensive there.

Usage:
  python3 data/DemandForecast/scripts/run_benchmarks.py --models DemandForecast_TabNet --stages faithfulness,robustness
  python3 data/DemandForecast/scripts/run_benchmarks.py --stages planted
  python3 data/DemandForecast/scripts/run_benchmarks.py --stages runtime        # alone, last
  add --quick for a small smoke run (numbers are not meaningful).

Outputs: data/DemandForecast/XAI_Outputs/benchmarks/
"""
import argparse
import json
import os
import pickle
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append("/Users/somyabhadada/nodex8")
import train_lstm as TL
from run_tri_explainer_tabnet_lstm import TabNetValue, LSTMScore
from src.benchmarks import pipeline as P
from src.benchmarks.suite import ExplainerSuite
from src.explainers.snapping import (integer_columns, snap_stochastic_via_standard_scaler, snap_via_standard_scaler)

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

DATA = "/Users/somyabhadada/nodex8/data/DemandForecast/samples/"
MODEL_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/Model_Outputs/"
P3 = "/Users/somyabhadada/nodex8/data/DemandForecast/XAI_Outputs/tri_explainer/"
OUT = "/Users/somyabhadada/nodex8/data/DemandForecast/XAI_Outputs/benchmarks/"
ALL_MODELS = ["DemandForecast_TabNet", "DemandForecast_LSTM"]
SEED, N_TABNET, N_LSTM = 42, 250, 120
T, F_ = TL.SEQ_LEN, 51


def setup():
    scaler = pickle.load(open(MODEL_DIR + "LSTM/scaler.pkl", "rb"))
    feat = json.load(open(MODEL_DIR + "LSTM/features.json"))
    dev = TL.clean_categoricals(pd.read_csv(DATA + "dev.csv", parse_dates=["Date"]))
    oos = TL.clean_categoricals(pd.read_csv(DATA + "oos.csv", parse_dates=["Date"]))
    X_dev = dev[feat].fillna(0).values.astype(np.float64)
    U_dev = scaler.transform(X_dev).astype(np.float32)
    int_mask = integer_columns(X_dev)
    inst_idx = np.random.RandomState(SEED).choice(len(oos), N_TABNET, replace=False)
    oos_inst = oos.iloc[inst_idx]
    return dict(scaler=scaler, feat=feat, dev=dev, oos=oos, U_dev=U_dev, int_mask=int_mask, oos_inst=oos_inst)


def _stored(model_name, ids, cols):
    att = pd.read_csv(f"{P3}/attributions_{model_name}.csv")
    out = {}
    for v, name in P.STORED.items():
        a = att[att.Variant == v]
        if len(a):
            out[name] = a.set_index("Instance_ID").loc[ids, cols].values.astype(np.float64)
    return out


def build_tabnet(S, quick):
    sc, feat = S["scaler"], S["feat"]
    reg = TabNetRegressor()
    reg.load_model(MODEL_DIR + "TabNet/tabnet_model.zip.zip")
    net = reg.network.eval()
    for prm in net.parameters():
        prm.requires_grad_(False)
    f = TabNetValue(net, sc.mean_, sc.scale_)
    native_fn = lambda U: reg.explain((np.asarray(U, dtype=np.float64) * sc.scale_ + sc.mean_).astype(np.float32))[0]
    suite = ExplainerSuite(f, S["U_dev"], snap=snap_via_standard_scaler(sc, S["int_mask"]),
                           snap_stochastic=snap_stochastic_via_standard_scaler(sc, S["int_mask"]), native_fn=native_fn)
    oi = S["oos_inst"]
    ids = oi["RowNum"].values
    U_inst = sc.transform(oi[feat].fillna(0).values.astype(np.float64)).astype(np.float32)
    pool = S["oos"].sample(3000, random_state=0)
    b = P.ModelBundle(name="DemandForecast_TabNet", kind="tabnet", suite=suite, U_inst=U_inst,
                      attrs_stored=_stored("DemandForecast_TabNet", ids, feat), U_normal=S["U_dev"], K=10,
                      U_runtime=sc.transform(pool[feat].fillna(0).values.astype(np.float64)).astype(np.float32),
                      sigma=S["U_dev"].std(0), run_kernelshap=False)
    if quick:
        b.n_runtime, b.n_mem, b.n_stab, b.n_adv, b.adv_iters = 20, 5, 8, 3, 4
    return b


def build_lstm(S, quick):
    sc, feat = S["scaler"], S["feat"]
    full = TL.clean_categoricals(pd.read_csv(TL.FULL_DATA_PATH, parse_dates=["Date"]))
    min_date = full["Date"].min()
    full["_day_idx"] = (full["Date"] - min_date).dt.days
    lookup = TL.build_day_lookup(full, feat, sc)
    dev = S["dev"].copy()
    dev["_day_idx"] = (dev["Date"] - min_date).dt.days
    X_dev_seq, _ = TL.build_sequences(dev, lookup, F_)
    U_ref = X_dev_seq.reshape(len(X_dev_seq), -1)

    def seq_rows(df):
        d = df.copy()
        d["_day_idx"] = (d["Date"] - min_date).dt.days
        X, _ = TL.build_sequences(d, lookup, F_)
        return X.reshape(len(X), -1).astype(np.float32)

    oi = S["oos_inst"].iloc[:N_LSTM]
    ids = oi["RowNum"].values
    U_inst = seq_rows(oi)
    cols = [f"{feat[k]}@t-{T - 1 - j}" for j in range(T) for k in range(F_)]
    lstm = TL.DemandLSTM(input_dim=F_, hidden_dim=64, num_layers=2, dropout=0.2)
    lstm.load_state_dict(torch.load(MODEL_DIR + "LSTM/best_lstm.pt"), strict=True)
    lstm.eval()
    for prm in lstm.parameters():
        prm.requires_grad_(False)
    f = LSTMScore(lstm, T, F_)
    int_mask = S["int_mask"]
    suite = ExplainerSuite(f, U_ref, snap=snap_via_standard_scaler(sc, int_mask, tile=T),
                           snap_stochastic=snap_stochastic_via_standard_scaler(sc, int_mask, tile=T), lime_samples=1500)
    pool = S["oos"].sample(300, random_state=0)
    b = P.ModelBundle(name="DemandForecast_LSTM", kind="lstm", suite=suite, U_inst=U_inst,
                      attrs_stored=_stored("DemandForecast_LSTM", ids, cols), U_normal=U_ref, K=50,
                      U_runtime=seq_rows(pool), sigma=U_ref.std(0), run_kernelshap=False,
                      n_runtime=200, n_mem=10, n_stab=30, n_adv=8, adv_iters=15, adv_cands=3)
    if quick:
        b.n_runtime, b.n_mem, b.n_stab, b.n_adv, b.adv_iters = 10, 3, 4, 2, 3
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--stages", default="faithfulness,planted,robustness")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--memory-child", default=None, help="internal: probe one explainer in this fresh process and exit")
    a = ap.parse_args()
    stages = a.stages.split(",")
    out = OUT + ("quick/" if a.quick else "")
    S = setup()
    build = {"DemandForecast_TabNet": build_tabnet, "DemandForecast_LSTM": build_lstm}
    if a.memory_child:
        P.memory_probe(build[a.models.split(",")[0]](S, a.quick), a.memory_child)
        return
    if "planted" in stages:
        sc = S["scaler"]
        P.run_planted("DemandForecast", S["U_dev"], snap_via_standard_scaler(sc, S["int_mask"]),
                      snap_stochastic_via_standard_scaler(sc, S["int_mask"]), out, reps=1 if a.quick else 3,
                      n_eval=20 if a.quick else 200, run_kernelshap=True)
    for m in [x for x in a.models.split(",") if x]:
        if not set(stages) & {"faithfulness", "robustness", "runtime", "memory"}:
            break
        print("\n" + "=" * 78 + f"\n{m}\n" + "=" * 78)
        b = build[m](S, a.quick)
        if "faithfulness" in stages:
            P.run_faithfulness(b, out)
        if "robustness" in stages:
            P.run_robustness(b, out)
        if "runtime" in stages:
            P.run_runtime(b, out)
        if "memory" in stages:
            P.run_memory(m, b.suite.available, [sys.executable, os.path.abspath(__file__), "--models", m] + (["--quick"] if a.quick else []), out)


if __name__ == "__main__":
    main()
