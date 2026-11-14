"""
Phase 5 — benchmark suite on TradeFraud (autoencoder, VAE, TabNet).  See src/benchmarks/ and ROADMAP.md Phase 5.

Stages (run any subset; RUNTIME must be run alone, because timings are meaningless if anything else is using the CPU):
  faithfulness   5.1  deletion / insertion curves on the Phase 3 instances
  groundtruth    5.2a 3 known corrupted features on 500 unseen normal clients (autoencoder and VAE only)
  planted        5.2b planted-signal MLP with known drivers (once for the dataset; stand-in for supervised models)
  robustness     5.3  stability under 1% noise + hill-climbing adversarial search
  runtime        5.4  p50 / p95 latency over 1,000 single-instance explanations
  memory         5.4  peak resident-memory growth, one fresh process per (model, explainer); can run alongside other stages

Usage:
  python3 data/TradeFraud/scripts/run_benchmarks.py --models Model4_Autoencoder --stages faithfulness,groundtruth,robustness
  python3 data/TradeFraud/scripts/run_benchmarks.py --stages planted
  python3 data/TradeFraud/scripts/run_benchmarks.py --stages runtime            # alone, last
  add --quick for a small smoke run (numbers are not meaningful).

Outputs: data/TradeFraud/XAI_Outputs/benchmarks/
"""
import argparse
import os
import pickle
import sys
import warnings
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append("/Users/somyabhadada/nodex8")
import run_consensus_cli as RC
from src.benchmarks import pipeline as P
from src.benchmarks.suite import ExplainerSuite
from src.explainers.snapping import (integer_columns, snap_stochastic_via_preprocessor, snap_via_preprocessor)

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

OUT = "/Users/somyabhadada/nodex8/data/TradeFraud/XAI_Outputs/benchmarks/"
ALL_MODELS = ["Model4_Autoencoder", "Model9_VAE", "Model7_TabNet"]


def setup():
    _, X_tr, y_tr, feat = RC.load_split("train.csv")
    X_norm = X_tr[y_tr == 0]
    pre = pickle.load(open(RC.MODEL_DIR + "Model4_Autoencoder/preprocessor.pkl", "rb"))
    U_normal = pre.transform(X_norm).astype(np.float32)
    int_mask = integer_columns(X_norm.values)
    snap = snap_via_preprocessor(pre, int_mask)
    snap_s = snap_stochastic_via_preprocessor(pre, int_mask)
    df_va, X_va, y_va, _ = RC.load_split("val_oos.csv")
    X_va = X_va.copy()
    X_va.index = df_va["Client_ID"].values
    return dict(feat=feat, pre=pre, U_normal=U_normal, snap=snap, snap_s=snap_s, X_va=X_va, y_va=y_va, p=len(feat))


def build_bundle(model_name, S, quick):
    pre, feat = S["pre"], S["feat"]
    f_torch, lrd_model, lrd_kind, native_raw = RC.build_model(model_name, S["p"])
    native_fn = (lambda U: native_raw(pre.inverse_transform(np.asarray(U, dtype=np.float64)).astype(np.float32))) if native_raw else None
    suite = ExplainerSuite(f_torch, S["U_normal"], snap=S["snap"], snap_stochastic=S["snap_s"], lrd_model=lrd_model,
                           lrd_kind=lrd_kind, native_fn=native_fn)
    inst = pd.read_csv(f"{RC.P3_DIR}/instances_{model_name}.csv")
    ids = inst["Instance_ID"].values
    U_inst = pre.transform(S["X_va"].loc[ids, feat]).astype(np.float32)
    att = pd.read_csv(f"{RC.P3_DIR}/attributions_{model_name}.csv")
    stored = {}
    for v, name in P.STORED.items():
        a = att[att.Variant == v]
        if len(a):
            stored[name] = a.set_index("Instance_ID").loc[ids, feat].values.astype(np.float64)
    normals = S["X_va"][S["y_va"] == 0]
    rs = np.random.RandomState(0)
    pool_idx = rs.choice(len(normals), 3000, replace=False)
    gt_ok = lrd_model is not None
    b = P.ModelBundle(name=model_name, kind=RC.MODELS[model_name]["kind"], suite=suite, U_inst=U_inst, attrs_stored=stored,
                      U_normal=S["U_normal"], K=20, U_runtime=pre.transform(normals.iloc[pool_idx]).astype(np.float32),
                      sigma=S["U_normal"].std(0), run_kernelshap=gt_ok,
                      gt_rows=normals.values if gt_ok else None,
                      gt_transform=(lambda Xr: pre.transform(pd.DataFrame(Xr, columns=feat)).astype(np.float32)) if gt_ok else None)
    if quick:
        b.n_runtime, b.n_kernel_runtime, b.n_mem, b.n_stab, b.n_adv, b.adv_iters = 20, 3, 5, 8, 3, 4
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--stages", default="faithfulness,groundtruth,planted,robustness")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--memory-child", default=None, help="internal: probe one explainer in this fresh process and exit")
    a = ap.parse_args()
    stages = a.stages.split(",")
    out = OUT + ("quick/" if a.quick else "")
    S = setup()
    if a.memory_child:
        P.memory_probe(build_bundle(a.models.split(",")[0], S, a.quick), a.memory_child)
        return
    if "planted" in stages:
        P.run_planted("TradeFraud", S["U_normal"], S["snap"], S["snap_s"], out, reps=1 if a.quick else 3, n_eval=20 if a.quick else 200)
    for m in [x for x in a.models.split(",") if x]:
        if not set(stages) & {"faithfulness", "groundtruth", "robustness", "runtime", "memory"}:
            break
        print("\n" + "=" * 78 + f"\n{m}\n" + "=" * 78)
        b = build_bundle(m, S, a.quick)
        if "faithfulness" in stages:
            P.run_faithfulness(b, out)
        if "groundtruth" in stages and b.gt_rows is not None:
            if a.quick:
                b.gt_rows = b.gt_rows[:200]
            P.run_groundtruth(b, out, n_rows=40 if a.quick else 500)
        if "robustness" in stages:
            P.run_robustness(b, out)
        if "runtime" in stages:
            P.run_runtime(b, out)
        if "memory" in stages:
            P.run_memory(m, b.suite.available + (["KernelSHAP"] if b.run_kernelshap else []),
                         [sys.executable, os.path.abspath(__file__), "--models", m] + (["--quick"] if a.quick else []), out)


if __name__ == "__main__":
    main()
