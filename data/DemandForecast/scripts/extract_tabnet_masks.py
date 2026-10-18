"""
Extract TabNet's native Sparsemax attention masks — DemandForecast.

IMPORTANT — what this is and is not:
  TabNet exposes, for free, which features its attention mechanism selected at
  each decision step. That is a *native model signal*, not an explainer.

  It is NOT one of the three explainers the research proposal requires
  (Adapted SHAP, Manifold-LIME, LRD). Those are built in Phase 3 and will be
  compared *against* these masks as a reference point.

Outputs
-------
XAI_Outputs/tabnet_masks.csv  with columns [Model, Signal, Feature, Attribution]
  Signal = "Sparsemax_Attention_Global"  -> one importance per feature, whole model
  Signal = "Instance_Mask_Attribution"   -> mean over the sampled instances
"""
import os
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor

DATA_DIR   = "/Users/somyabhadada/nodex8/data/DemandForecast/samples/"
MODEL_DIR  = "/Users/somyabhadada/nodex8/data/DemandForecast/Model_Outputs/"
OUTPUT_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/XAI_Outputs/"

EXCLUDE_COLS = ["RowNum", "Date", "UPC", "COMP_STR_NO", "QUANTITY", "BRAND", "RET_LIR_ID"]
CATEGORICAL_COLS = ["PROMOTION_FLG", "C_PROMOTION_FLG"]
N_SAMPLE = 200
SEED = 42


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    val_df = pd.read_csv(os.path.join(DATA_DIR, "oos.csv"))
    for col in CATEGORICAL_COLS:
        if col in val_df.columns:
            val_df[col] = val_df[col].map({'Y': 1, 'N': 0, '1': 1, '0': 0, 1: 1, 0: 0}).fillna(0)
    feat_cols = [c for c in val_df.columns if c not in EXCLUDE_COLS]

    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(val_df), min(N_SAMPLE, len(val_df)), replace=False)
    X_sample = val_df.iloc[idx][feat_cols].fillna(0).values.astype(np.float32)

    results = []

    # 1. Global Sparsemax importances, written out by the training script
    fi_path = os.path.join(MODEL_DIR, "TabNet/feature_importance.csv")
    if os.path.exists(fi_path):
        for _, row in pd.read_csv(fi_path).iterrows():
            results.append({
                "Model": "TabNet",
                "Signal": "Sparsemax_Attention_Global",
                "Feature": row["Feature"],
                "Attribution": float(row["Importance"]),
            })
        print(f"  Global Sparsemax importances: {len(results)} features.")

    # 2. Instance-level masks, averaged over the sample
    model_path = os.path.join(MODEL_DIR, "TabNet/tabnet_model.zip.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "TabNet/tabnet_model.zip")

    if os.path.exists(model_path):
        clf = TabNetRegressor()
        clf.load_model(model_path)
        explain_matrix, _ = clf.explain(X_sample)
        mean_mask = np.mean(explain_matrix, axis=0)
        for f, imp in zip(feat_cols, mean_mask):
            results.append({
                "Model": "TabNet",
                "Signal": "Instance_Mask_Attribution",
                "Feature": f,
                "Attribution": float(imp),
            })
        print(f"  Instance masks averaged over {len(X_sample)} instances.")
    else:
        print("  TabNet model file not found — skipping instance masks.")

    out = os.path.join(OUTPUT_DIR, "tabnet_masks.csv")
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nSaved {len(results)} rows to {out}")


if __name__ == "__main__":
    main()
