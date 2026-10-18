"""
Generate Standard SHAP and LIME explanations for TradeFraud baseline models.

Phase 0 fix applied (see ARCHITECTURE.md Section 9): duplicate dispute rows
(10x repeated in the supplied data) are dropped before sampling background
data or explaining instances, so explanations are not disproportionately
drawn from the same handful of repeated clients.
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import warnings

sys.path.append("/Users/somyabhadada/nodex8")
from src.eval_utils import dedupe_rows

warnings.filterwarnings("ignore")

DATA_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/Modeling_Data_Enriched/"
MODEL_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/Model_Outputs/"
OUTPUT_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/XAI_Outputs/"

EXCLUDE_COLS = ["Client", "Client_ID", "Date", "Dispute", "Segment", "Activeness_Segment"]

MODELS = [
    ("Model2_XGBoost", "model2_xgboost.pkl", "Tree"),
    ("Model3_RandomForest", "model3_rf.pkl", "Tree"),
    ("Model5_LightGBM", "model5_lightgbm.pkl", "Tree"),
    ("Model6_LogReg", "model6_logreg.pkl", "Linear")
]

def load_dataset(fname):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df["target"] = (df["Dispute"] == "D").astype(int)
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != "target"]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = dedupe_rows(df, feat_cols)
    X = df[feat_cols].fillna(0)
    y = df["target"]
    return df, X, y, feat_cols

def run_xai():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading data...")
    # Use train for background (subset)
    _, X_train, y_train, feat_cols = load_dataset("train.csv")
    bg_data = X_train.sample(min(200, len(X_train)), random_state=42)
    
    # Use OOS for explaining
    df_oos, X_oos, y_oos, _ = load_dataset("val_oos.csv")
    
    # For local explanations, pick 15 samples: 5 high risk, 5 border, 5 low risk
    # We will pick these per model based on their predictions.
    
    global_shap_results = []
    cohort_shap_results = []
    
    for model_name, pkl_name, family in MODELS:
        print(f"\n--- Processing {model_name} ---")
        model_path = os.path.join(MODEL_DIR, model_name, pkl_name)
        if not os.path.exists(model_path):
            print(f"Skipping {model_name} (not found)")
            continue
            
        with open(model_path, "rb") as f:
            model_pkg = pickle.load(f)
            
        if family == "Linear":
            model = model_pkg['model']
            scaler = model_pkg['scaler']
            X_oos_proc = pd.DataFrame(scaler.transform(X_oos), columns=feat_cols)
            bg_data_proc = pd.DataFrame(scaler.transform(bg_data), columns=feat_cols)
        else:
            model = model_pkg
            X_oos_proc = X_oos
            bg_data_proc = bg_data

        # 1. Generate Predictions for Stratification
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X_oos_proc)[:, 1]
        else:
            # LightGBM standard API returns probs
            preds = model.predict(X_oos_proc)
            
        df_oos["pred_prob"] = preds
        df_oos["Decile"] = pd.qcut(df_oos["pred_prob"].rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])

        # 2. Compute SHAP
        print("  Computing SHAP...")
        if family == "Tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_oos_proc, check_additivity=False)
            if isinstance(shap_values, list): # RF
                shap_values = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
        else:
            explainer = shap.LinearExplainer(model, bg_data_proc)
            shap_values = explainer.shap_values(X_oos_proc)
            
        # Global SHAP
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        for f, val in zip(feat_cols, mean_abs_shap):
            global_shap_results.append({
                "Model": model_name,
                "Feature": f,
                "Mean_Abs_SHAP": val
            })
            
        # Cohort SHAP (By Segment)
        for seg in df_oos["Segment"].dropna().unique():
            idx = df_oos["Segment"] == seg
            if idx.sum() > 5:
                seg_shap = np.abs(shap_values[idx]).mean(axis=0)
                for f, val in zip(feat_cols, seg_shap):
                    cohort_shap_results.append({
                        "Model": model_name,
                        "Segment": seg,
                        "Feature": f,
                        "Mean_Abs_SHAP": val
                    })
                    
        # Cohort SHAP (By Decile)
        for dec in df_oos["Decile"].unique():
            idx = df_oos["Decile"] == dec
            if idx.sum() > 5:
                dec_shap = np.abs(shap_values[idx]).mean(axis=0)
                for f, val in zip(feat_cols, dec_shap):
                    cohort_shap_results.append({
                        "Model": model_name,
                        "Decile": dec,
                        "Feature": f,
                        "Mean_Abs_SHAP": val
                    })
                    
        print(f"  Finished SHAP for {model_name}")

    # Save aggregated SHAP
    pd.DataFrame(global_shap_results).to_csv(os.path.join(OUTPUT_DIR, "shap_global.csv"), index=False)
    pd.DataFrame(cohort_shap_results).to_csv(os.path.join(OUTPUT_DIR, "shap_cohort.csv"), index=False)
    
    print("\n✅ SHAP extraction complete! Outputs in", OUTPUT_DIR)

if __name__ == "__main__":
    run_xai()
