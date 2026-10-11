"""
Generate Standard SHAP explanations for DemandForecast baseline models.
"""
import os
import pickle
import numpy as np
import pandas as pd
import shap
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/samples/"
MODEL_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/Model_Outputs/"
OUTPUT_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/XAI_Outputs/"

EXCLUDE_COLS = ["RowNum", "Date", "UPC", "COMP_STR_NO", "QUANTITY", "BRAND", "RET_LIR_ID"]
CATEGORICAL_COLS = ["PROMOTION_FLG", "C_PROMOTION_FLG"]

MODELS = [
    ("XGBoost", "model_xgboost.pkl", "Tree"),
    ("RandomForest", "model_rf.pkl", "Tree"),
    ("LightGBM", "model_lightgbm.pkl", "Tree"),
    ("LinearRegression", "model_ols.pkl", "Linear")
]

def load_dataset(fname, dummify=False):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    
    if dummify:
        df_model = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns], drop_first=True)
    else:
        df_model = df.copy()
        for col in CATEGORICAL_COLS:
            if col in df_model.columns:
                df_model[col] = df_model[col].map({'Y': 1, 'N': 0, '1': 1, '0': 0, 1: 1, 0: 0}).fillna(0)
                
    feat_cols = [c for c in df_model.columns if c not in EXCLUDE_COLS]
    X = df_model[feat_cols]
    return df, X, feat_cols

def run_shap():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading data...")
    df_bg_raw, X_bg, _ = load_dataset("dev.csv", dummify=False)
    bg_data = X_bg.sample(min(200, len(X_bg)), random_state=42)
    
    df_bg_dummy_raw, X_bg_dummy, _ = load_dataset("dev.csv", dummify=True)
    bg_data_dummy = X_bg_dummy.sample(min(200, len(X_bg_dummy)), random_state=42).fillna(0)
    
    df_oos, X_oos, feat_cols_tree = load_dataset("oos.csv", dummify=False)
    
    _, X_oos_dummy, feat_cols_lin = load_dataset("oos.csv", dummify=True)
    X_oos_dummy = X_oos_dummy.fillna(0)
    
    global_shap_results = []
    
    for model_name, pkl_name, family in MODELS:
        print(f"\n--- Processing {model_name} ---")
        model_path = os.path.join(MODEL_DIR, model_name, pkl_name)
        if not os.path.exists(model_path):
            print(f"Skipping {model_name} (not found)")
            continue
            
        with open(model_path, "rb") as f:
            model_pkg = pickle.load(f)
            
        print("  Computing SHAP...")
        if family == "Linear":
            model = model_pkg['model']
            scaler = model_pkg['scaler']
            expected_feats = model_pkg['features']
            
            # Align features
            X_eval = X_oos_dummy.copy()
            for col in expected_feats:
                if col not in X_eval.columns:
                    X_eval[col] = 0
            X_eval = X_eval[expected_feats]
            
            X_bg_eval = bg_data_dummy.copy()
            for col in expected_feats:
                if col not in X_bg_eval.columns:
                    X_bg_eval[col] = 0
            X_bg_eval = X_bg_eval[expected_feats]
            
            X_scaled = scaler.transform(X_eval)
            bg_scaled = scaler.transform(X_bg_eval)
            
            explainer = shap.LinearExplainer(model, bg_scaled)
            shap_values = explainer.shap_values(X_scaled)
            features_used = expected_feats
        else:
            model = model_pkg
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_oos, check_additivity=False)
            features_used = feat_cols_tree
            
        # Global SHAP
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        for f, val in zip(features_used, mean_abs_shap):
            global_shap_results.append({
                "Model": model_name,
                "Feature": f,
                "Mean_Abs_SHAP": val
            })
            
        print(f"  Finished SHAP for {model_name}")

    pd.DataFrame(global_shap_results).to_csv(os.path.join(OUTPUT_DIR, "shap_global.csv"), index=False)
    print("\n✅ SHAP extraction complete! Outputs in", OUTPUT_DIR)

if __name__ == "__main__":
    run_shap()
