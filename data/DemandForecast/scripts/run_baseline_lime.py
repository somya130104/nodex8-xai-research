"""
Generate Standard LIME explanations for DemandForecast baseline models.

Phase 0 fix applied (see ARCHITECTURE.md Section 9): LIME attributions used
to be mapped back to feature names by checking whether a feature's name was
a *substring* of LIME's text condition (e.g. "lag_14 <= 0.57"). Because
"lag_1" is a substring of "lag_14", and "SALE_PRICE" is a substring of
"C_SALE_PRICE", that silently credited one feature's weight to a different
feature — 15 features (every C_* competitor feature, lag_14, day_of_week,
discount_pct, etc.) ended up with a LIME weight of exactly zero for every
model. The fix uses exp.as_map(), which LIME returns as (feature_index,
weight) pairs with no string parsing involved.
"""
import os
import pickle
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
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

def run_lime():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading data...")
    _, X_train, _ = load_dataset("dev.csv", dummify=False)
    _, X_train_dummy, _ = load_dataset("dev.csv", dummify=True)
    X_train_dummy = X_train_dummy.fillna(0)
    
    df_oos, X_oos, feat_cols_tree = load_dataset("oos.csv", dummify=False)
    _, X_oos_dummy, feat_cols_lin = load_dataset("oos.csv", dummify=True)
    X_oos_dummy = X_oos_dummy.fillna(0)
    
    # Sample 100 instances from OOS to run LIME
    sample_idx = np.random.RandomState(42).choice(len(X_oos), min(100, len(X_oos)), replace=False)
    X_sample_tree = X_oos.iloc[sample_idx]
    X_sample_lin = X_oos_dummy.iloc[sample_idx]
    
    lime_global_results = []
    
    for model_name, pkl_name, family in MODELS:
        print(f"\n--- Processing {model_name} ---")
        model_path = os.path.join(MODEL_DIR, model_name, pkl_name)
        if not os.path.exists(model_path):
            print(f"Skipping {model_name} (not found)")
            continue
            
        with open(model_path, "rb") as f:
            model_pkg = pickle.load(f)
            
        print(f"  Computing LIME for 100 instances...")
        
        if family == "Linear":
            model = model_pkg['model']
            scaler = model_pkg['scaler']
            expected_feats = model_pkg['features']
            
            # Align features
            X_eval = X_sample_lin.copy()
            for col in expected_feats:
                if col not in X_eval.columns:
                    X_eval[col] = 0
            X_eval = X_eval[expected_feats]
            
            X_bg = X_train_dummy.copy()
            for col in expected_feats:
                if col not in X_bg.columns:
                    X_bg[col] = 0
            X_bg = X_bg[expected_feats]
            
            def predict_fn(x): 
                return model.predict(scaler.transform(pd.DataFrame(x, columns=expected_feats)))
                
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_bg.values[:2000], 
                feature_names=expected_feats,
                mode='regression',
                random_state=42
            )
            
            X_eval_proc = X_eval.values
            features_used = expected_feats
        else:
            model = model_pkg
            # Needs a function that takes numpy array and returns predictions
            def predict_fn(x):
                x_df = pd.DataFrame(x, columns=feat_cols_tree)
                return model.predict(x_df)
                
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values[:2000],
                feature_names=feat_cols_tree,
                mode='regression',
                random_state=42
            )
            
            X_eval_proc = X_sample_tree.values
            features_used = feat_cols_tree

        lime_values = np.zeros((len(X_eval_proc), len(features_used)))

        for i, row in enumerate(X_eval_proc):
            exp = explainer.explain_instance(row, predict_fn, num_features=len(features_used), num_samples=500)
            # as_map()[1] gives (feature_index, weight) pairs directly, with
            # the sign matching as_list() — no text parsing, so no substring
            # collisions between feature names like lag_1 / lag_14.
            for feat_idx, weight in exp.as_map()[1]:
                lime_values[i, feat_idx] += weight
        
        mean_abs_lime = np.abs(lime_values).mean(axis=0)
        for f, val in zip(features_used, mean_abs_lime):
            lime_global_results.append({
                "Model": model_name,
                "Feature": f,
                "Mean_Abs_LIME": val
            })
            
        print(f"  Finished LIME for {model_name}")

    pd.DataFrame(lime_global_results).to_csv(os.path.join(OUTPUT_DIR, "lime_global.csv"), index=False)
    print("\n✅ LIME extraction complete! Outputs in", OUTPUT_DIR)

if __name__ == "__main__":
    run_lime()
