"""
Generate Standard LIME explanations for TradeFraud baseline models.

Phase 0 note (see ARCHITECTURE.md Section 9): no TradeFraud feature name is
a substring of another, so the original text-matching approach below did not
actually mis-map any weights on this dataset (unlike DemandForecast, where it
did). It is still replaced with exp.as_map(), which uses LIME's own
(feature_index, weight) pairs directly, so the result no longer depends on
that coincidence holding as features are added or renamed in the future.

Also applies the Phase 0 dedupe fix: duplicate dispute rows are dropped
before sampling or explaining.
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
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

def run_lime():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading data...")
    _, X_train, y_train, feat_cols = load_dataset("train.csv")
    df_oos, X_oos, y_oos, _ = load_dataset("val_oos.csv")
    
    # We will sample 250 instances from OOS to run LIME (LIME is slow)
    sample_idx = np.random.RandomState(42).choice(len(X_oos), min(250, len(X_oos)), replace=False)
    X_sample = X_oos.iloc[sample_idx]
    df_sample = df_oos.iloc[sample_idx]
    
    lime_global_results = []
    lime_cohort_results = []
    
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
            X_train_proc = scaler.transform(X_train)
            X_sample_proc = scaler.transform(X_sample)
            def predict_fn(x): return model.predict_proba(x)
        else:
            model = model_pkg
            X_train_proc = X_train.values
            X_sample_proc = X_sample.values
            if hasattr(model, "predict_proba"):
                def predict_fn(x): return model.predict_proba(pd.DataFrame(x, columns=feat_cols))
            else:
                # LightGBM binary gives 1D array of probs
                def predict_fn(x):
                    probs = model.predict(pd.DataFrame(x, columns=feat_cols))
                    return np.vstack([1-probs, probs]).T

        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train_proc[:5000], # Subsample train for background stats
            feature_names=feat_cols,
            class_names=['Normal', 'Dispute'],
            mode='classification',
            random_state=42
        )
        
        # Get predictions for deciling
        preds = predict_fn(X_sample_proc)[:, 1]
        df_sample["pred_prob"] = preds
        df_sample["Decile"] = pd.qcut(df_sample["pred_prob"].rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
        
        print(f"  Computing LIME for {len(X_sample)} instances...")
        lime_values = np.zeros((len(X_sample_proc), len(feat_cols)))
        
        for i, row in enumerate(X_sample_proc):
            exp = explainer.explain_instance(row, predict_fn, num_features=len(feat_cols), num_samples=1000)
            # as_map()[1] gives (feature_index, weight) pairs for the
            # "Dispute" class directly — no text parsing, so no risk of a
            # substring collision between feature names.
            for feat_idx, weight in exp.as_map()[1]:
                lime_values[i, feat_idx] += weight
        
        # Global LIME
        mean_abs_lime = np.abs(lime_values).mean(axis=0)
        for f, val in zip(feat_cols, mean_abs_lime):
            lime_global_results.append({
                "Model": model_name,
                "Feature": f,
                "Mean_Abs_LIME": val
            })
            
        # Cohort LIME (By Segment)
        for seg in df_sample["Segment"].dropna().unique():
            idx = df_sample["Segment"] == seg
            if idx.sum() > 3:
                seg_lime = np.abs(lime_values[idx]).mean(axis=0)
                for f, val in zip(feat_cols, seg_lime):
                    lime_cohort_results.append({
                        "Model": model_name,
                        "Segment": seg,
                        "Feature": f,
                        "Mean_Abs_LIME": val
                    })
                    
        # Cohort LIME (By Decile)
        for dec in df_sample["Decile"].unique():
            idx = df_sample["Decile"] == dec
            if idx.sum() > 3:
                dec_lime = np.abs(lime_values[idx]).mean(axis=0)
                for f, val in zip(feat_cols, dec_lime):
                    lime_cohort_results.append({
                        "Model": model_name,
                        "Decile": dec,
                        "Feature": f,
                        "Mean_Abs_LIME": val
                    })
                    
        print(f"  Finished LIME for {model_name}")

    # Save aggregated LIME
    pd.DataFrame(lime_global_results).to_csv(os.path.join(OUTPUT_DIR, "lime_global.csv"), index=False)
    pd.DataFrame(lime_cohort_results).to_csv(os.path.join(OUTPUT_DIR, "lime_cohort.csv"), index=False)
    
    print("\n✅ LIME extraction complete! Outputs in", OUTPUT_DIR)

if __name__ == "__main__":
    run_lime()
