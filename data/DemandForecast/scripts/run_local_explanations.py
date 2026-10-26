"""
Phase 1.1 — Local (per-instance) SHAP and LIME explanations, DemandForecast.

Same purpose as TradeFraud's run_local_explanations.py, adapted for a
regression problem, which has no decision threshold and no false
positive/negative concept. The three buckets here are regression's honest
equivalents (see src/local_explain_utils.select_regression_cases):
  - "high_value":  5 instances with the highest PREDICTED demand
  - "borderline":  5 instances whose prediction is closest to the median
                   prediction (the "typical, ambiguous" case)
  - "high_error":  5 instances with the largest |actual - predicted| error
                   (the regression equivalent of "misclassified")

For each of the 4 baseline models (XGBoost, Random Forest, LightGBM, Linear
Regression), SHAP and LIME are run on exactly the same 15 selected instances.

Outputs (all in XAI_Outputs/):
  - shap_local_samples.csv, lime_local_samples.csv, local_explanations.csv,
    baseline_instance_cli.csv  (same shapes/meaning as TradeFraud's)
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular

sys.path.append("/Users/somyabhadada/nodex8")
from src.eval_utils import baseline_instance_tau
from src.local_explain_utils import (
    select_regression_cases, top_features_signed, text_explanation_regression,
)

warnings.filterwarnings("ignore")

DATA_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/samples/"
MODEL_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/Model_Outputs/"
OUTPUT_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/XAI_Outputs/"

EXCLUDE_COLS = ["RowNum", "Date", "UPC", "COMP_STR_NO", "QUANTITY", "BRAND", "RET_LIR_ID"]
CATEGORICAL_COLS = ["PROMOTION_FLG", "C_PROMOTION_FLG"]
N_EACH = 5
LIME_SAMPLES = 500  # matches run_baseline_lime.py's setting for this dataset

MODELS = [
    ("XGBoost", "model_xgboost.pkl", "Tree"),
    ("RandomForest", "model_rf.pkl", "Tree"),
    ("LightGBM", "model_lightgbm.pkl", "Tree"),
    ("LinearRegression", "model_ols.pkl", "Linear"),
]


def load_dataset(fname, dummify=False):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    if dummify:
        df_model = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns],
                                   drop_first=True)
    else:
        df_model = df.copy()
        for col in CATEGORICAL_COLS:
            if col in df_model.columns:
                df_model[col] = df_model[col].map(
                    {"Y": 1, "N": 0, "1": 1, "0": 0, 1: 1, 0: 0}).fillna(0)
    feat_cols = [c for c in df_model.columns if c not in EXCLUDE_COLS]
    X = df_model[feat_cols]
    return df, X, feat_cols


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    _, X_train, _ = load_dataset("dev.csv", dummify=False)
    _, X_train_dummy, _ = load_dataset("dev.csv", dummify=True)
    X_train_dummy = X_train_dummy.fillna(0).sample(min(200, len(X_train_dummy)), random_state=42)

    df_oos, X_oos, feat_cols_tree = load_dataset("oos.csv", dummify=False)
    _, X_oos_dummy, feat_cols_lin = load_dataset("oos.csv", dummify=True)
    X_oos_dummy = X_oos_dummy.fillna(0)

    actual_qty = df_oos["QUANTITY"].fillna(0).values
    row_ids = df_oos["RowNum"].values

    shap_rows, lime_rows, text_rows, cli_rows = [], [], [], []

    for model_name, pkl_name, family in MODELS:
        print(f"\n--- {model_name} ---")
        model_path = os.path.join(MODEL_DIR, model_name, pkl_name)
        if not os.path.exists(model_path):
            print("  Skipping (model file not found)")
            continue

        with open(model_path, "rb") as f:
            model_pkg = pickle.load(f)

        if family == "Linear":
            model = model_pkg["model"]
            scaler = model_pkg["scaler"]
            expected_feats = model_pkg["features"]

            X_eval = X_oos_dummy.copy()
            for col in expected_feats:
                if col not in X_eval.columns:
                    X_eval[col] = 0
            X_eval = X_eval[expected_feats]

            X_bg_eval = X_train_dummy.copy()
            for col in expected_feats:
                if col not in X_bg_eval.columns:
                    X_bg_eval[col] = 0
            X_bg_eval = X_bg_eval[expected_feats]

            X_oos_scaled = scaler.transform(X_eval)
            bg_scaled = scaler.transform(X_bg_eval)
            feat_cols = expected_feats

            pred = model.predict(X_oos_scaled)

            def predict_fn(x):
                return model.predict(scaler.transform(pd.DataFrame(x, columns=expected_feats)))

            lime_bg = X_bg_eval.values
            X_oos_for_shap = X_oos_scaled
            bg_for_shap = bg_scaled
        else:
            model = model_pkg
            feat_cols = feat_cols_tree
            pred = model.predict(X_oos)

            def predict_fn(x):
                return model.predict(pd.DataFrame(x, columns=feat_cols_tree))

            lime_bg = X_train.values
            X_oos_for_shap = X_oos
            bg_for_shap = None  # TreeExplainer doesn't need a background sample

        cases = select_regression_cases(pred, actual_qty, n_each=N_EACH)
        idx_to_case_type = {int(idx): case_type for case_type, idx_arr in cases.items() for idx in idx_arr}
        selected_idx = sorted(idx_to_case_type.keys())
        print(f"  selected {len(selected_idx)} cases "
              f"({ {k: len(v) for k, v in cases.items()} })")

        X_sel_shap = (X_oos_for_shap.iloc[selected_idx] if hasattr(X_oos_for_shap, "iloc")
                      else X_oos_for_shap[selected_idx])
        X_sel_plain = X_oos.iloc[selected_idx] if family != "Linear" else X_eval.iloc[selected_idx]

        print("  Computing local SHAP...")
        if family == "Tree":
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_sel_shap, check_additivity=False)
        else:
            explainer = shap.LinearExplainer(model, bg_for_shap)
            shap_vals = explainer.shap_values(X_sel_shap)

        print("  Computing local LIME...")
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            lime_bg[:2000], feature_names=feat_cols, mode="regression", random_state=42,
        )
        lime_vals = np.zeros((len(selected_idx), len(feat_cols)))
        for row_i, oos_i in enumerate(selected_idx):
            row_vec = X_sel_plain.iloc[row_i].values
            exp = lime_explainer.explain_instance(row_vec, predict_fn,
                                                   num_features=len(feat_cols),
                                                   num_samples=LIME_SAMPLES)
            for feat_idx, weight in exp.as_map()[1]:
                lime_vals[row_i, feat_idx] += weight

        for row_i, oos_i in enumerate(selected_idx):
            case_type = idx_to_case_type[oos_i]
            row_id = row_ids[oos_i]
            actual = float(actual_qty[oos_i])
            predicted = float(pred[oos_i])
            shap_vec = shap_vals[row_i]
            lime_vec = lime_vals[row_i]

            shap_rows.append({
                "Model": model_name, "RowNum": row_id, "Case_Type": case_type,
                "Actual_Quantity": actual, "Predicted_Quantity": predicted,
                **dict(zip(feat_cols, shap_vec)),
            })
            lime_rows.append({
                "Model": model_name, "RowNum": row_id, "Case_Type": case_type,
                "Actual_Quantity": actual, "Predicted_Quantity": predicted,
                **dict(zip(feat_cols, lime_vec)),
            })

            top3 = top_features_signed(shap_vec, feat_cols, k=3)
            explanation = text_explanation_regression(top3, predicted)
            text_rows.append({
                "Model": model_name, "RowNum": row_id, "Case_Type": case_type,
                "Actual_Quantity": round(actual, 3), "Predicted_Quantity": round(predicted, 3),
                "Explanation": explanation,
            })

            tau = baseline_instance_tau(shap_vec, lime_vec)
            cli_rows.append({
                "Model": model_name, "RowNum": row_id, "Case_Type": case_type,
                "SHAP_vs_LIME_Tau": tau,
            })

        print(f"  Done: {len(selected_idx)} cases explained with both SHAP and LIME.")

    pd.DataFrame(shap_rows).to_csv(os.path.join(OUTPUT_DIR, "shap_local_samples.csv"), index=False)
    pd.DataFrame(lime_rows).to_csv(os.path.join(OUTPUT_DIR, "lime_local_samples.csv"), index=False)
    pd.DataFrame(text_rows).to_csv(os.path.join(OUTPUT_DIR, "local_explanations.csv"), index=False)
    pd.DataFrame(cli_rows).to_csv(os.path.join(OUTPUT_DIR, "baseline_instance_cli.csv"), index=False)

    print(f"\n✅ Local explanations complete. 4 files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
