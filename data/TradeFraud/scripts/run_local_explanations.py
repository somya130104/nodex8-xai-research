"""
Phase 1.1 — Local (per-instance) SHAP and LIME explanations, TradeFraud.

This is the "Level 2" explainability Angshuman asked for: not "which
features matter on average" (global) or "within this segment" (cohort), but
"for THIS ONE client, why did the model say what it said." This is also the
level a real adverse-action notice (the regulatory requirement behind all of
this) is built from.

For each of the 4 baseline models (XGBoost, Random Forest, LightGBM,
Logistic Regression), 15 representative Validation-OOS clients are picked —
5 highest-risk, 5 borderline (closest to that model's own fixed decision
threshold from Phase 0), 5 misclassified — and BOTH SHAP and LIME are run on
exactly those same 15 clients, so the two sets of outputs can be compared
case-by-case.

Outputs (all in XAI_Outputs/):
  - shap_local_samples.csv   one row per (model, client), one column per feature
  - lime_local_samples.csv   same shape, from LIME
  - local_explanations.csv   one row per (model, client): a plain-English
                              explanation generated from the top-3 SHAP features
  - baseline_instance_cli.csv  SHAP-vs-LIME agreement (see
                              src/eval_utils.baseline_instance_tau — a
                              placeholder for Phase 4's real CLI) per case
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
from src.eval_utils import dedupe_rows, baseline_instance_tau
from src.local_explain_utils import (
    select_classification_cases, top_features_signed, text_explanation_classification,
)

warnings.filterwarnings("ignore")

DATA_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/Modeling_Data_Enriched/"
MODEL_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/Model_Outputs/"
OUTPUT_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/XAI_Outputs/"

EXCLUDE_COLS = ["Client", "Client_ID", "Date", "Dispute", "Segment", "Activeness_Segment"]
N_EACH = 5          # 5 per bucket x 3 buckets = 15 cases per model
LIME_SAMPLES = 1000  # same setting run_baseline_lime.py uses, for consistency

MODELS = [
    ("Model2_XGBoost", "model2_xgboost.pkl", "Tree"),
    ("Model3_RandomForest", "model3_rf.pkl", "Tree"),
    ("Model5_LightGBM", "model5_lightgbm.pkl", "Tree"),
    ("Model6_LogReg", "model6_logreg.pkl", "Linear"),
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


def get_pred_prob(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)  # LightGBM's native Booster API returns probabilities directly


def get_threshold(model_name):
    scorecard = pd.read_csv(os.path.join(MODEL_DIR, model_name, "performance_scorecard.csv"))
    row = scorecard[scorecard["Dataset"] == "Validation OOS"]
    return float(row["Threshold_Used"].iloc[0])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    _, X_train, y_train, feat_cols = load_dataset("train.csv")
    bg_data = X_train.sample(min(200, len(X_train)), random_state=42)
    df_oos, X_oos, y_oos, _ = load_dataset("val_oos.csv")
    client_ids = df_oos["Client_ID"].values

    shap_rows, lime_rows, text_rows, cli_rows = [], [], [], []

    for model_name, pkl_name, family in MODELS:
        print(f"\n--- {model_name} ---")
        model_path = os.path.join(MODEL_DIR, model_name, pkl_name)
        if not os.path.exists(model_path):
            print(f"  Skipping (model file not found)")
            continue

        with open(model_path, "rb") as f:
            model_pkg = pickle.load(f)

        if family == "Linear":
            model = model_pkg["model"]
            scaler = model_pkg["scaler"]
            X_oos_proc = pd.DataFrame(scaler.transform(X_oos), columns=feat_cols)
            bg_proc = pd.DataFrame(scaler.transform(bg_data), columns=feat_cols)
        else:
            model = model_pkg
            X_oos_proc = X_oos
            bg_proc = bg_data

        pred_prob = get_pred_prob(model, X_oos_proc)
        threshold = get_threshold(model_name)

        cases = select_classification_cases(pred_prob, y_oos.values, threshold, n_each=N_EACH)
        idx_to_case_type = {int(idx): case_type for case_type, idx_arr in cases.items() for idx in idx_arr}
        selected_idx = sorted(idx_to_case_type.keys())
        print(f"  Threshold={threshold:.4f} | selected {len(selected_idx)} cases "
              f"({ {k: len(v) for k, v in cases.items()} })")

        X_sel = X_oos_proc.iloc[selected_idx]

        # ---- SHAP, on just these instances ----
        print("  Computing local SHAP...")
        if family == "Tree":
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_sel, check_additivity=False)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]
        else:
            explainer = shap.LinearExplainer(model, bg_proc)
            shap_vals = explainer.shap_values(X_sel)

        # ---- LIME, on the SAME instances ----
        print("  Computing local LIME...")
        if family == "Linear":
            lime_bg = bg_proc.values
            def predict_fn(x):
                return model.predict_proba(pd.DataFrame(x, columns=feat_cols))
        else:
            lime_bg = X_train.values
            if hasattr(model, "predict_proba"):
                def predict_fn(x):
                    return model.predict_proba(pd.DataFrame(x, columns=feat_cols))
            else:
                def predict_fn(x):
                    probs = model.predict(pd.DataFrame(x, columns=feat_cols))
                    return np.vstack([1 - probs, probs]).T

        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            lime_bg[:5000], feature_names=feat_cols,
            class_names=["Normal", "Dispute"], mode="classification", random_state=42,
        )

        lime_vals = np.zeros((len(selected_idx), len(feat_cols)))
        for row_i, oos_i in enumerate(selected_idx):
            row_vec = X_sel.iloc[row_i].values
            exp = lime_explainer.explain_instance(row_vec, predict_fn,
                                                   num_features=len(feat_cols),
                                                   num_samples=LIME_SAMPLES)
            for feat_idx, weight in exp.as_map()[1]:
                lime_vals[row_i, feat_idx] += weight

        # ---- Assemble outputs for each selected case ----
        for row_i, oos_i in enumerate(selected_idx):
            case_type = idx_to_case_type[oos_i]
            client_id = client_ids[oos_i]
            true_label = int(y_oos.values[oos_i])
            prob = float(pred_prob[oos_i])
            shap_vec = shap_vals[row_i]
            lime_vec = lime_vals[row_i]

            shap_rows.append({
                "Model": model_name, "Client_ID": client_id, "Case_Type": case_type,
                "True_Label": true_label, "Predicted_Prob": prob, "Threshold_Used": threshold,
                **dict(zip(feat_cols, shap_vec)),
            })
            lime_rows.append({
                "Model": model_name, "Client_ID": client_id, "Case_Type": case_type,
                "True_Label": true_label, "Predicted_Prob": prob, "Threshold_Used": threshold,
                **dict(zip(feat_cols, lime_vec)),
            })

            top3 = top_features_signed(shap_vec, feat_cols, k=3)
            explanation = text_explanation_classification(top3, prob, threshold)
            text_rows.append({
                "Model": model_name, "Client_ID": client_id, "Case_Type": case_type,
                "True_Label": true_label, "Predicted_Prob": round(prob, 4),
                "Threshold_Used": threshold, "Explanation": explanation,
            })

            tau = baseline_instance_tau(shap_vec, lime_vec)
            cli_rows.append({
                "Model": model_name, "Client_ID": client_id, "Case_Type": case_type,
                "SHAP_vs_LIME_Tau": tau,
            })

        print(f"  Done: {len(selected_idx)} cases explained with both SHAP and LIME.")

    pd.DataFrame(shap_rows).to_csv(os.path.join(OUTPUT_DIR, "shap_local_samples.csv"), index=False)
    pd.DataFrame(lime_rows).to_csv(os.path.join(OUTPUT_DIR, "lime_local_samples.csv"), index=False)
    pd.DataFrame(text_rows).to_csv(os.path.join(OUTPUT_DIR, "local_explanations.csv"), index=False)
    pd.DataFrame(cli_rows).to_csv(os.path.join(OUTPUT_DIR, "baseline_instance_cli.csv"), index=False)

    print(f"\n✅ Local explanations complete. 4 files written to {OUTPUT_DIR}")
    print("   shap_local_samples.csv, lime_local_samples.csv, "
          "local_explanations.csv, baseline_instance_cli.csv")


if __name__ == "__main__":
    main()
