# XAI — Sensitivity Metrics Implementation Plan

## Background

Sensitivity metrics measure how much a model's output changes in response to changes in feature values. This will be computed across three explainability frameworks — **SHAP**, **LIME**, and **Surrogate Regressions** — at three levels of granularity: **Global**, **Decile**, and **Major Group (Segment)**.

## User Review Required

> [!NOTE]
> **Surrogate Regressions**: We train a simple, interpretable model (Linear Regression) on the predictions of the black-box model. The surrogate's feature coefficients naturally encode sensitivity (how much the output is expected to change per unit change in each input). The fidelity R-squared of the surrogate vs the original model is also reported as a "stability" indicator.

> [!NOTE]
> **LIME Sensitivity**: LIME fits a local linear model for each observation. We will compute LIME for a representative sample (~5% of each dataset/decile/brand group) and aggregate to derive sensitivity metrics. Full row-by-row LIME would be extremely slow given 110+ models.

> [!IMPORTANT]
> **Models Covered**: XGBoost, LightGBM, ARDL Base, ARDL Adjusted.

## Proposed Changes

### [NEW] `XAI/src/sensitivity_utils.py`
A new utility module containing helper functions for:
- `compute_shap_sensitivity(shap_df, feature_cols)`: Using existing local SHAP matrices, compute per-feature sensitivity as `std(SHAP values)` — a measure of variability in feature contribution.
- `compute_lime_sensitivity(df, predict_fn, feature_cols, n_samples)`: Run LIME on a sample, aggregate to produce per-feature importance and sensitivity.
- `compute_surrogate_sensitivity(df, preds, feature_cols)`: Fit a LinearRegression surrogate on actual features vs model predictions; use model coefficients as sensitivity metrics with R² as fidelity indicator.
- Helper: `aggregate_sensitivity(result_df, group_col=None)`: Aggregates sensitivity by Decile or Segment group.

### [NEW] `XAI/notebooks/04_sensitivity.ipynb`
A single notebook covering all three levels for each explainability framework:

| Section | Method | Level |
|---------|--------|-------|
| 2 | SHAP Sensitivity | Global |
| 3 | SHAP Sensitivity | Decile |
| 4 | SHAP Sensitivity | Segment |
| 5 | LIME Sensitivity | Global |
| 6 | LIME Sensitivity | Decile |
| 7 | LIME Sensitivity | Segment |
| 8 | Surrogate Regression | Global |
| 9 | Surrogate Regression | Decile |
| 10 | Surrogate Regression | Segment |

### [NEW] `XAI/notebooks/05_lime_explanations.ipynb`
A new notebook mirroring the SHAP process, but utilizing Local Interpretable Model-Agnostic Explanations (LIME):
- Uses `lime.lime_tabular.LimeTabularExplainer` to fit local linear surrogate models around individual predictions.
- **Sampling Strategy**: Because LIME requires ~5,000 perturbed predictions *per observation* to fit its local surrogate, running it on all 35,000+ rows would take hours. We will apply a **10% stratified sample** (stratified across Deciles) to generate the Local LIME matrices.
- Computes **Local LIME** matrices over the sample (feature weights per prediction).
- Aggregates weights to compute **Global LIME** and **Decile-based Global LIME** feature importances.
- Output paths will mirror SHAP: `XAI/Outputs/lime_local/` and `XAI/Outputs/lime_global/`.

### [NEW] `XAI/notebooks/06_surrogate_explanations.ipynb`
A new notebook to generate **Surrogate Regression** explanations:
- Fits a highly interpretable model (Linear Regression) to explicitly mimic the predictions of the complex inputs (XGBoost, LightGBM, ARDL Base, ARDL Adjusted).
- **Panel Data Structure**: The linear regressions will be fitted distinctly for each **Major Group (Segment)**, ensuring the surrogate respects the panel nature of the data. 
    - *Dependent Variable ($y$)*: The specific prediction output of the model (e.g. `xgb_predicted_quantity`).
    - *Explanatory Variables ($X$)*: The base features from the `.pkl` artifact.
- **Global Explanations**: Calculates the global feature importance as the extracted regression coefficients for each Segment.
- **Local Explanations**: Calculates the explicit observation-level feature decomposition using the formula: `Value_ij = Coefficient_j * X_ij` (the actual feature value multiplied by its specific Segment regression coefficient).
- **Local Explanations**: Calculates the explicit observation-level feature decomposition using the formula: `Value_ij = Coefficient_j * X_ij` (the actual feature value multiplied by its specific Segment regression coefficient).
- Outputs will be saved to `XAI/Outputs/surrogate_local/` and `XAI/Outputs/surrogate_global/`.

### [NEW] `XAI/notebooks/07_drift_analysis.ipynb`
A new notebook to perform comprehensive **Drift Analysis**.
- **Benchmark Data**: Uses the `Dev` dataset defined in `manifesto_config.pkl` as the gold-standard benchmark.
- **Model Stability Score**: Compares prediction distributions between the benchmark dataset and other datasets (e.g., `Validation`) using the Population Stability Index (PSI) applied to model predictions.
- **Population Stability Score**: Calculates the PSI sequentially for all input features (continuous and categorical) comparing the evaluation data to the benchmark data.
- **Continuous Variables Statistics**: For every continuous predictive feature, calculates:
  - `Count`, `Null Count`, `Mean`, `Standard Deviation`
  - 13 exact Percentiles: `p0.5, p1, p2.5, p5, p10, p25, p50 (median), p75, p90, p95, p97.5, p99, p99.5`
- **Categorical Variables Statistics**: For every categorical feature, calculates:
  - `Variable Name`, `Count`, `Null Count`
  - `Top Classes (up to 10)`: Including `Class Value`, `Class Count`, and `Class Percentage`.
- **Output**: Stores all comparative results in heavily structured tables at `XAI/Outputs/drift_analysis/`.

## Generalization Standard (Multi-Model, Multi-Dataset, Multi-Group)
To ensure the XAI pipeline is future-proof and "plug-and-play" ready for newly developed datasets (like an Out-Of-Time test set) or completely new distinct models (e.g., a new Deep Learning architecture alongside XGBoost):

**All core analysis notebooks** (`01_model_performance`, `02_group_performance`, `03_shap`, `05_lime`, `06_surrogate`) will be refactored to support **arbitrary datasets**, **arbitrary models**, and **arbitrary grouping variables**.

**Specifically for `00_input_manifesto.ipynb`**:
- **Centralized Configuration**: It will export a single `dataset_configs` array and `group_vars` as a pickled artifact or JSON so that all downstream pipelines (01-06) ingest the identical configuration automatically.
- **Flattened Schema**: As requested, `dataset_configs` will be a flat list of dictionaries perfectly describing every run, utilizing variables: `dataset_name`, `path`, `model_domain` (e.g. regression), `model_type`, `model_name`, and `pred_col`.

**Specifically for Notebooks 01 through 06**:
- **Ingestion**: Instead of defining any configuration logic themselves, they will simply open `XAI/Outputs/manifesto_config.pkl` and loop directly through the master `dataset_configs` list, executing global metrics, local metrics, LIME, SHAP, etc. based dynamically on the `model_domain`.
- **Dynamic Output**: A single `global_model_performance.csv` containing the comprehensive consolidated performance of every model across every dataset.

**Specifically for `02_group_performance.ipynb`**:
- **Group Agnostic**: The notebook will ingest a `group_vars` list (e.g., `['Segment']`, which can be expanded to `['Segment', 'REGION', 'CATEGORY']`).
- **Model & Dataset Agnostic**: It will iterate through the defined `dataset_configs` and `model_configs` exactly like the explainability notebooks.
- **Combined Output**: It will generate a master consolidated table containing all permutations, utilizing `Group_Variable` (e.g., "Segment") and `Group_Value` (e.g., "TOPS") columns instead of a hardcoded "Segment" column.

## Expected Outputs
- `XAI/Outputs/surrogate_local/{model}_surrogate_{dataset}.csv`
- `XAI/Outputs/surrogate_global/global_feature_importance.csv`
- `XAI/Outputs/surrogate_global/global_feature_importance_by_brand.csv`
- `XAI/Outputs/sensitivity/...` (Retained SHAP metrics)
- `XAI/Outputs/sensitivity/shap_sensitivity_decile.csv`
- `XAI/Outputs/sensitivity/shap_sensitivity_brand.csv`
- `XAI/Outputs/sensitivity/lime_sensitivity_global.csv`
- `XAI/Outputs/sensitivity/lime_sensitivity_decile.csv`
- `XAI/Outputs/sensitivity/lime_sensitivity_brand.csv`
- `XAI/Outputs/sensitivity/surrogate_sensitivity_global.csv`
- `XAI/Outputs/sensitivity/surrogate_sensitivity_decile.csv`
- `XAI/Outputs/sensitivity/surrogate_sensitivity_brand.csv`

## Verification Plan
- Check that all 9 output CSVs are produced without errors.
- Verify that the LIME fidelity from sampling is consistent across multiple decile slices.
- Verify Surrogate R² > 0.8 for XGBoost/LightGBM (strong models), and lower for ARDL.
