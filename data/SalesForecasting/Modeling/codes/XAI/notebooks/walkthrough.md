# XAI — Model Performance Results

We have implemented a generic evaluation suite and generated a consolidated performance report for all model types in the **XAI** project.

---

## 1. Metrics Framework
A generic utility `xai_utils.py` was built to provide consistent evaluation across different model algorithms:
- **Standard Regression**: MAE, RMSE, MAPE, R-Squared, Adjusted R-Squared.
- **Business Accuracy**: `1 - MAPE` percentage.
- **Precision / Recall / F1**: Calculated using a **±20% tolerance** around the actual value.
- **Confidence Buckets**:
    - **High**: Error ≤ 10%
    - **Medium**: 10% < Error ≤ 25%
    - **Low**: Error > 25%

---

## 2. Model Comparison (Validation Set)

| Model | Accuracy % | Precision % | F1 Score % | R-Squared | High Conf % | Low Conf % |
|-------|------------|-------------|------------|-----------|-------------|------------|
| **LightGBM** | **80.52%** | 54.68% | 54.68% | 0.933 | 36.07% | 44.59% |
| **XGBoost** | 79.59% | **56.07%** | **56.07%** | **0.940** | **36.47%** | **44.18%** |
| **ARDL Adjusted** | 0.00% | 16.01% | 16.01% | -0.233 | 5.22% | 86.66% |
| **ARDL Base** | 0.00% | 15.64% | 15.64% | -0.210 | 5.26% | 86.65% |

---

## 3. Group-Level Comparison (Validation Set)

Performance varies significantly across different brands, highlighting which product lines are more predictable.

| Segment | Model | Samples | Accuracy % | F1 Score % | High Conf % |
|-------|-------|---------|------------|------------|-------------|
| **TOPS** | XGBoost | 2,192 | **87.35%** | **81.30%** | **60.49%** |
| **YOCRNCH** | XGBoost | 249 | 84.56% | 40.56% | 24.90% |
| **CHOBANI** | XGBoost | 3,437 | 84.09% | 58.66% | 38.70% |
| **YOPLAIT** | XGBoost | 1,980 | 80.99% | 60.71% | 39.55% |
| **DANNON** | XGBoost | 2,814 | 68.95% | 46.80% | 27.29% |
| **MULLER** | XGBoost | 390 | 0.00% | 0.00% | 0.00% |

- **Top Performers**: TOPS and CHOBANI show excellent predictability with 84%+ accuracy.
- **Challenges**: Brands like MULLER and CABOT show very low accuracy, likely due to data sparsity or extreme volatility.

---

## 4. Feature Importance (Global SHAP)

Global Explanations highlight the fundamental differences between how the models learn:

| Rank | XGBoost / LightGBM | ARDL (Adjusted) |
|------|--------------------|-----------------|
| **1.** | `qty_diff_7` (~30-31%) | `price_discount_pct` (~48%) |
| **2.** | `qty_diff_1` (~16-18%) | `price_discount` (~32%) |
| **3.** | `qty_pct_change_7` (~12-13%) | `PROMOTION_FLG_NUM` (~20%) |

- **Tree Models**: Rely almost entirely on historical interaction and lag differences (`qty_diff_7`). They identify time-series auto-regressive patterns effectively.
- **Hierarchical Linear (ARDL)**: Relies heavily on **pricing and promotional flags**. Historical lags are assigned essentially zero weight directly, because the base brand models largely act as constants, and adjustments are driven purely by price.

---

## 5. SHAP Explanations by Decile & Brand

To understand if features drive demand differently for low vs. high volume periods, we segmented the datasets into **10 Deciles** based on the actual `QUANTITY`.
Additionally, we computed Global Importance driven specifically by **Segment** identity.

- **Local SHAP Matrix**: Every row in the local SHAP CSVs (`xgb_shap_val.csv`, etc.) now includes a `Decile` column and can be joined to `Segment`.
- **Global SHAP**: We computed the mean absolute SHAP feature importances specifically aggregated by:
    - **Overall Validation**: `global_feature_importance.csv`
    - **Decile**: `global_feature_importance_by_decile.csv`
    - **Segment (Major Group)**: `global_feature_importance_by_brand.csv`

**Stability Check (Explicit Decile vs Global Calculation):**
To ensure reliability, we explicitly recalculated local SHAP values on data strictly isolated to each Decile for all 4 models:
- **XGBoost & LightGBM (TreeExplainer)**: Yielded exactly **0.0 maximum difference**. TreeExplainer calculates expectations based on the structural paths of the model's trees during training, meaning local SHAP values are perfectly stable and independent of how the scoring data is sliced.
- **Panel ARDL (Base & Adjusted)**: Yielded an average shift of ~2.18. Linear SHAP calculations (`coefficient * (X - Expected_X)`) inherently shift when restricting the background `Expected_X` strictly to the decile's mean.

---

## 6. Sensitivity Metrics (SHAP Variance)

To measure how sensitive the model's output is to changes in specific features, we analyzed the **Standard Deviation of Local SHAP Values** per feature. 
A high SHAP standard deviation indicates that a feature's contribution varies significantly across different predictions (high sensitivity).

We computed this SHAP Sensitivity at three distinct aggregation levels:
1. **Global Sensitivity**: Spread across the entire Validation dataset.
2. **Decile-Level Sensitivity**: Observing if specific volume deciles exhibit higher variance for certain features.
3. **Segment-Level Sensitivity**: Observing if the models are mathematically more sensitive to features for specific brands.

*Example Insight (XGBoost Global)*: The model is highly sensitive to `qty_diff_7` (Std Dev ~ 5.04), meaning its contribution swings wildly depending on the recent sales trajectory, effectively acting as the primary dynamic lever for the prediction.

---

## 7. LIME Explanations (Local Interpretable Model-Agnostic Explanations)

In addition to SHAP, we implemented **LIME** via the `lime_tabular.LimeTabularExplainer`. LIME explains individual predictions by fitting a local linear surrogate model around the observation's input space using random perturbations.

Because LIME is highly computationally intensive (requiring ~5,000 permutations per individual prediction), we applied a **robust 10% stratified sampling strategy** aligned by Cust_ID and Decile to ensure balanced representation across the forecasting space without requiring days of compute time.

**Available LIME Outputs**:
1. **Local LIME Matrices**: The individual feature weights for the sampled observations. (`XAI/Outputs/lime_local/`)
2. **Global LIME Feature Importance**: Computed by averaging the *absolute* local LIME weights for each feature across the entire 10% sample. (`XAI/Outputs/lime_global/global_feature_importance.csv`)
3. **Decile-Based Global LIME**: The global importance aggregated specifically within each decile stratum. (`XAI/Outputs/lime_global/global_feature_importance_by_decile.csv`)
4. **Segment-Based Global LIME**: The global importance aggregated specifically for each major brand group. (`XAI/Outputs/lime_global/global_feature_importance_by_brand.csv`)

---

## 8. Surrogate Regression Explanations

To provide a highly interpretable, explicitly quantifiable explanation of the black-box algorithms, we implemented a **Panel Surrogate Regression** approach.

We structured the surrogate perfectly mimicking the distinct Major Groups:
- For each algorithm's absolute predictions (e.g. `xgb_predicted_quantity`), we fitted a separate **Linear Regression** model isolated to each **Segment**.
- This yields mathematically explicit linear coefficients per feature, bounded strictly to the dynamics of the specific algorithm.

**Available Surrogate Outputs**:
1. **Local Decompositions**: Explicit observation-level feature part-worths spanning the entire Validation set. Calculated as `Part-Worth = Coefficient_brand * Feature_Actual`. Saved per model in `XAI/Outputs/surrogate_local/`.
2. **Global Surrogate Coefficients by Brand**: The distinct, interpretable linear weights extracted natively from the Surrogate Regression for each Brand and Feature. (`XAI/Outputs/surrogate_global/global_surrogate_coefficients_by_brand.csv`)
3. **Overall Global Importance**: The mean absolute local decompositions aggregated across the entire Validation dataset.

## 9. Multi-Dataset, Multi-Model, & Multi-Group Generalization Architecture

To ensure the XAI pipeline is future-proof and "plug-and-play" ready for newly developed datasets or new distinct models:

**All analytical processes** (**Global Performance**, **Group Performance**, **SHAP**, **LIME**, and **Surrogate Regressions**) have been fully refactored to be **Dataset-Agnostic**, **Model-Agnostic**, and **Grouping-Agnostic**:
- The `00_input_manifesto` process serves as the **Single Source of Truth**, defining the unconstrained configuration object and exporting it to `XAI/Outputs/manifesto_config.pkl`. 
- The schema dynamically exports flattened *execution runs* specifying: `dataset_name`, `path`, `model_domain` (e.g. Regression), `model_type` (e.g. Linear), `model_name`, and `pred_col`.
- **All downstream pipelines** dynamically ingest this exact `.pkl` file rather than hardcoding logic.
- They iterate through all permutations seamlessly and correctly select standard evaluators vs explainers.
- **Explainability Output Files** are dynamically generated appending the exact dataset to the explicit name (e.g., `...xgboost_shap_dev.csv`).
- **Global Summaries & Performance Tables** natively append the unified `Dataset`, `Model`, `Model_Type`, and `Model_Domain` explicitly as column metadata to allow instantaneous slice-and-dice multi-dimensional filtering natively in tools like Tableau or Excel.

---

## 10. Key Observations
- **Tree-based models (XGBoost/LightGBM)** are vastly superior to the ARDL approach for this dataset.
- **Brand Identity** is a strong determinant of accuracy; generic forecasting challenges vary by up to 80 points between brands.
- **Algorithm Behavior**: XGBoost prioritizes lag vectors, whereas Panel ARDL is exceptionally price/promotion-sensitive.
- **Volume Dependence**: Feature importance scales dynamically with sales volume (e.g. historical momentum matters much more during high-sales periods).

## 10. Drift Analysis & Stability Scoring

A dedicated module (`07_drift_analysis`) has been constructed to continuously monitor prediction metrics out-of-time.
- **Population Stability Index (PSI)** is utilized natively on both the model predictions natively (Model Stability) and the discrete input distributions (Feature Stability) comparing mapping evaluation datasets strictly against their `Dev` set baseline.
- **Continuous Dimension Percentiles**: Every prediction numerical feature executes an extraction of 13 percentiles ranging strictly between 0.5% and 99.5% for continuous evaluation.
- **Categorical Top-K Analysis**: Identifies and extracts precise proportional frequencies for the Top 10 discrete identifiers for any categorical features inputted into the model.

## 7. Files Produced
- **Utility**: [XAI/src/shap_utils.py](file:///Users/angshubh/DemandForecasting/XAI/src/shap_utils.py)
- **Notebooks**: [03_shap](file:///Users/angshubh/DemandForecasting/XAI/notebooks/03_shap_explanations.ipynb)
- **Local SHAP**: 8 comprehensive row-by-row feature matrices saved in `XAI/Outputs/shap_local/` (now with `Decile` tracking).
- **Global SHAP**: [global_feature_importance.csv](file:///Users/angshubh/DemandForecasting/XAI/Outputs/shap_global/global_feature_importance.csv) and [global_feature_importance_by_decile.csv](file:///Users/angshubh/DemandForecasting/XAI/Outputs/shap_global/global_feature_importance_by_decile.csv)

