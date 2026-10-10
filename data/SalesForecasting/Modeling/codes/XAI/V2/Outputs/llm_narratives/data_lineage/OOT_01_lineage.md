# Data Lineage Report: OOT_01 Dataset
> Reference Benchmark: DEV

## Data Custodian

The data lineage pipeline for OOT_01 shows **significant anomalies across two primary stages** (Feature Engineering and Model Input) impacting data quality.

**CRITICAL ISSUE DETECTED**

*   **Severe Distribution Break:** The most critical issue is the substantial distribution drift observed between the `DEV` benchmark dataset and the `OOT_01` evaluation dataset. This is evidenced by the `FE_LSDV` model reporting a `model_PSI` of 1.26 and the `XGBoost` model reporting a `model_PSI` of 0.25, both classified as "Significant Drift".
    *   Specifically, features like `quarter` (PSI: 8.20), `global_t` (PSI: 8.18), `week_sin` (PSI: 4.70), and `week_cos` (PSI: 3.81) exhibit extremely high PSI values, indicating a fundamental shift in their distributions.
    *   Other features such as `August` (PSI: 1.50), `September` (PSI: 1.55), and `momentum_4_13` (PSI: 1.32) also show significant drift.
    *   This widespread drift suggests that the feature engineering process is not consistently transforming data to maintain similar distributions between training/benchmark data and out-of-time data, leading to a high risk of model degradation.

**WARNING**

*   **Distribution Drift in Multiple Features:** Beyond the critical issues, a considerable number of features are flagged for "Significant Drift" or "Monitor" status across both model drift checks.
    *   **Significant Drift:** `Christmas` (PSI: 0.28), `NewYear` (PSI: 0.91), `Spring` (PSI: 0.78), `Thanksgiving` (PSI: 0.76), `VetreansDay` (PSI: 0.76), `is_year_end` (PSI: 0.63), `is_year_start` (PSI: 0.66), `pre_promo_1` (PSI: 0.32), `pre_promo_2` (PSI: 0.56), `promo_intensity` (PSI: 0.43), `weeks_to_next_promo` (PSI: 0.38), `wow_growth` (PSI: 0.31), `zero_frac_13` (PSI: 0.28).
    *   **Monitor:** `Weekly_Index` (PSI: 0.19), `cust_hist_mean` (PSI: 0.18), `holiday_x_windex` (PSI: 0.23), `roll_mean_13` (PSI: 0.18), `roll_mean_4` (PSI: 0.14), `roll_std_4` (PSI: 0.18), `weeks_since_promo` (PSI: 0.11), `UnitSales` (PSI: 0.15).
    *   These indicate that while not as severe as the critical issues, these features are also showing changes in their distributions that warrant attention.

**Data Quality Trend**

*   **Raw Data to Feature Engineering:** The `OOT_01` and `DEV` datasets both report `null_pct_max` of 0.0 and no `high_null_features`. This suggests that the initial data ingestion and any upstream transformations leading to the raw features are robust in terms of null handling.
*   **Feature Engineering to Model Input:** The significant distribution drift identified in the `FE_LSDV` and `XGBoost` model drift reports indicates a major quality degradation at the feature engineering stage when producing inputs for the models. The `OOT_01` dataset, when used as model input, has distributions that deviate substantially from the `DEV` benchmark.
    *   The `DEV` dataset (benchmark) has 58 continuous and 23 categorical features.
    *   The `OOT_01` dataset has 50 continuous and 31 categorical features. While the number of features differs, the primary concern is the distributional shift of shared features.
    *   For instance, `global_t` is present in `DEV` with a mean of 86.24 and std of 42.03, but its distribution in `OOT_01` is not directly comparable in the provided `OOT_01` descriptive stats. However, the drift report clearly shows a massive divergence. Similarly, `week_sin` and `week_cos` show significant drift, implying transformations applied to temporal features are not consistent.

**Recommended Actions**

1.  **Investigate Feature Engineering Logic for Drifted Features:** Conduct a thorough audit of the feature engineering code responsible for creating `quarter`, `global_t`, `week_sin`, `week_cos`, `August`, `September`, and `momentum_4_13`. Identify and rectify any logic that causes distributional differences between training/benchmark data and out-of-time data.
2.  **Implement Robust Feature Transformation Validation:** Enhance the feature engineering pipeline to include automated checks for distribution drift against established benchmarks (e.g., `DEV` dataset) for all generated features. This should trigger alerts for any PSI values exceeding predefined thresholds.
3.  **Review Feature Set Consistency:** Verify that the feature sets between `DEV` and `OOT_01` are as consistent as possible in terms of definition and transformation. Investigate the discrepancy in the number of categorical features (31 in `OOT_01` vs. 23 in `DEV`) and ensure no critical categorical features are being lost or improperly handled in `OOT_01`.
4.  **Analyze `global_t` and Temporal Features:** Pay special attention to the `global_t` feature and temporal features (`week_sin`, `week_cos`, `quarter`, month indicators) as they show the most severe drift. Understand the underlying data sources and transformations for these features to pinpoint the root cause.
5.  **Establish Data Quality Monitoring for Model Inputs:** Implement a continuous monitoring system for the `OOT_01` dataset once it's prepared as model input. This system should track key data quality metrics, including PSI scores for critical features, and alert the engineering team to any further deviations.

---
