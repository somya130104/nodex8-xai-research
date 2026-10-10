# Data Lineage Report: OOS Dataset
> Reference Benchmark: DEV

## Data Custodian

The data lineage pipeline for OOS shows 0 anomalies across 2 stages.

**CRITICAL ISSUE DETECTED**:
None.

**WARNING**:
None.

**Data Quality Trend**:
The OOS dataset, representing 14,461 records, exhibits excellent data quality. A maximum null percentage of 0.0% across all features indicates no missing values, which is a strong positive indicator. The `cont_summary` for OOS shows no null counts for any of its 58 continuous features.

Comparing OOS to the benchmark DEV dataset (33,101 records), both datasets demonstrate consistent data quality. The `cont_summary` for DEV also shows 0 null counts for all features.

Distribution drift analysis between the DEV benchmark and the OOS evaluation dataset reveals no significant anomalies. The `FE_LSDV` model evaluation shows a `model_PSI` of 0.000472, with all 40 features reported as "Stable". The highest `Feature_PSI` observed is 0.001715 for `weeks_since_promo`, which is well within acceptable stability thresholds. Similarly, the `XGBoost` model evaluation reports a `model_PSI` of 0.000946, with all 41 features also classified as "Stable". The highest `Feature_PSI` in this evaluation is also 0.001715 for `weeks_since_promo`.

Overall, the data quality trend from raw data (implied by the absence of critical/warning issues) through feature engineering (as evidenced by stable drift metrics) to model input remains consistently high and stable between the OOS and DEV datasets.

**Recommended Actions**:
1.  Continue regular monitoring of data quality metrics and distribution drift for the OOS dataset.
2.  Establish baseline PSI thresholds for "Stable", "Monitor", and "Drifted" statuses to automate anomaly detection.
3.  Investigate the potential upstream sources of the `weeks_since_promo` feature if its PSI, while currently stable, is a key driver for downstream models, to ensure its continued reliability.

---
