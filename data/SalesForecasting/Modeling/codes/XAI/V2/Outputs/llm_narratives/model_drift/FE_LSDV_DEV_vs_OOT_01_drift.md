# Model Drift Report: FE_LSDV
> Benchmark: DEV  |  Evaluation: OOT_01  |  Champion Model: No

## Business Executive

The drift analysis for FE_LSDV comparing DEV (benchmark) to OOT_01 shows **Significant Drift** across **20 features**.

**CRITICAL ISSUE DETECTED**

The FE_LSDV model is experiencing significant drift, meaning its predictions are likely becoming less reliable. The overall model drift score is 1.26, indicating a substantial change in the data the model is seeing compared to what it was trained on. This is driven by major shifts in key features:

*   **`quarter`** (PSI 8.20) and **`global_t`** (PSI 8.18) have changed dramatically. These features likely represent core time-based patterns or trends that the model relies on. Their significant drift suggests the model is misinterpreting fundamental aspects of the data.
*   **`week_sin`** (PSI 4.70) and **`week_cos`** (PSI 3.81) also show substantial drift. These features likely capture cyclical weekly patterns. Their change means the model may be misinterpreting weekly sales or activity cycles.
*   Other features like **`August`** (PSI 1.50) and **`September`** (PSI 1.55) also show significant changes, pointing to shifts in monthly patterns.
*   Features like **`momentum_4_13`** (PSI 1.32) and those related to promotions (**`pre_promo_1`**, **`pre_promo_2`**, **`promo_intensity`**) are also drifting. This suggests the model's understanding of sales momentum and promotional impacts is out of sync.

The commercial impact of this significant drift is a likely degradation in forecasting accuracy and business decision-making. For example, the model's ability to predict sales based on time of year, weekly cycles, or promotional impact is compromised. This can lead to poor inventory management, missed sales opportunities, or inefficient marketing spend. The performance metrics confirm this, with **Accuracy_pct dropping from 38.98% to 15.08%** and **MAPE increasing from 0.61 to 0.85** when moving from the benchmark to the out-of-time data. This directly impacts revenue and profitability.

**WARNING**

Several features are showing signs of potential future drift and require monitoring:

*   **`is_year_end`** (PSI 0.63) and **`is_year_start`** (PSI 0.66): While not yet critical, these features, related to year-end and year-start events, are showing notable shifts.
*   **`weeks_to_next_promo`** (PSI 0.38) and **`wow_growth`** (PSI 0.31): These features, related to promotional timing and growth rates, are also showing considerable changes that warrant attention.
*   **`pre_promo_1`** (PSI 0.32) and **`pre_promo_2`** (PSI 0.56): These features, reflecting the period before a promotion, are also drifting significantly.
*   **`promo_intensity`** (PSI 0.43): The intensity of promotions is also changing.

**Stable Features**

A small number of features remain stable, indicating their behavior has not significantly changed:

*   **`any_holiday`** (PSI 0.02), **`is_promo`** (PSI 0.00), **`lag_1`** (PSI 0.10), **`lag_2`** (PSI 0.10), **`lag_4`** (PSI 0.07), **`post_promo_2`** (PSI 0.11), **`promo_density_13`** (PSI 0.04), **`promo_x_holiday`** (PSI 0.00), **`promo_x_windex`** (PSI 0.08), **`zero_streak`** (PSI 0.09).
*   Features like **`Weekly_Index`** (PSI 0.19), **`cust_hist_mean`** (PSI 0.18), **`holiday_x_windex`** (PSI 0.23), **`post_promo_2`** (PSI 0.11), **`roll_mean_13`** (PSI 0.18), **`roll_mean_4`** (PSI 0.14), **`roll_std_4`** (PSI 0.18), **`weeks_since_promo`** (PSI 0.11), and **`UnitSales`** (PSI 0.15) are in the 'Monitor' category, showing some change but not yet critical.

**Data Quality Trend**

The comparison between the benchmark (DEV) and the out-of-time data (OOT_01) reveals a significant shift in the underlying data distribution across many features. While the overall data quality (e.g., no new nulls) appears consistent, the *patterns* and *values* that the model learned from the DEV dataset are no longer representative of the OOT_01 dataset. This is a clear indication of concept drift.

**Recommended Actions**

1.  **Retrain the FE_LSDV model immediately:** Given the critical drift and performance degradation, the model needs to be retrained on more recent data that reflects the patterns seen in OOT_01.
2.  **Investigate root causes for `quarter` and `global_t` drift:** Understand *why* these fundamental features have changed so drastically. This could be due to market shifts, new business strategies, or external factors.
3.  **Monitor features in the 'Warning' and 'Monitor' categories closely:** Pay particular attention to `is_year_end`, `is_year_start`, `weeks_to_next_promo`, `wow_growth`, and promotional features. Any further drift here could exacerbate performance issues.
4.  **Review the impact of promotional and seasonal features:** The drift in `August`, `September`, `week_sin`, `week_cos`, and promotional features suggests a need to re-evaluate how these are incorporated or weighted in the model.
5.  **Consider feature engineering or selection:** If certain features continue to drift significantly and unpredictably, explore alternative features or combinations that might be more stable predictors.

---

## Data Scientist

The drift analysis for FE_LSDV comparing DEV (benchmark) to OOT_01 shows **20 anomalies across 20 features**.

**CRITICAL ISSUE DETECTED**

The `FE_LSDV` model exhibits significant drift when comparing the DEV benchmark to the OOT_01 evaluation dataset, as indicated by a `model_PSI` of **1.264726**. This is further evidenced by **20 features** showing `Significant Drift` status. The most substantial drifts are observed in:
*   `quarter` with a PSI of **8.200832**.
*   `global_t` with a PSI of **8.177284**.
*   `week_sin` with a PSI of **4.704905**.
*   `week_cos` with a PSI of **3.809373**.

The implications of this widespread feature drift are severe. The model's underlying assumptions about the distribution of these features may no longer hold true in the OOT_01 dataset. This can lead to a degradation in predictive performance, as demonstrated by the drop in `R2` from -0.037993 in DEV to -0.078249 in OOT_01, and a decrease in `Accuracy_pct` from 38.9802% to 15.077%. Specifically, the large PSI values for `quarter` and `global_t` suggest a fundamental shift in the temporal or cyclical patterns the model is trying to capture. Similarly, significant drift in `week_sin` and `week_cos` indicates a change in the weekly seasonality or phase. The drift in categorical-like features such as `August`, `September`, `NewYear`, `Spring`, `VetreansDay`, and `Thanksgiving` (all with PSI > 0.75) also points to shifts in the prevalence or impact of specific temporal events. The `momentum_4_13` feature also shows a PSI of **1.317402**, suggesting a change in the underlying trend or momentum calculations.

**WARNING**

No features were identified with a 'Monitor' status in this evaluation.

**Stable Features**

A majority of features (18 out of 38 analyzed) remain within acceptable drift thresholds, including:
*   `any_holiday` (PSI: 0.016203)
*   `is_promo` (PSI: 0.000526)
*   `lag_1` (PSI: 0.095944)
*   `lag_2` (PSI: 0.095451)
*   `lag_4` (PSI: 0.068419)
*   `post_promo_1` (PSI: 0.040371)
*   `promo_density_13` (PSI: 0.037123)
*   `promo_x_holiday` (PSI: 0.000512)
*   `promo_x_windex` (PSI: 0.08491)
*   `zero_streak` (PSI: 0.090676)

These features are not contributing to the observed model drift.

**Data Quality Trend**

The overall data quality trend from DEV to OOT_01 shows a significant divergence in feature distributions. While no null values were reported in either dataset, the substantial shifts in the PSI scores for many features, particularly `quarter`, `global_t`, `week_sin`, and `week_cos`, indicate a change in the underlying data generation process or the sampling of the OOT_01 dataset. The descriptive statistics reveal some key differences:
*   **Temporal Features:** `quarter` shows a dramatic shift, as expected given its high PSI. `August` has a mean of 0.0312 in DEV and -0.0128 in OOT_01, with a standard deviation of 0.265 vs 0.0589 respectively, indicating a concentration of values around zero in OOT_01. `September` has a mean of 0.0959 in DEV and 0.0022 in OOT_01, with a standard deviation of 0.2683 vs 0.0099 respectively, again showing a significant reduction in variance and shift in central tendency. `NewYear` has a mean of 0.2203 and std of 0.3974 in DEV, but a mean of 0.3732 and std of 0.4955 in OOT_01, indicating a shift towards higher values and increased variance.
*   **Cyclical Features:** `week_sin` and `week_cos` also show large PSI values. Their means in DEV are -0.1549 and -0.0361 respectively, with standard deviations of 0.6925 and 0.7037. The OOT_01 descriptive statistics for these features are not provided, but the high PSI values confirm a significant distributional shift.
*   **Momentum and Lagged Features:** `momentum_4_13` has a mean of 0.8784 and std of 0.6354 in DEV, and a mean of 1.1023 and std of 0.9657 in OOT_01. This indicates an increase in momentum and a wider spread of values.
*   **Sales and Performance Indicators:** `UnitSales` shows a mean of 120.5137 with a large std of 477.4696 in DEV, and a mean of 120.805 with a std of 359.6155 in OOT_01. While the mean is similar, the standard deviation is lower in OOT_01, suggesting a tighter distribution of sales. However, `wow_growth` has a massive mean of 37819672.1653 in DEV and 36535894.9256 in OOT_01, with extremely high standard deviations in both. The PSI for `wow_growth` is 0.307847, indicating a significant shift.
*   **Promo-related Features:** `promo_intensity` shows a mean of 11.2613 and std of 22.1702 in DEV, and a mean of 3.689 and std of 15.1498 in OOT_01. This is a substantial drop in the average promo intensity and its variance. `pre_promo_1` and `pre_promo_2` also exhibit significant drift, with PSI values of 0.316402 and 0.564083 respectively.

**Recommended Actions**

1.  **Investigate Root Cause:** Conduct a deep dive into the data generation process for the OOT_01 dataset to understand why `quarter`, `global_t`, `week_sin`, `week_cos`, and other heavily drifted features have changed distribution.
2.  **Feature Engineering Review:** Re-evaluate the feature engineering for `quarter`, `global_t`, and the weekly sinusoidal features. Consider if they are capturing underlying patterns effectively or if alternative representations are needed.
3.  **Model Retraining:** Retrain the `FE_LSDV` model on the OOT_01 dataset or a more recent, representative dataset to adapt to the observed data shifts.
4.  **Feature Re-evaluation/Removal:** For features with extreme drift and potentially unstable relationships (e.g., `quarter`, `global_t`), consider if they should be removed or if their contribution needs to be re-weighted.
5.  **Monitor Performance Metrics:** Closely monitor key performance indicators such as MAE, RMSE, MAPE, and R2 on new incoming data to assess the impact of the drift and any retraining efforts. The observed degradation in performance metrics (MAE increased, RMSE decreased but likely due to scale changes, MAPE increased, Accuracy_pct decreased, R2 became more negative) necessitates immediate action.

---

## Risk Manager

The drift analysis for FE_LSDV comparing DEV (benchmark) to OOT_01 shows 20 anomalies across 37 features, with a model-wide PSI of 1.264726 indicating significant drift.

**CRITICAL ISSUE DETECTED**

The model FE_LSDV exhibits significant drift when comparing the benchmark dataset DEV to the out-of-time dataset OOT_01. The overall model PSI is 1.264726, which is considerably high and indicates a substantial change in the data distribution. This is further evidenced by 20 features showing "Significant Drift".

The most concerning features with significant drift are:
*   **quarter**: PSI of 8.200832. This suggests a major shift in the distribution of quarters, which are fundamental temporal components of the model.
*   **global_t**: PSI of 8.177284. This indicates a significant change in the underlying time trend captured by the model.
*   **week_sin** and **week_cos**: PSI values of 4.704905 and 3.809373 respectively. These are sinusoidal representations of the week of the year, and their significant drift implies a change in the cyclical patterns within the data.
*   **September** and **August**: PSI values of 1.551873 and 1.50475 respectively. These monthly indicators have also drifted significantly.
*   **momentum_4_13**: PSI of 1.317402. This feature, likely representing a momentum indicator over a specific period, shows a substantial shift.

The drift in these key temporal and cyclical features, alongside others like `NewYear` (0.907789), `Spring` (0.781587), and `VetreansDay` (0.75891), suggests that the model's understanding of temporal dynamics and seasonalities has degraded.

**Model Sensitivity to High-Weighted Features:**
The features `quarter`, `global_t`, `week_sin`, and `week_cos` are likely high-weighted features given their temporal and cyclical nature, and their significant drift is a primary driver of the overall model drift. This indicates that the model is highly sensitive to changes in these fundamental temporal components.

**Over-Reliance Risks:**
The model appears to be over-reliant on the temporal and cyclical patterns captured by features like `quarter`, `global_t`, and `week_sin`/`week_cos`. The significant drift in these features suggests that the model's predictions may no longer accurately reflect current market conditions or temporal trends, leading to potentially erroneous forecasts.

**Temporal Stability:**
The temporal stability of the model is severely compromised. The significant drift in core temporal features (`quarter`, `global_t`, `week_sin`, `week_cos`) and seasonal indicators points to a breakdown in the model's ability to generalize across different time periods.

**Regulatory Implications of Data Drift:**
The significant data drift, particularly in temporal features, has regulatory implications. If FE_LSDV is used for critical decision-making, such as regulatory reporting or capital allocation, the drift could lead to non-compliance with regulations that require accurate and reliable forecasting. Regulators may question the validity of models that exhibit such substantial shifts in data distributions without appropriate remediation.

**Reliability for Long-Term Forecasting:**
The current state of FE_LSDV, with significant drift in core temporal features, renders it unreliable for long-term forecasting. The model's learned relationships from the benchmark data (DEV) are no longer representative of the current data distribution (OOT_01). Any long-term forecasts produced by this model would likely be inaccurate and misleading.

**WARNING**

Several features are flagged with "Monitor" status, indicating a moderate level of drift that warrants attention:
*   **August** (PSI: 1.50475) and **September** (PSI: 1.551873) are listed under "top_drifted" with "Significant Drift" status and PSI values above 1.0. However, in the `all_feature_psi` list, these are also marked as "Significant Drift". This is a discrepancy in reporting. Assuming the `top_drifted` list is accurate, these are critical.
*   **NewYear** (PSI: 0.907789), **Spring** (PSI: 0.781587), and **VetreansDay** (PSI: 0.75891) are also listed as "Significant Drift" in the `top_drifted` list, indicating their PSI values are above the threshold for "Monitor" but below the highest tier of concern.
*   **momentum_4_13** (PSI: 1.317402) also shows significant drift.
*   **is_year_end** (PSI: 0.632577) and **is_year_start** (PSI: 0.664295) exhibit moderate drift.
*   **pre_promo_1** (PSI: 0.316402) and **pre_promo_2** (PSI: 0.564083) show moderate drift.
*   **promo_intensity** (PSI: 0.428732) has drifted.
*   **weeks_to_next_promo** (PSI: 0.381144) indicates a shift in promotional timing patterns.
*   **wow_growth** (PSI: 0.307847) and **zero_frac_13** (PSI: 0.275068) show moderate drift.

While these features are not as critically drifted as `quarter` or `global_t`, their moderate drift suggests potential changes in the underlying data generating process that could impact model performance over time.

**Stable Features**

A number of features remain stable, indicating their distributions have not significantly changed between the benchmark and evaluation datasets. These include:
*   `any_holiday` (PSI: 0.016203)
*   `is_promo` (PSI: 0.000526)
*   `lag_1` (PSI: 0.095944), `lag_2` (PSI: 0.095451), `lag_4` (PSI: 0.068419)
*   `post_promo_1` (PSI: 0.040371)
*   `promo_density_13` (PSI: 0.037123)
*   `promo_x_holiday` (PSI: 0.000512)
*   `promo_x_windex` (PSI: 0.08491)
*   `zero_streak` (PSI: 0.090676)

Features with "Monitor" status in the `all_feature_psi` list, but not appearing in the `top_drifted` list (suggesting PSI values below the top threshold but still above "Stable"), include:
*   `Weekly_Index` (PSI: 0.186635)
*   `cust_hist_mean` (PSI: 0.183461)
*   `holiday_x_windex` (PSI: 0.227577)
*   `post_promo_2` (PSI: 0.108009)
*   `roll_mean_13` (PSI: 0.181509)
*   `roll_mean_4` (PSI: 0.142628)
*   `roll_std_4` (PSI: 0.179503)
*   `weeks_since_promo` (PSI: 0.114802)
*   `UnitSales` (PSI: 0.151283)

The stability of these features is positive, but the overall model drift suggests that the cumulative effect of the drifted features is significantly impacting performance.

**Data Quality Trend**

The data quality appears consistent from a null value perspective, with no features reporting a maximum null percentage of 0.0 in either dataset. However, the significant drift in feature distributions, particularly temporal and cyclical ones, indicates a substantial shift in the underlying data generation process or external factors influencing the data. This is a concerning trend as it suggests the model's assumptions about the data are no longer valid. The performance metrics also show a decline from DEV to OOT_01 (e.g., R2 from -0.037993 to -0.078249, Accuracy_pct from 38.9802% to 15.077%), corroborating the impact of the observed drift.

**Governance Concerns**

1.  **Lack of Proactive Monitoring and Retraining Trigger:** The current state suggests that either monitoring thresholds were not adequately set, or the model was not retrained or re-evaluated promptly when drift indicators began to appear. The "Significant Drift" status across multiple features, including core temporal ones, indicates a potential failure in the model lifecycle management process.
2.  **Inconsistent Reporting of Drifted Features:** The discrepancy between the `top_drifted` list and the `all_feature_psi` list for features like `August` and `September` needs to be investigated and clarified. This inconsistency can lead to miscommunication and delayed decision-making.
3.  **Model Revalidation and Business Impact Assessment:** A critical governance concern is the lack of immediate action upon detecting significant drift. The model's performance degradation, as seen in the OOT_01 metrics, needs a thorough business impact assessment.
4.  **Documentation of Drift Remediation:** The process for addressing such significant drift, including retraining strategies, feature re-engineering, or model replacement, needs to be clearly documented and approved.
5.  **Responsibility for Drift Management:** It is unclear from the provided data who is ultimately responsible for monitoring drift, initiating remediation, and approving model changes. Clear ownership is crucial for effective AI governance.

**Recommended Actions**

1.  **Immediate Model Retraining/Revalidation:** Initiate a process to retrain FE_LSDV on the OOT_01 data or a more recent dataset that reflects current conditions. If retraining does not sufficiently mitigate the drift and performance degradation, consider model replacement.
2.  **Investigate Root Cause of Drift:** Conduct a thorough investigation into the root causes of the significant drift in features like `quarter`, `global_t`, `week_sin`, `week_cos`, and other temporal/cyclical features. This may involve analyzing external factors, changes in business operations, or data pipeline issues.
3.  **Review and Adjust Monitoring Thresholds:** Re-evaluate the drift monitoring thresholds for individual features and the overall model PSI. Ensure thresholds are aligned with the risk appetite and the criticality of the model's use cases.
4.  **Clarify Reporting Discrepancies:** Investigate and resolve the inconsistencies in reporting drift statuses between the "top_drifted" and "all_feature_psi" lists to ensure accurate communication.
5.  **Update Model Documentation:** Document the observed drift, the root cause analysis, the remediation actions taken, and the updated performance metrics. This is crucial for auditability and future reference.
6.  **Enhance Model Governance Framework:** Strengthen the AI governance framework to include more robust processes for continuous monitoring, automated alerts for significant drift, clearly defined retraining triggers, and a formal model revalidation process for models exhibiting substantial drift.
7.  **Assess Impact on Downstream Systems:** Evaluate the impact of the model's degraded performance on any downstream systems or business processes that rely on its predictions.

---
