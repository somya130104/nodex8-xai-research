# Model Drift Report: FE_LSDV
> Benchmark: DEV  |  Evaluation: OOS  |  Champion Model: No

## Business Executive

The drift analysis for FE_LSDV comparing DEV (benchmark) to OOS shows 0 anomalies across 42 features.

**CRITICAL ISSUE DETECTED**
N/A

**WARNING**
N/A

**Stable Features**
All 42 features analyzed show stable performance between the benchmark (DEV) and the evaluation dataset (OOS). This indicates that the model's understanding of these features has not significantly changed. Notable features with very low drift include `is_promo` (0.000009 PSI), `post_promo_2` (0.000002 PSI), and `pre_promo_1` (0.00001 PSI). Features with the highest, yet still stable, drift include `weeks_since_promo` (0.001715 PSI), `roll_mean_13` (0.001181 PSI), and `cust_hist_mean` (0.00117 PSI).

**Data Quality Trend**
The data quality trend from the benchmark (DEV) to the evaluation dataset (OOS) is stable. Both datasets have no missing values and a similar distribution of continuous and categorical features. Descriptive statistics for most features remain consistent, with minor variations in mean and standard deviation that do not indicate a significant shift in data patterns.

**Recommended Actions**
1.  **Continue Monitoring:** Maintain current monitoring of FE_LSDV, as all features are stable.
2.  **No Retraining Necessary:** No immediate retraining of the FE_LSDV model is required based on this drift analysis.
3.  **Review High Drift Features:** While stable, pay attention to features with the highest PSI values like `weeks_since_promo` (0.001715 PSI), `roll_mean_13` (0.001181 PSI), and `cust_hist_mean` (0.00117 PSI) in future analyses to ensure they remain predictable.

---

## Data Scientist

The drift analysis for FE_LSDV comparing DEV (benchmark) to OOS shows 0 anomalies across 0 features.

**CRITICAL ISSUE DETECTED**
N/A

**WARNING**
N/A

**Stable Features**
The FE_LSDV model shows a `model_PSI` of 0.000472, indicating overall stability between the DEV benchmark and the OOS evaluation dataset. All individual features also exhibit stable distributions, with no feature exceeding typical drift thresholds.

Notable stable features include:
*   `weeks_since_promo` with a PSI of 0.001715.
*   `roll_mean_13` with a PSI of 0.001181.
*   `cust_hist_mean` with a PSI of 0.00117.
*   `zero_frac_13` with a PSI of 0.001083.
*   `roll_mean_4` with a PSI of 0.000909.
*   `roll_std_4` with a PSI of 0.000847.
*   `momentum_4_13` with a PSI of 0.000825.
*   `Weekly_Index` with a PSI of 0.000942.
*   `holiday_x_windex` with a PSI of 0.00074.
*   `wow_growth` with a PSI of 0.00071.

All other features have PSI values well below 0.0005, reinforcing the model's stability.

**Data Quality Trend**
The data quality trend from DEV to OOS is stable. Both datasets have 0% null values across all features. The descriptive statistics for most features show very minor variations in mean, standard deviation, and p50 values between DEV and OOS. For instance, `cust_hist_mean` has a mean of 103.6092 (std: 246.8217) in DEV and 114.549 (std: 454.4022) in OOS. Similarly, `lag_1` shows a mean of 107.3828 (std: 455.2131) in DEV and 111.3578 (std: 550.4178) in OOS. While some features like `cust_hist_mean`, `lag_1`, `lag_2`, `lag_4`, `roll_mean_13`, `roll_mean_4`, `roll_std_4`, and `UnitSales` show an increase in standard deviation in the OOS set, these changes are not substantial enough to cause significant distributional shifts given the overall low PSI values.

Performance metrics also show minor fluctuations:
*   MAE increased from 98.6684 in DEV to 103.5255 in OOS.
*   RMSE increased from 486.4480 in DEV to 572.1978 in OOS.
*   MAPE remained relatively stable, increasing slightly from 0.610198 to 0.615996.
*   R2 slightly decreased from -0.037993 in DEV to -0.030662 in OOS.

These performance shifts are minor and align with the observed stable feature distributions.

**Recommended Actions**
1.  **Continue Monitoring**: Given the overall stability, maintain the current monitoring frequency for FE_LSDV and its features.
2.  **Review Feature Importance**: Although drift is stable, it's always beneficial to periodically review the feature importance of the model to ensure the most impactful features in DEV are still relevant in OOS.
3.  **Investigate Performance Delta**: While minor, the slight increase in MAE and RMSE, and decrease in R2, warrants a brief investigation into potential drivers. This could involve examining specific time periods or segments within the OOS data where these performance degradations are most pronounced.
4.  **Feature Engineering Review**: Consider if any feature engineering logic, particularly for time-series dependent features like lags and rolling statistics, might be implicitly sensitive to longer-term trends not captured by the current drift monitoring.

---

## Risk Manager

The drift analysis for FE_LSDV comparing DEV (benchmark) to OOS shows 0 anomalies across 35 features.

**CRITICAL ISSUE DETECTED**
N/A

**WARNING**
N/A

**Stable Features**
All 35 features monitored for FE_LSDV exhibit stable distributions between the DEV (benchmark) and OOS (evaluation) datasets. The model-level PSI is 0.000472, which is well within acceptable thresholds, indicating no significant drift in the overall model behavior. Individual feature PSIs are also very low, with the highest observed for `weeks_since_promo` at 0.001715, `zero_frac_13` at 0.001083, `roll_mean_13` at 0.001181, and `cust_hist_mean` at 0.001170. All other features have PSI values significantly below 0.001.

**Data Quality Trend**
The data quality appears consistent between the DEV and OOS datasets. Both datasets have no null values reported for any features. The descriptive statistics for continuous features show minor variations in mean, standard deviation, and median values, which are expected and reflected in the low feature-level PSI scores. This indicates that the underlying data characteristics driving the model's predictions have not substantially changed.

**Model Sensitivity to High-Weighted Features:**
While the provided data does not explicitly list feature weights, the low PSI values across all features suggest that the model is not overly sensitive to shifts in any single feature's distribution. If high-weighted features were experiencing significant drift, the overall model PSI would likely be higher. The stability across all features, including those potentially related to sales dynamics (e.g., `cust_hist_mean`, `roll_mean_13`, `weeks_since_promo`), is a positive sign.

**Over-Reliance Risks:**
Given the model's stability and the lack of significant drift in any feature, there is currently no indication of over-reliance on specific features that might be experiencing undetected shifts. The uniform stability across features suggests a robust model that is not overly dependent on a narrow set of potentially volatile inputs.

**Temporal Stability:**
The evaluation of the model on the OOS dataset against the DEV benchmark demonstrates good temporal stability. The low model-level PSI of 0.000472 signifies that the model's predictive distribution has not shifted significantly over time. The consistent performance metrics between DEV (R2: -0.037993) and OOS (R2: -0.030662), though both are negative and indicate poor predictive power, also suggest that the model's performance characteristics are temporally stable.

**Regulatory Implications of Data Drift:**
As no significant data drift has been detected (model_status is "Stable" and n_drifted is 0), there are currently no immediate regulatory concerns related to data drift for FE_LSDV. This stability simplifies compliance efforts, as the model's behavior remains predictable and aligned with its benchmark. However, continuous monitoring is crucial to detect future drifts that could trigger regulatory scrutiny.

**Reliability for Long-Term Forecasting:**
The current analysis indicates that FE_LSDV is temporally stable in terms of data distribution. However, the model's performance metrics (negative R2 values) suggest it has very poor predictive power, even on the OOS dataset. This inherent lack of predictive accuracy severely limits its reliability for any forecasting, including long-term. While the data distribution is stable, the model's underlying logic or feature relevance might be flawed, making its forecasts untrustworthy regardless of drift.

**Governance Concerns:**
1.  **Performance vs. Drift:** The most significant governance concern is the stark contrast between the model's *data drift stability* and its *poor predictive performance*. A model with negative R2 values is fundamentally unreliable for its intended purpose (forecasting). This raises questions about the model's initial development, validation, and ongoing suitability.
2.  **Lack of Feature Weight Information:** While PSI is useful, understanding feature weights (or feature importance) is critical for assessing sensitivity to high-weighted features. Without this, a deep dive into why certain features have slightly higher PSI (even if stable) is limited.
3.  **Continuous Monitoring Protocol:** While the current report shows stability, a robust governance framework requires clearly defined thresholds for "stable," "monitor," and "drifted" for both model-level and feature-level PSI, along with pre-defined remediation actions for each. The report indicates `n_monitor` and `n_drifted` are 0, suggesting these thresholds might be set very high or the monitoring system is correctly identifying stability.
4.  **Model Purpose and Validation:** The negative R2 scores suggest a fundamental issue with the model's ability to capture the underlying patterns in the data. Governance requires that models in production are actively contributing value and are not simply maintained due to perceived stability. The rationale for keeping a model with such poor performance needs to be clearly documented and justified.

**Recommended Actions:**
1.  **Investigate Poor Predictive Performance:** Prioritize a thorough investigation into why FE_LSDV exhibits such poor predictive performance (negative R2). This should include re-evaluating feature relevance, model architecture, and training data quality.
2.  **Enhance Feature Importance Analysis:** Integrate feature importance or weight information into drift reports to better understand sensitivity to key drivers of the model's predictions.
3.  **Review Model Performance Thresholds:** Re-evaluate the performance metrics and thresholds for model acceptance and ongoing monitoring. A negative R2 is generally unacceptable for a production forecasting model.
4.  **Consider Model Replacement/Retraining:** Given the poor performance, explore options for retraining FE_LSDV with potentially improved features or alternative model architectures, or consider replacing it with a more performant model.
5.  **Maintain Continuous Monitoring:** Continue to monitor both data drift and model performance metrics rigorously. Establish a clear escalation path for performance degradation or significant data drift.

---
