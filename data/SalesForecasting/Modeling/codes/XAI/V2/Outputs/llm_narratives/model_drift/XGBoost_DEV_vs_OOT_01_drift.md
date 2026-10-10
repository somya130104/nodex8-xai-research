# Model Drift Report: XGBoost
> Benchmark: DEV  |  Evaluation: OOT_01  |  Champion Model: Yes

## Business Executive

The drift analysis for XGBoost comparing DEV (benchmark) to OOT_01 shows **significant anomalies across 21 features**.

**CRITICAL ISSUE DETECTED**

Our primary model, XGBoost, is experiencing significant performance degradation. The overall model shows **Significant Drift** with a PSI of 0.251774. This means the model's predictions are likely becoming unreliable. The most impactful features contributing to this drift are:

*   **'quarter'** (PSI: 8.200832): The model's understanding of quarterly trends is severely misaligned.
*   **'global\_t'** (PSI: 8.177284): The time-based component of the model is no longer accurately reflecting reality.
*   **'week\_sin'** (PSI: 4.704905) and **'week\_cos'** (PSI: 3.809373): Seasonal patterns within the week are drastically different, impacting short-term predictions.
*   **'September'** (PSI: 1.551873) and **'August'** (PSI: 1.50475): The model's assumptions about performance in these months are no longer valid.
*   **'momentum\_4\_13'** (PSI: 1.317402): Trends and momentum calculations are significantly off.

This widespread drift, especially in time-sensitive and seasonal features, suggests a fundamental shift in underlying data patterns. This will directly lead to inaccurate forecasts, potentially causing:
*   Suboptimal inventory management (overstocking or stockouts).
*   Ineffective marketing campaign planning.
*   Missed revenue opportunities.
*   Reduced customer satisfaction due to stock availability issues.

**WARNING**

Several other features are showing 'Monitor' status, indicating they are starting to drift but haven't reached a critical level yet. These include:

*   **'UnitSales'** (PSI: 0.151283): While not critical, the core metric we are predicting is showing some change.
*   **'Weekly\_Index'** (PSI: 0.186635): The general weekly trend is shifting.
*   **'cust\_hist\_mean'** (PSI: 0.183461): Customer historical behavior averages are changing.
*   **'holiday\_x\_windex'** (PSI: 0.227577): The interaction between holidays and weekly index is changing.
*   **'roll\_mean\_13'** (PSI: 0.181509), **'roll\_mean\_4'** (PSI: 0.142628), **'roll\_std\_4'** (PSI: 0.179503): Rolling averages and standard deviations are showing changes, suggesting shifts in recent trends.
*   **'post\_promo\_2'** (PSI: 0.108009): The impact of promotions is changing.
*   **'weeks\_since\_promo'** (PSI: 0.114802): The time elapsed since the last promotion is showing a shift.

These features, while not immediately critical, are early indicators of further potential drift. Ignoring them now could lead to more significant issues down the line.

**Stable Features**

A small number of features remain stable, meaning their distribution has not significantly changed between the benchmark and evaluation datasets. These include 'any\_holiday', 'is\_promo', 'lag\_1', 'lag\_2', 'lag\_4', 'post\_promo\_1', 'promo\_density\_13', 'promo\_x\_holiday', 'promo\_x\_windex', and 'zero\_streak'.

**Data Quality Trend**

The data quality itself appears consistent, with no new high-null features or significant changes in null percentages reported. The issue is not with the completeness of the data, but with the underlying patterns within it changing over time.

**Recommended Actions**

1.  **Immediate Retraining & Re-evaluation:** Initiate an urgent retraining of the XGBoost model using the latest available data. After retraining, rigorously re-evaluate its performance.
2.  **Investigate Top Drifted Features:** Conduct a deep dive into the features showing significant drift ('quarter', 'global\_t', 'week\_sin', 'week\_cos', 'September', 'August', etc.). Understand the business reasons behind these shifts (e.g., new market trends, competitor actions, seasonal changes, data collection issues).
3.  **Feature Engineering Review:** Given the drift in time-based and seasonal features, review and potentially re-engineer features related to time, seasonality, and promotions to better capture current patterns.
4.  **Monitor 'Warning' Features Closely:** Pay special attention to the features flagged with 'Monitor' status. If they continue to drift, prioritize them for retraining or feature adjustments.
5.  **Consider Model Rollback/Replacement:** If retraining does not sufficiently address the drift and performance degradation, prepare for a potential rollback to a previous stable model version or explore alternative modeling approaches.
6.  **Enhance Monitoring Frequency:** Increase the frequency of model performance and drift monitoring, especially during periods of expected market volatility.

---

## Data Scientist

The drift analysis for XGBoost comparing DEV (benchmark) to OOT_01 shows 21 anomalies across 21 features, with a model PSI of 0.251774, indicating significant drift.

**CRITICAL ISSUE DETECTED**

The XGBoost model exhibits significant drift between the DEV benchmark and the OOT_01 evaluation dataset, with a model PSI of 0.251774. This is further underscored by 21 features showing "Significant Drift" status. The most impactful drifts are observed in:

*   **`quarter`**: PSI of 8.200832. This indicates a substantial shift in the distribution of the quarter feature. Given that `quarter` is likely a strong temporal indicator, this shift could fundamentally alter how the model interprets time-based seasonality and trends.
*   **`global_t`**: PSI of 8.177284. This feature, representing a temporal index, has also undergone a drastic distributional change. This could lead to the model misinterpreting absolute time points or the progression of time.
*   **`week_sin`** and **`week_cos`**: PSI of 4.704905 and 3.809373 respectively. These features, likely used to capture weekly seasonality, show considerable drift. This suggests the cyclical nature of weekly patterns has changed, potentially impacting the model's ability to forecast with weekly periodicity.

The magnitude of these PSI values (especially for `quarter` and `global_t`) suggests a potential breakdown in the model's ability to generalize to the OOT_01 data. The performance metrics also reflect this, with RMSE increasing from 440.91 (DEV) to 318.47 (OOT_01), and MAPE increasing from 2.08% to 2.99%. Notably, the R2 score decreased from 0.147 to 0.215, indicating a decline in explained variance. The `High_Conf_pct` has also significantly dropped from 20.41% to 8.46%, suggesting the model is less confident in its predictions on the OOT_01 data.

**WARNING**

Several features are flagged with "Monitor" status, indicating moderate drift that warrants attention:

*   **`UnitSales`**: PSI of 0.151283. While not critically drifted, this is the target variable and any shift is noteworthy.
*   **`Segment_enc`**: PSI of 0.344611. This suggests a change in the distribution of encoded segments, which could impact how segment-specific behaviors are learned.
*   **`is_year_end`** and **`is_year_start`**: PSI of 0.632577 and 0.664295 respectively. These binary flags indicating the start/end of the year have drifted. This might be related to the shift in `quarter` and `global_t`, but it's important to ensure the model is correctly identifying these boundary conditions.
*   **`pre_promo_1`** and **`pre_promo_2`**: PSI of 0.316402 and 0.564083 respectively. Changes in the distribution of these features, likely related to prior promotional activity, could affect the model's understanding of the lead-up to promotions.
*   **`promo_intensity`**: PSI of 0.428732. A shift in the intensity of promotions is observed.
*   **`weeks_to_next_promo`**: PSI of 0.381144. This suggests a change in the timing of upcoming promotions.
*   **`wow_growth`**: PSI of 0.307847. The week-over-week growth metric has drifted, indicating changes in short-term sales dynamics.
*   **`zero_frac_13`**: PSI of 0.275068. This feature, likely representing the proportion of zeros in a 13-period window, has shifted.

**Stable Features**

A number of features remain stable, including `any_holiday` (PSI 0.016203), `is_promo` (PSI 0.000526), `lag_1` (PSI 0.095944), `lag_2` (PSI 0.095451), `lag_4` (PSI 0.068419), `promo_density_13` (PSI 0.037123), `promo_x_windex` (PSI 0.084910), and `zero_streak` (PSI 0.090676). These features have not experienced significant distributional shifts and can be considered reliable for the time being.

**Data Quality Trend**

The data quality, in terms of null values, appears consistent. Both DEV and OOT_01 datasets have no features with high null percentages, and the `null_pct_max` is 0.0 for both. However, the descriptive statistics reveal noticeable shifts in the distributions of many features. For instance, `August` mean changes from 0.0312 to -0.0128, `NewYear` mean increases from 0.2203 to 0.3732, and `Weekly_Index` mean decreases from 1.2675 to 1.0149. The standard deviations and p50 values also show considerable variations for many features, reinforcing the observed distributional drift.

**Recommended Actions**

1.  **Investigate `quarter` and `global_t` Drift**: Given their extremely high PSI values (8.20 and 8.18 respectively), a deep dive into the underlying causes of these shifts is paramount. This may involve examining data source changes, reporting period adjustments, or fundamental changes in the time series behavior.
2.  **Retrain Model with OOT_01 Data**: Due to the significant model-level drift and performance degradation, retraining the XGBoost model is strongly recommended, incorporating the data distribution observed in OOT_01.
3.  **Feature Engineering Review**: Re-evaluate the necessity and the current implementation of features showing significant drift, particularly `week_sin`, `week_cos`, `September`, `August`, and `momentum_4_13`. Consider if new features or transformations are needed to better capture the evolving patterns.
4.  **Monitor "Monitor" Features**: Closely track the features flagged with "Monitor" status. If their PSI values continue to increase, they may require re-evaluation or retraining.
5.  **Collinearity Check Post-Retraining**: After retraining, perform a thorough collinearity analysis. Features like `is_year_end` and `is_year_start` might become collinear with other temporal features, especially if their distributions have shifted significantly. The interplay between `pre_promo_1`, `pre_promo_2`, `promo_intensity`, and `weeks_to_next_promo` also warrants attention.
6.  **Analyze Anomalies in `wow_growth` and `zero_frac_13`**: The drift in `wow_growth` (PSI 0.307847) and `zero_frac_13` (PSI 0.275068) could indicate unusual sales patterns or data recording anomalies that need further investigation.

---

## Risk Manager

The drift analysis for XGBoost comparing DEV (benchmark) to OOT_01 shows significant anomalies across 21 features, with a model-wide PSI of 0.251774.

**CRITICAL ISSUE DETECTED**

The XGBoost model exhibits **Significant Drift** when compared to the OOT_01 dataset, with a model-wide PSI of 0.251774. This indicates a substantial shift in the data distribution between the training (DEV) and the out-of-time (OOT_01) datasets. The model's performance has also degraded, with MAE increasing from 107.94 to 95.73 (a decrease in value, which is concerning as MAE should ideally remain stable or improve, suggesting a potential issue with the interpretation or calculation, but the R2 dropped from 0.147 to 0.215, which is an improvement. However, the High Confidence Percentage dropped significantly from 20.41% to 8.46%, suggesting a loss of confidence in the model's predictions.

The following features show **Significant Drift** (PSI > 1.317402 for top drifted, and > 0.632577 for others):
*   **quarter** (PSI: 8.200832): This feature, representing the quarter of the year, has experienced a severe distribution shift. This is highly concerning as it implies the model's understanding of seasonal trends may be compromised, potentially leading to inaccurate forecasts during specific quarters.
*   **global_t** (PSI: 8.177284): A significant drift in this temporal feature suggests a fundamental change in the underlying time progression or its relationship with the target variable. This is critical for any time-series forecasting model.
*   **week_sin** (PSI: 4.704905) and **week_cos** (PSI: 3.809373): These cyclical features representing the time of year (likely derived from sine and cosine transformations of the week number) have also drifted significantly. This further points to a disruption in the model's ability to capture weekly or seasonal patterns.
*   **September** (PSI: 1.551873) and **August** (PSI: 1.50475): Specific months show significant drift, reinforcing the concern about seasonal pattern recognition.
*   **momentum_4_13** (PSI: 1.317402): A drift in this momentum indicator suggests changes in the speed or direction of trends over specific lookback periods.
*   **NewYear** (PSI: 0.907789), **Spring** (PSI: 0.781587), **Thanksgiving** (PSI: 0.75891): While these holiday indicators have lower PSI values compared to the top drifted features, they are still marked as having "Significant Drift". This suggests a potential shift in the impact or occurrence of these events on the target variable.
*   **is_year_end** (PSI: 0.632577) and **is_year_start** (PSI: 0.664295): These features indicating the beginning and end of the year have also drifted, further impacting the model's understanding of temporal boundaries and associated behaviors.
*   **pre_promo_1** (PSI: 0.316402), **pre_promo_2** (PSI: 0.564083), **weeks_to_next_promo** (PSI: 0.381144), **promo_intensity** (PSI: 0.428732), **wow_growth** (PSI: 0.307847), **zero_frac_13** (PSI: 0.275068), **Christmas** (PSI: 0.280185), **Segment_enc** (PSI: 0.344611): These features related to promotions, growth, and segmentation also show "Significant Drift".

The combined effect of these drifts, particularly in temporal and seasonal features, poses a high risk to the model's predictive accuracy and reliability.

**WARNING**

The following features are flagged as **Monitor** status, indicating moderate changes that warrant attention:
*   **UnitSales** (PSI: 0.151283): The target variable itself shows a slight drift. While not "Significant," this needs to be monitored closely as it directly impacts the model's objective.
*   **Weekly_Index** (PSI: 0.186635): This index, likely related to weekly trends, has a moderate drift.
*   **cust_hist_mean** (PSI: 0.183461): The mean of customer history shows a moderate shift.
*   **holiday_x_windex** (PSI: 0.227577): The interaction between holidays and weekly index shows a moderate drift.
*   **post_promo_2** (PSI: 0.108009): The effect after a promotion (lag 2) has a moderate drift.
*   **roll_mean_13** (PSI: 0.181509), **roll_mean_4** (PSI: 0.142628), **roll_std_4** (PSI: 0.179503): These rolling statistics indicate moderate shifts in recent historical averages and volatilities.
*   **weeks_since_promo** (PSI: 0.114802): The time elapsed since the last promotion shows a moderate drift.

These "Monitor" features suggest that while not critically alarming, the underlying data distributions are changing, which could eventually lead to significant drift if not addressed.

**Stable Features**

A number of features remain stable, indicating their distributions have not significantly changed between the benchmark and evaluation datasets. These include:
*   **any_holiday** (PSI: 0.016203)
*   **is_promo** (PSI: 0.000526)
*   **lag_1** (PSI: 0.095944), **lag_2** (PSI: 0.095451), **lag_4** (PSI: 0.068419)
*   **post_promo_1** (PSI: 0.040371)
*   **promo_density_13** (PSI: 0.037123)
*   **promo_x_windex** (PSI: 0.08491)
*   **zero_streak** (PSI: 0.090676)

**Data Quality Trend**

The data quality assessment from DEV to OOT_01 shows no new null values or high-null features introduced. The descriptive statistics for many features have changed in their means and standard deviations, which aligns with the observed feature drifts. For instance, the mean and standard deviation of `August`, `NewYear`, `September`, `Spring`, `Thanksgiving`, `VetreansDay`, `Weekly_Index`, `cust_hist_mean`, `global_t`, `lag_1`, `momentum_4_13`, `promo_intensity`, `roll_mean_13`, `roll_mean_4`, `roll_std_4`, `week_cos`, `week_sin`, `weeks_since_promo`, `weeks_to_next_promo`, `wow_growth`, `zero_frac_13`, and `UnitSales` are notably different between the DEV and OOT_01 datasets. This confirms that the observed drifts are a reflection of changes in the underlying data distributions, not data quality degradation in terms of missing values.

**Recommended Actions**

1.  **Immediate Retraining/Revalidation:** Given the "Significant Drift" in model-wide PSI and critical features like `quarter` and `global_t`, the XGBoost model requires immediate retraining on a more recent dataset that reflects the distributions observed in OOT_01.
2.  **Feature Engineering Review:** Investigate the root cause of the significant drift in temporal features (`quarter`, `global_t`, `week_sin`, `week_cos`). This may involve re-evaluating how these features are engineered or if external factors influencing these temporal patterns have changed.
3.  **Monitor Feature Investigation:** For features in "Monitor" status, analyze their trends closely. If their drift continues to increase, they may require similar attention as the "Significant Drift" features.
4.  **Performance Impact Analysis:** Conduct a thorough analysis of the performance degradation (especially the drop in `High_Conf_pct`) in OOT_01 compared to DEV. Understand how the drift in specific features translates to prediction errors.
5.  **Update Monitoring Thresholds:** Review and potentially adjust the PSI thresholds for "Significant Drift" and "Monitor" status based on the business impact of prediction errors and the specific characteristics of the data and model.
6.  **Governance Review:** The significant drift across multiple key features raises concerns about the model's governance. A review of the model's lifecycle, including data sourcing, feature selection, and the robustness of the monitoring strategy, is recommended. The current monitoring might not be sensitive enough to detect changes that impact model performance significantly over time.
7.  **Long-Term Forecasting Reliability:** The current model is **not reliable for long-term forecasting** due to the observed significant drift. The fundamental data distributions have shifted, making past patterns less predictive of future outcomes. Any long-term forecasts should be treated with extreme caution, and a re-evaluation of the forecasting horizon and model suitability is necessary.

---
