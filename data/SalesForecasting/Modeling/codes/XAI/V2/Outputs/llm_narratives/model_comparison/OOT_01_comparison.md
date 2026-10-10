# Model Comparison Report: OOT_01 Dataset
> Champion Model: XGBoost  |  Baseline: DEV

## Business Executive

Comparing the available models for Sales Forecasting on OOT_01 reveals distinct trade-offs.

**Performance Leader**:
The XGBoost model leads in predictive accuracy on the OOT_01 dataset, showing a R² of 0.215. This is a significant improvement over the FE_LSDV model, which has a R² of -0.078.

**Explainability Champion**:
The FE_LSDV model is the clear winner for interpretability. As a linear model, its decision-making process is more transparent and easier to understand compared to the more complex XGBoost model.

**Champion Model Assessment** (XGBoost):
Our current champion, XGBoost, demonstrates superior performance on the OOT_01 dataset with a R² of 0.215, outperforming FE_LSDV's R² of -0.078. However, when looking at the DEV dataset, which serves as our baseline, XGBoost's R² is 0.147, while FE_LSDV's R² is -0.038. This suggests XGBoost is generally more accurate for forecasting. On the other hand, FE_LSDV shows a higher percentage of accurate predictions on the DEV dataset (38.98% vs 0.0% for XGBoost), but this is not reflected in its R² value. XGBoost has a significantly higher "High Confidence" prediction rate on both datasets (8.46% on OOT_01 and 20.41% on DEV) compared to FE_LSDV (3.72% on OOT_01 and 5.69% on DEV).

**Recommendation**:
For **maximum forecasting accuracy** on the OOT_01 dataset, **XGBoost is the recommended model**. Its R² of 0.215 indicates a better ability to capture the underlying trends. For situations where **understanding *why* a forecast is made is critical**, and a slight dip in accuracy is acceptable, the **FE_LSDV model should be used**. Its linear nature provides greater transparency.

---

## Data Scientist

Comparing the available models for Sales Forecasting on OOT_01 reveals distinct trade-offs...

**Performance Leader**: The XGBoost model demonstrates superior performance on the OOT_01 dataset, achieving an R² of 0.215483, significantly outperforming the FE_LSDV model's R² of -0.078249. This indicates XGBoost captures more variance in the target variable, UnitSales, on this out-of-time data.

**Explainability Champion**: The FE_LSDV model, with its "Linear" family, offers clearer interpretability due to its inherent structure. While XGBoost is a "Tree" based model, its ensemble nature typically leads to more complex interactions, making direct feature-wise explanation more challenging without specific interpretability techniques like SHAP values.

**Champion Model Assessment** (XGBoost): The champion model, XGBoost, exhibits a strong performance on OOT_01 compared to its DEV baseline. On OOT_01, XGBoost achieved an MAE of 95.728286 and an RMSE of 318.471532. This is an improvement over its DEV performance where MAE was 107.940794 and RMSE was 440.912221. While the MAPE on OOT_01 (2.999165) is higher than on DEV (2.080663), the R² (0.215483 vs 0.147242) and Adj_R² (0.205629 vs 0.146236) clearly show better predictive power on the out-of-time data. The Accuracy_pct is 0.0 for both, suggesting it's not a suitable metric for this problem, and High_Conf_pct increased from 20.4072 on DEV to 8.4579 on OOT_01, indicating a more conservative prediction interval on the out-of-time data.

**Recommendation**: For **production deployment and maximizing predictive accuracy on out-of-time data**, the **XGBoost** model is strongly recommended due to its superior R² on OOT_01 (0.215483 vs -0.078249 for FE_LSDV) and improved MAE and RMSE compared to its own DEV baseline. For **exploratory analysis or when absolute interpretability of feature contributions is paramount**, the **FE_LSDV** model could be considered, although its performance on OOT_01 is significantly weaker and it trails its own DEV baseline performance on most metrics.

---

## Risk Manager

Comparing the available models for Sales Forecasting on OOT_01 reveals distinct trade-offs.

**Performance Leader**:
The XGBoost model demonstrates superior performance on the OOT_01 dataset, achieving a higher R² of 0.215483 compared to the FE_LSDV model's R² of -0.078249. While Accuracy_pct is reported as 0.0 for XGBoost on OOT_01, this metric is often less informative for regression tasks and should be interpreted with caution. XGBoost also exhibits a lower MAE (95.728286 vs 110.761298) and RMSE (318.471532 vs 373.360961) on OOT_01, indicating better overall predictive accuracy.

**Explainability Champion**:
The FE_LSDV model, being a linear model, offers clearer interpretability compared to the tree-based XGBoost model. Linear models allow for direct assessment of feature importances and their linear impact on the target variable. XGBoost, while powerful, is a black-box model where feature interactions can be complex and less intuitive to explain.

**Champion Model Assessment** (XGBoost):
The XGBoost model, designated as the organizational champion, shows a mixed performance when compared to the DEV baseline. On the OOT_01 dataset, XGBoost leads in terms of R² (0.215483 vs -0.078249), MAE (95.728286 vs 110.761298), and RMSE (318.471532 vs 373.360961). However, its MAPE on OOT_01 (2.999165) is significantly higher than FE_LSDV's (0.84923), suggesting potential issues with the relative error. The Accuracy_pct is reported as 0.0 for XGBoost on OOT_01, while FE_LSDV shows 15.077%. High_Conf_pct for XGBoost on OOT_01 (8.4579) is higher than FE_LSDV (3.7202).

On the DEV dataset, which serves as the baseline reference, XGBoost trails FE_LSDV in most performance metrics. FE_LSDV has a lower MAE (98.668376 vs 107.940794), RMSE (486.44795 vs 440.912221), and MAPE (0.610198 vs 2.080663). FE_LSDV also exhibits a higher Accuracy_pct (38.9802 vs 0.0) and a less negative R² (-0.037993 vs 0.147242). However, XGBoost on DEV has a substantially higher High_Conf_pct (20.4072 vs 5.6917).

**Recommendation**:
For **Sales Forecasting on OOT_01**, the **XGBoost** model is recommended due to its superior R², MAE, and RMSE, indicating better overall predictive accuracy. However, the significant increase in MAPE and the lower Accuracy_pct compared to FE_LSDV on this dataset warrant further investigation into the drivers of this discrepancy.

For **general forecasting on the DEV dataset**, **FE_LSDV** is the preferred model based on its stronger performance across MAE, RMSE, MAPE, and R².

**Governance Concerns and Further Analysis**:

*   **Model Sensitivity to High-Weighted Features**: The provided data does not include feature importance information. To assess model sensitivity, a feature importance analysis for XGBoost on OOT_01 is crucial. Identifying high-weighted features would allow for an understanding of which inputs are driving predictions and how sensitive the model is to changes in these features.
*   **Over-Reliance Risks**: The significant performance drop of XGBoost on OOT_01 compared to its performance on DEV, particularly the high MAPE, suggests a potential over-reliance on the model without fully understanding its limitations in this specific out-of-time scenario. The drop in Accuracy_pct from 38.98% on DEV for FE_LSDV to 15.077% on OOT_01, and to 0.0% for XGBoost on OOT_01, is concerning.
*   **Temporal Stability and Regulatory Implications of Data Drift**: The stark performance difference between DEV and OOT_01, especially for XGBoost, strongly indicates data drift. The temporal stability of the XGBoost model is questionable. Regulatory scrutiny concerning data drift is increasing, and significant drift could invalidate model assumptions and lead to compliance issues if not adequately managed and reported. The model's performance on OOT_01 suggests it may not be robust to recent shifts in the data distribution.
*   **Reliability for Long-Term Forecasting**: Given the observed performance degradation on OOT_01, the reliability of the current XGBoost model for long-term forecasting is questionable. The model may not generalize well to future data if the underlying patterns continue to evolve as observed in the OOT_01 set. Further validation on more recent out-of-time data is essential.
*   **Explicit Governance Concerns**:
    *   **Lack of Feature Importance Analysis**: Without understanding feature importance, the model's decision-making process remains opaque, hindering effective risk management and bias detection.
    *   **Inadequate Out-of-Time Performance**: The significant performance degradation on OOT_01 compared to DEV raises concerns about the model's robustness and generalization capabilities, potentially leading to misinformed business decisions.
    *   **Data Drift Management**: The observed data drift necessitates a formal process for monitoring, detecting, and mitigating its impact. The current reporting does not indicate such a process is in place.
    *   **Champion Model Justification**: While XGBoost is the champion, its performance on OOT_01 is concerning, and its interpretability is low. The justification for its champion status needs to be continually reassessed against its real-world performance and the potential benefits of more interpretable models.
    *   **Definition of Accuracy_pct**: The low or zero reported Accuracy_pct for XGBoost on OOT_01 and DEV, respectively, needs clarification. If this metric is intended to reflect some form of prediction correctness, its current values suggest a fundamental issue with the model's output or how this metric is calculated for regression.

---
