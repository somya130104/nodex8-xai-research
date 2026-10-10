# Model Comparison Report: OOS Dataset
> Champion Model: XGBoost  |  Baseline: DEV

## Business Executive

Comparing the available models for Sales Forecasting on OOS reveals distinct trade-offs...

**Performance Leader**:
The FE_LSDV model leads in terms of prediction accuracy on the OOS dataset, with an R² of -0.03. While this is a negative value, indicating it performs worse than a simple average prediction, it is still a better outcome than the XGBoost model's R² of 0.07 on the OOS dataset. However, when looking at the DEV dataset, XGBoost has a better R² of 0.15 compared to FE_LSDV's R² of -0.04.

**Explainability Champion**:
The FE_LSDV model is the clear winner for interpretability. As a linear model, its workings are more transparent and easier to explain compared to the complex tree-based structure of XGBoost.

**Champion Model Assessment** (XGBoost):
Our current champion, XGBoost, performs worse on the OOS dataset compared to FE_LSDV in terms of prediction accuracy (R² of 0.07 vs -0.03). However, XGBoost shows stronger performance on the DEV dataset with an R² of 0.15, significantly outperforming FE_LSDV's R² of -0.04. XGBoost also demonstrates a much higher confidence in its predictions, with 20.16% of its predictions being highly confident on OOS, compared to FE_LSDV's 5.59%. On the DEV dataset, XGBoost has 20.41% high confidence predictions versus FE_LSDV's 5.69%.

**Recommendation**:
For situations requiring the most accurate sales forecasts on the OOS dataset, **FE_LSDV should be used**, despite its poor performance and lack of explainability. However, this comes with a significant risk due to its poor accuracy and limited insight into *why* it is making these predictions.

For all other scenarios, especially where a balance of prediction accuracy and understanding is needed, and for future development, **continue to use XGBoost**. Its higher confidence in predictions and better performance on the DEV dataset make it the more reliable choice for strategic planning and decision-making. We should investigate why FE_LSDV is performing so poorly on OOS and if it can be improved, or if its use should be discontinued.

---

## Data Scientist

Comparing the available models for Sales Forecasting on OOS reveals distinct trade-offs.

**Performance Leader**:
The XGBoost model demonstrates a superior R² of **0.071066** on the OOS dataset compared to the FE_LSDV model's R² of **-0.030662**. While both models exhibit negative Adjusted R² values on OOS, suggesting potential overfitting or a poor fit, XGBoost's positive R² indicates a marginally better explanation of variance. However, it's crucial to note that both models perform poorly on OOS when compared to their DEV dataset performance, where XGBoost achieves an R² of **0.147242** and FE_LSDV achieves **-0.037993**. The MAE for XGBoost on OOS is **116.146733**, which is higher than FE_LSDV's **103.525519**. This suggests FE_LSDV might be more precise in its point predictions on OOS, despite XGBoost's better overall fit. The MAPE values are starkly different, with FE_LSDV at **0.615996** and XGBoost at **2.159526** on OOS. This suggests FE_LSDV is significantly more accurate in terms of percentage error.

**Explainability Champion**:
The FE_LSDV model, being a Linear model, inherently offers greater explainability due to its simpler structure and interpretable coefficients. XGBoost, a tree-based ensemble, while powerful, typically requires more advanced techniques (like SHAP values) to derive feature importance and understand its decision-making process.

**Champion Model Assessment** (XGBoost):
The XGBoost model, designated as the champion, shows a notable decline in performance when moving from the DEV dataset to the OOS dataset. Its R² drops from **0.147242** on DEV to **0.071066** on OOS, a decrease of over 50%. Similarly, the MAE increases from **107.940794** on DEV to **116.146733** on OOS. The MAPE also sees a significant jump from **2.080663** on DEV to **2.159526** on OOS, although this is still considerably higher than FE_LSDV's MAPE. The High Confidence Percentage on OOS is **20.1646**, which is a substantial improvement over FE_LSDV's **5.5874**, indicating XGBoost is more confident in its predictions, even if the overall accuracy on OOS is questionable.

**Recommendation**:
Given the current performance on the OOS dataset, the **FE_LSDV** model should be considered for its lower MAE (**116.146733** vs **103.525519**) and significantly better MAPE (**0.615996** vs **2.159526**), making it more precise in its point predictions and percentage error. Its linear nature also provides better interpretability. However, the **XGBoost** model, despite its poorer performance on OOS metrics like MAE and MAPE, shows a much higher High Confidence Percentage (**20.1646** vs **5.5874**), suggesting it is more decisive in its predictions. If higher confidence in predictions is paramount, and the higher error margins are acceptable, XGBoost might still be preferred. Further investigation into feature contributions and potential distributional shifts between DEV and OOS is strongly recommended for both models to understand the performance degradation. If interpretability is a key requirement, FE_LSDV is the clear choice.

---

## Risk Manager

Comparing the available models for Sales Forecasting on OOS reveals distinct trade-offs...

**Performance Leader**:
The **FE_LSDV** model demonstrates superior performance on the OOS dataset, particularly in terms of **MAE** (103.53 vs 116.15) and **RMSE** (572.20 vs 543.23). While the **XGBoost** model has a higher **R²** on OOS (0.07 vs -0.03), this is significantly overshadowed by its substantially worse **MAPE** (2.16 vs 0.62). The **High_Conf_pct** is also considerably higher for XGBoost (20.16% vs 5.59%), indicating a greater proportion of predictions falling within acceptable confidence intervals, which is a positive for XGBoost. However, the overall accuracy and error metrics for FE_LSDV are more favourable.

**Explainability Champion**:
The **FE_LSDV** model is the clear champion for interpretability. Its `family` is listed as "Linear", suggesting a more straightforward, interpretable structure compared to the "Tree" based family of XGBoost. Linear models generally allow for direct interpretation of feature coefficients, making it easier to understand the drivers of predictions.

**Champion Model Assessment** (XGBoost):
The current champion model, **XGBoost**, trails the **FE_LSDV** model on key performance metrics when evaluated on the OOS dataset. Specifically, XGBoost exhibits a higher **MAE** (116.15 vs 103.53) and a worse **MAPE** (2.16 vs 0.62) compared to FE_LSDV. While XGBoost's **R²** on OOS (0.07) is better than FE_LSDV's (-0.03), this is a single metric and does not outweigh the significantly poorer error rates. On the DEV dataset, XGBoost also shows higher MAE (107.94 vs 98.67) and MAPE (2.08 vs 0.61) than FE_LSDV, indicating a consistent pattern of FE_LSDV outperforming the current champion. The **Accuracy_pct** of 0.0 for XGBoost on OOS is a critical concern and suggests the model is not meeting any accuracy thresholds, whereas FE_LSDV reports 38.40%. The **High_Conf_pct** for XGBoost (20.16%) is substantially higher than FE_LSDV (5.59%), implying it is more confident in its predictions, but this confidence is not translating into better accuracy.

**Recommendation**:
Given the current performance data on the OOS dataset, a strong case can be made to challenge the champion status of **XGBoost**. The **FE_LSDV** model demonstrates significantly better performance metrics, particularly in error reduction (MAE, RMSE, MAPE) and overall accuracy (Accuracy_pct), which are critical for sales forecasting. Therefore, **FE_LSDV** should be considered for promotion to champion status for the OOS dataset.

**Governance Concerns and Further Assessment:**

1.  **Model Sensitivity to High-Weighted Features**: The provided data does not explicitly detail feature importances or weights for either model. A critical next step is to identify the high-weighted features for both FE_LSDV and XGBoost. Subsequent analysis should assess how sensitive the models' predictions are to changes in these specific features. If FE_LSDV is overly sensitive to a few key features, it could indicate a risk of overfitting or an undue reliance on potentially volatile input drivers.

2.  **Over-Reliance Risks**: The drastically low `Accuracy_pct` of 0.0% for XGBoost on the OOS dataset is a major governance concern. This suggests the model is failing to meet even the most basic accuracy requirements. If this model is currently being used operationally, it implies a significant over-reliance on a model that is demonstrably underperforming, potentially leading to poor business decisions and financial losses. The high `High_Conf_pct` for XGBoost (20.16%) combined with low accuracy is particularly concerning, as it suggests the model is confidently making incorrect predictions.

3.  **Temporal Stability and Regulatory Implications of Data Drift**: The comparison is based on a single OOS dataset. To assess temporal stability and the potential for data drift, it is imperative to:
    *   **Monitor performance over time**: Track the performance metrics of both FE_LSDV and XGBoost on subsequent out-of-sample datasets. A decline in FE_LSDV's performance or a sudden improvement in XGBoost's performance could signal data drift.
    *   **Analyze feature distributions**: Compare the distributions of key features in the training data, DEV dataset, and OOS dataset. Significant shifts in feature distributions are indicators of data drift.
    *   **Regulatory Compliance**: For sales forecasting, regulations might not be directly applicable to the model itself, but inaccurate forecasts can have regulatory implications (e.g., in financial reporting, inventory management compliance, or antitrust concerns if forecasting is used for market manipulation). Data drift that leads to consistently inaccurate forecasts could expose the organization to scrutiny. The `family: Linear` for FE_LSDV might be less susceptible to certain types of drift than tree-based models, but this needs empirical validation.

4.  **Reliability for Long-Term Forecasting**:
    *   **FE_LSDV**: Its superior performance on OOS and interpretability make it a more reliable candidate for short-to-medium term forecasting. However, its linear nature might limit its ability to capture complex, non-linear trends that often emerge in long-term forecasts.
    *   **XGBoost**: Despite its poor current performance on OOS, tree-based models like XGBoost can sometimes capture complex interactions and non-linearities better than linear models, which can be advantageous for long-term forecasting. However, its current unreliability on OOS makes it unsuitable for any forecasting horizon without significant investigation and potential retraining or feature engineering. The extremely low `Accuracy_pct` on OOS is a red flag for its use in any capacity, let alone long-term.

**Explicit Governance Concerns:**

*   **Champion Model Inadequacy**: The current champion model (XGBoost) is performing exceptionally poorly on the OOS dataset, evidenced by its 0.0% `Accuracy_pct` and high error metrics. This indicates a potential failure in the model selection or champion designation process.
*   **Lack of Feature Importance Analysis**: The absence of feature importance data prevents an assessment of model sensitivity and potential bias stemming from specific input variables.
*   **Insufficient Temporal Monitoring**: The reliance on a single OOS dataset limits the ability to detect data drift and assess temporal stability, which are crucial for long-term reliability and regulatory compliance.
*   **Misleading Confidence**: XGBoost's high `High_Conf_pct` coupled with low accuracy suggests a model that is confidently wrong, posing a significant risk if its predictions are acted upon.

---
