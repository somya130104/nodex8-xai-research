# Explanation Consensus & Triangulation: FE_LSDV — Decile Level
> Champion Model: No  |  Benchmark: DEV
> Frameworks: SHAP, LIME, Surrogate Regression

## Business Executive

Here's an analysis of the explanation method consensus for the FE_LSDV model across demand deciles:

## Consensus Features

There are **zero features** where all three explainability methods (SHAP, LIME, and Surrogate) consistently agree across any decile in the DEV dataset. This indicates a general lack of strong, unified understanding of the model's drivers across different customer segments.

## Method Disagreements

The majority of identified features fall into the **Divergent** category, meaning only one method flagged them as important within the top 10 drivers for a given decile. This highlights that different methods are capturing distinct aspects of the model's behavior, and no single method provides a complete picture.

Here's a breakdown of what each method's unique findings might reveal:

*   **SHAP:** When SHAP identifies a feature as important and other methods do not, it suggests this feature has a significant marginal impact on the model's predictions when considered in isolation from the full model's perspective. This can point to complex interactions or subtle influences that simpler local or global approximations might miss. For example, in DEV Decile 1, SHAP highlights `any_holiday` (rank 1), `promo_intensity` (rank 3), and `Segment_Effect` (rank 2) while other methods do not rank them as highly or at all. This suggests these factors are key drivers for this specific segment, according to SHAP's global, model-agnostic approach.

*   **LIME:** LIME's unique insights are difficult to assess directly from this data as it shows no unique top features. However, if LIME were to highlight a feature others miss, it would suggest that the feature has a strong local linear influence for a specific prediction or group of predictions. This is useful for understanding *why* a particular outcome occurred but might not generalize well.

*   **Surrogate Regression:** When Surrogate Regression identifies a feature as important and others do not, it indicates that this feature has a strong, consistent linear relationship with the model's output across the entire dataset (or a large portion of it). For example, in DEV Decile 1, Surrogate Regression highlights `global_t` (rank 1) and `zero_streak` (rank 3) as important, while SHAP only ranks `global_t` at 9. This suggests `global_t` has a strong linear influence that the Surrogate model captures effectively, but SHAP sees it as less impactful in its marginal contribution analysis. In DEV Decile 1, `is_promo` is ranked highly by SHAP (rank 6) and Surrogate (rank 2), showing some agreement but not full consensus.

## Framework-Specific Insights

*   **SHAP Only:** Features like `Segment_Effect` in DEV D1, `promo_intensity` in DEV D1 and D3, and `any_holiday` in DEV D1, D2, D3, D4, D5, D6, D7, D8, D9, D10 are frequently highlighted by SHAP alone. This suggests these features have a notable impact that SHAP's marginal contribution analysis captures.

*   **LIME Only:** No features are exclusively identified by LIME in the top 10 across any decile in the provided data.

*   **Surrogate Only:** Features like `global_t` in DEV D1, D2, D3, D4, D7, D10, and `zero_streak` in DEV D1, D3, D4, D7, D10, D6, D2 are often unique to the Surrogate model. This points to features with strong linear relationships that the Surrogate model picks up globally. For instance, in DEV D1, `global_t` is ranked #1 by Surrogate but #9 by SHAP, indicating a strong linear driver that SHAP's more complex analysis doesn't emphasize as much.

## Confidence Assessment

**Overall Explanation Confidence: Low**

*   **Strong Consensus:** 0 features.
*   **Divergent:** A significant majority of features identified (14 out of 17 in DEV D1, 12 out of 16 in DEV D10, and similar ratios across other deciles) are only flagged by a single method.

This low level of consensus across methods means we cannot place high confidence in any single feature's importance without further investigation. The divergence suggests that our understanding of the FE_LSDV model's drivers is fragmented, with each method offering a potentially incomplete or biased view.

## Recommendations

1.  **High Confidence Drivers (with Caution):** Features that appear with "Partial Agreement" across multiple deciles, even if not all methods agree, warrant attention. For example:
    *   **`any_holiday`**: Shows partial agreement in DEV D1, D2, D3, D4, D8, D9, D10 and OOS D1, D2, D3, D4, D5, D6, D8, D9. This suggests it's generally an important factor, but its precise influence varies by decile and method.
    *   **`promo_intensity`**: Appears with partial agreement in several DEV deciles (D1, D2, D3, D4, D5, D6, D7, D8, D9, D10) and OOS deciles (D1, D2, D3, D4, D5, D6, D8, D9, D10). This is a consistently important driver.
    *   **`global_t`**: Shows partial agreement in several DEV deciles (D1, D2, D3, D4, D7) and OOS deciles (D1, D2, D3, D4, D6, D7, D8, D9, D10). This appears to be a significant driver across many segments.

2.  **Features Requiring Caution due to Method Disagreement:**
    *   **Features unique to SHAP (e.g., `Segment_Effect` in DEV D1):** These might represent complex interactions or subtle influences. Investigate further to understand their business impact, especially if they are highly ranked.
    *   **Features unique to Surrogate Regression (e.g., `zero_streak` in DEV D1, `August` in DEV D9/D10):** These likely represent strong linear trends. The extreme values seen for `August` in DEV D9 and D10 (400324.789519 and 109735611.731626 respectively) in the Surrogate model are concerning and suggest potential data issues or model instability for these high-demand deciles. This needs immediate investigation.
    *   **Features unique to LIME (if any emerge):** These would indicate local effects that might not be broadly applicable.

3.  **Extreme Decile Focus:**
    *   **High Demand (D9, D10):** In DEV, `promo_intensity` is a very strong driver in D10 (rank 1) and D9 (rank 1.9) according to SHAP. Surrogate also shows extremely high values for `August` and `promo_x_windex` in these deciles, which is a red flag for model reliability at peak demand. The contrast between SHAP's emphasis on promotions and Surrogate's unusual spikes for `August` and `promo_x_windex` suggests a potential instability or over-reliance on specific features in high-demand scenarios.
    *   **Low Demand (D1):** SHAP consistently ranks `any_holiday` highest in DEV D1. Surrogate also highlights `global_t` as a key driver. The divergence here suggests that while holidays are important for lower demand segments, other factors like `global_t` are also critical, and understanding their interplay is key.

4.  **Cross-Dataset Comparison (DEV vs. OOS):** Generally, the patterns of agreement and disagreement between methods are similar between the DEV and OOS datasets. For example, `any_holiday`, `promo_intensity`, and `global_t` show partial agreement across deciles in both datasets, indicating a consistent, albeit not perfectly agreed upon, set of drivers. The extreme values observed for `August` and `promo_x_windex` in DEV D9/D10 using the Surrogate model are not mirrored in OOS D9/D10, suggesting these might be specific to the training data's high-demand segments or data anomalies.

**Action:** Prioritize investigating the features with partial agreement, paying close attention to those with extremely high or unusual importance scores, particularly in the highest deciles (D9, D10) of the DEV dataset. The Surrogate model's findings for `August` and `promo_x_windex` in DEV D9/D10 warrant immediate review for data quality or model behavior issues. Further analysis should aim to triangulate findings from multiple methods to build a more robust understanding of model drivers.

---

## Data Scientist

Here's a report on the explanation method consensus for the `FE_LSDV` model at the decile level, comparing the consensus across different deciles and referencing the DEV dataset as the benchmark.

## Consensus Features

There are **zero** features that achieved "Strong Consensus" (agreement across all three frameworks) for the `FE_LSDV` model at the decile level when analyzed against the DEV dataset. This indicates a lack of robust agreement across SHAP, LIME, and Surrogate regression on the most critical drivers within any specific decile.

## Method Disagreements

The majority of features exhibit "Divergent" consensus, meaning only one framework identifies them as important within the top N features for a given decile. This divergence highlights the distinct sensitivities and assumptions of each explanation method:

*   **SHAP (Marginal Contribution):** When SHAP identifies a feature as important and other methods do not, it suggests that this feature has a significant marginal impact on the model's prediction for instances within that decile, considering the full model's complexity. For example, in **D1** (DEV), `Segment_Effect` is ranked 2nd by SHAP but not captured by LIME or Surrogate. This implies SHAP's accurate accounting for feature interactions within the `FE_LSDV` model reveals its importance for this decile, which might be missed by local approximations or global linear models. Similarly, in **D7** (DEV), SHAP flags `any_holiday` (rank 1), `Segment_Effect` (rank 2), and `promo_intensity` (rank 3) as top drivers while others do not. This suggests SHAP is capturing their unique contribution in that decile's context.

*   **LIME (Local Linear Approximation):** LIME's results are not explicitly provided in the consensus data (indicated by `null` ranks), making it difficult to pinpoint unique LIME contributions from this specific output. However, its nature is to provide a local explanation. If LIME were to highlight a feature not seen by others, it would imply that, in the immediate vicinity of specific predictions within a decile, that feature is a strong linear predictor, even if its global or marginal impact is less pronounced.

*   **Surrogate (Global Linear Model):** When Surrogate regression identifies a feature as important and others do not, it suggests that, on a global scale within that decile, the feature exhibits a consistent linear relationship with the model's output. For instance, in **D1** (DEV), `global_t` is ranked 1st by Surrogate, but only 9th by SHAP. This implies that while `global_t` has a consistent linear influence across the decile according to the Surrogate, its marginal contribution as calculated by SHAP might be less significant or more complex due to interactions. The extreme values for `August` and `promo_x_windex` in **D9** and **D10** (OOS) by Surrogate (e.g., `August` at 400324.789519 in D9 DEV) are particularly anomalous and indicative of a potential issue with the Surrogate model's fit or the data distribution in these high-demand deciles.

The prevalence of divergent features indicates that no single explanation method provides a complete picture of feature importance across all deciles. The interpretation of these divergent features depends heavily on understanding the underlying methodology of the framework that identified them.

## Framework-Specific Insights

*   **SHAP:** SHAP consistently ranks holiday-related features and promotional features highly across various deciles. For instance, `any_holiday` appears in the top 10 for most deciles in DEV, often with a strong rank (e.g., rank 1 in D1, D2, D7, D8, D6 DEV). `promo_intensity` also shows up frequently, particularly in higher deciles like **D10** (DEV) where it is ranked 1st by SHAP (avg_rank 6.33 across frameworks).

*   **LIME:** LIME's specific contributions are not detailed in the provided consensus data.

*   **Surrogate:** The Surrogate model reveals interesting patterns, especially in extreme deciles.
    *   In **D1** (DEV), `global_t` is ranked 1st by Surrogate, suggesting a strong linear driver for this lowest demand decile.
    *   In **D10** (DEV), `August` is ranked 1st by Surrogate, with an extremely high magnitude (109735611.731626), alongside `promo_x_windex` and `holiday_x_windex` also showing very large magnitudes. This suggests that for the highest demand decile in the DEV dataset, the Surrogate model identifies specific temporal and interaction features as overwhelmingly dominant, which is a significant deviation from SHAP's findings.
    *   In **D6** (OOS), `global_t` is ranked 1st by Surrogate, mirroring its importance in **D1** (DEV).

The Surrogate model's extreme magnitudes in high-demand deciles (D9, D10) of the OOS dataset, particularly for `August` and interaction terms, warrant further investigation. This could indicate a distributional shift or an artifact of the global linear approximation in these specific segments.

## Confidence Assessment

Overall explanation confidence is **Medium**.

*   **Strong Consensus:** 0 features.
*   **Partial Agreement:** Between 1 and 6 features per decile (e.g., D8 DEV has 6, D2 DEV has 5). This indicates some level of agreement, but not across all methods.
*   **Divergent:** The vast majority of features are divergent, with 10-18 features per decile being flagged by only one method. For instance, in **D1** (DEV), 14 out of 17 features are divergent.

The high number of divergent features suggests that while the model `FE_LSDV` is making predictions, there isn't strong, consistent agreement among the leading explanation methods on *why* across all deciles. This necessitates careful interpretation, as trust in any single feature's explanation is limited when consensus is low. The lack of strong consensus implies that the model's behaviour might be complex and context-dependent across different demand levels.

## Recommendations

1.  **High Confidence Features:**
    *   **`any_holiday`**: This feature shows consistent partial agreement across many deciles in the DEV dataset, often ranking highly (e.g., rank 1 in D1, D2, D7, D8 DEV). Its presence across different explanation methods, even if not always in strong consensus, suggests it is a reliable driver of model predictions.
    *   **`promo_intensity`**: Similar to `any_holiday`, this feature frequently appears in the top N and shows partial agreement across several deciles. Its importance is particularly pronounced in higher deciles like **D10** (DEV), where SHAP ranks it 1st.

2.  **Features Requiring Caution due to Method Disagreement:**
    *   **`global_t`**: While flagged by SHAP and Surrogate as important in several deciles (e.g., rank 1 in D1 DEV by Surrogate, rank 1 in D7 DEV by Surrogate), its overall consensus is often "Partial Agreement" or "Divergent." The significant difference in ranking between SHAP and Surrogate in **D1** (DEV) (rank 9 vs. rank 1) warrants caution.
    *   **`Segment_Effect`**: This feature is often ranked highly by SHAP (e.g., rank 2 in D1 DEV) but frequently appears as "Divergent." Its importance might be captured differently by each method, suggesting its influence could be more nuanced than a simple linear relationship.
    *   **Temporal/Interaction Features (e.g., `August`, `promo_x_windex`, `holiday_x_windex`)**: The Surrogate model's extremely large magnitudes for these features in high-demand deciles (D9, D10) of the OOS dataset, compared to their more moderate rankings by SHAP in the DEV dataset, are a major red flag. This indicates a potential distributional shift between DEV and OOS for these deciles, or that the Surrogate model is overfitting or misinterpreting these segments. The explanation for these features is highly unstable and requires further investigation before business reliance.

**Cross-Dataset Comparison (DEV as Benchmark):**
The analysis of the OOS dataset reveals significant discrepancies, particularly in higher deciles (D9, D10) for features like `August`, `promo_x_windex`, and `holiday_x_windex` when using the Surrogate model. The magnitudes observed in OOS are orders of magnitude larger than those in DEV, suggesting that the model's behavior or the data distribution for these deciles has shifted considerably. This contrasts with features like `any_holiday` and `promo_intensity`, which show more consistent partial agreement across deciles in DEV and appear to have more stable importance, though their absolute impact in OOS is not directly comparable without further analysis. The high divergence across most features in both datasets indicates a general challenge in achieving strong tripartite consensus for the `FE_LSDV` model.

---

## Risk Manager

## Explanation Consensus & Triangulation Report: FE_LSDV Model

This report assesses the consensus and divergence of explanation methods (SHAP, LIME, Surrogate Regression) across different deciles of the FE_LSDV model, using the DEV dataset as the benchmark.

### Consensus Features

There are **zero features** with "Strong Consensus" (agreement across all three frameworks) at any decile level. This indicates a general lack of robust agreement on the most influential features across the three explanation methods.

### Method Disagreements

A significant number of features exhibit "Divergent" consensus, meaning only one framework flagged them as important within the top N features for a given decile. This divergence highlights the distinct perspectives each method offers:

*   **SHAP (Marginal Contribution):** When SHAP uniquely identifies a feature, it suggests that this feature's marginal contribution to the prediction is significant in the context of the full FE_LSDV model. For example, in **DEV D1**, SHAP uniquely highlights `Segment_Effect` (rank 2) and `promo_intensity` (rank 3). This implies that while other methods might not rank them as highly globally or locally, their individual impact within the complex model structure is notable.
*   **LIME (Local Linear Approximation):** The provided data does not show any features exclusively identified by LIME. This is likely due to LIME's local nature, which focuses on explaining individual predictions. If LIME were to identify a unique feature, it would indicate a driver that is highly influential for specific instances within a decile but perhaps not consistently across the broader decile or globally.
*   **Surrogate Regression (Global Interpretable Model):** Features uniquely identified by the Surrogate method suggest that a global, simpler interpretable model (like a panel regression) finds them to be important drivers. For instance, in **DEV D1**, Surrogate uniquely flags `global_t` (rank 1) and `zero_streak` (rank 3). This indicates that these features have a consistent linear relationship with the model's output across the entire dataset, even if SHAP or LIME do not prioritize them in their specific analyses. This can reveal underlying linear trends or relationships that more complex methods might abstract away.

The prevalence of "Divergent" features implies that no single feature is unequivocally agreed upon as a top driver across all explanation frameworks for any given decile. This necessitates a careful interpretation of the model's behavior, as different methods point to different underlying mechanics.

### Framework-Specific Insights

*   **SHAP-Exclusive Features:** In the **DEV** dataset, SHAP uniquely identifies features like `Segment_Effect` (DEV D1, rank 2), `promo_intensity` (DEV D1, rank 3), `promo_intensity` (DEV D10, rank 1), `any_holiday` (DEV D7, rank 1), `promo_intensity` (DEV D4, rank 2), `any_holiday` (DEV D3, rank 1), `promo_intensity` (DEV D9, rank 1), `is_promo` (DEV D8, rank 2), `promo_intensity` (DEV D6, rank 2), `any_holiday` (DEV D5, rank 1), `promo_intensity` (DEV D3, rank 2). These features are highlighted by SHAP's assessment of marginal contributions, suggesting their importance is most evident when considering their impact within the full model's context.
*   **LIME-Exclusive Features:** No features are exclusively identified by LIME in the provided data. This suggests that LIME's local explanations do not introduce unique top-ranked features compared to SHAP and Surrogate in the top N for these deciles.
*   **Surrogate-Exclusive Features:** Surrogate uniquely identifies features such as `global_t` (DEV D1, rank 1), `zero_streak` (DEV D1, rank 3), `global_t` (DEV D10, rank 1), `holiday_x_windex` (DEV D10, rank 3), `global_t` (DEV D7, rank 1), `roll_mean_13` (DEV D7, rank 2), `is_year_start` (DEV D2, rank 3), `quarter` (DEV D4, rank 2), `roll_mean_13` (DEV D3, rank 6), `Spring` (DEV D9, rank 2), `weeks_to_next_promo` (DEV D9, rank 8), `roll_mean_4` (DEV D6, rank 2), `roll_mean_4` (DEV D1, rank 1), `lag_1` (DEV D1, rank 2), `roll_mean_13` (DEV D4, rank 1), `cust_hist_mean` (DEV D4, rank 2), `cust_hist_mean` (DEV D3, rank 1), `roll_mean_13` (DEV D3, rank 2), `roll_mean_4` (DEV D6, rank 2), `roll_mean_4` (DEV D1, rank 1), `lag_1` (DEV D1, rank 2), `roll_mean_13` (DEV D4, rank 1), `cust_hist_mean` (DEV D4, rank 2), `roll_mean_13` (DEV D3, rank 2), `cust_hist_mean` (DEV D3, rank 1). These features are prominent in the global linear approximation provided by the surrogate model, suggesting they represent consistent linear drivers across the data. Notably, `August` shows extremely high values in the Surrogate pivot for **DEV D9 and D10**, indicating a strong linear association in these high-demand deciles according to the surrogate model, which is not as consistently captured by SHAP or LIME in the top N.

### Confidence Assessment

**Overall Explanation Confidence: Low to Medium.**

The analysis reveals a **low level of strong consensus (0 features)** and a **high level of divergent consensus (14+ features per decile)** when comparing SHAP, LIME, and Surrogate Regression on the DEV dataset. This lack of agreement across multiple explanation methods significantly reduces the confidence in any single feature being a universally agreed-upon driver.

*   **Strong Consensus:** 0 features.
*   **Partial Agreement:** The number of features with partial agreement varies by decile, ranging from 1 (DEV D7) to 6 (DEV D8, OOS D10). This indicates that while some features might be ranked highly by two out of three methods, there's no complete alignment.
*   **Divergent:** The number of divergent features is consistently high, ranging from 8 (DEV D8, OOS D10) to 18 (DEV D7, D4, D6). This signifies that for most top-ranked features, only one explanation method considers them important enough to include in the top N for a given decile.

This pattern suggests that the FE_LSDV model's behavior is complex and can be interpreted differently by various explainability techniques. Relying on any single explanation method for critical decision-making would be precarious.

### Recommendations

1.  **High Confidence Features (with caveats):**
    *   Features consistently appearing in **Partial Agreement** across multiple deciles are candidates for higher confidence, but still require careful consideration. Examples include `any_holiday` (appears in Partial Agreement for multiple deciles in both DEV and OOS), `promo_intensity`, `global_t`, `is_promo`, `NewYear`, `Christmas`, and `September`. Their partial agreement suggests they are important, but the lack of strong consensus means their exact ranking and impact might differ depending on the specific decile and explanation method.
    *   Features that show **strong, consistent importance in the Surrogate pivot for high-demand deciles (D9, D10)**, such as `August` and `promo_x_windex`, should be noted. While the Surrogate model highlights them, the lack of corroboration from SHAP and LIME means their influence might be a linear artifact captured by the surrogate rather than a complex interaction within the FE_LSDV model itself. This requires further investigation.

2.  **Features Requiring Caution:**
    *   **All "Divergent" features:** These features should be treated with caution. Their importance is only recognized by a single method, making them unreliable as sole justifications for business decisions. For example, `Segment_Effect` is often highlighted by SHAP but not consistently by others. `global_t` is frequently highlighted by Surrogate but not always by SHAP or LIME.
    *   **Features with significant differences between DEV and OOS:** While not explicitly detailed in the provided consensus data, a thorough model risk assessment would involve comparing feature importance across DEV and OOS datasets. Any substantial shifts in feature importance between these datasets would indicate potential **data drift** and a need to re-evaluate the model's reliability and the regulatory implications of using it for forecasting.
    *   **Temporal Stability:** The provided data focuses on decile-level consensus. A critical next step is to assess the temporal stability of these feature importances. If feature rankings fluctuate wildly over time within deciles, it suggests the model's drivers are not stable, impacting its reliability for long-term forecasting.
    *   **Model Sensitivity:** The analysis of feature importance itself contributes to understanding model sensitivity. Features with high average ranks across multiple deciles (e.g., `promo_intensity`, `any_holiday`, `global_t`, `Segment_Effect`, `NewYear`, `Christmas`, `September`) indicate areas where the model is sensitive. However, the lack of consensus means we cannot confidently state *how* it is sensitive (e.g., positively or negatively correlated, linear or non-linear) without further investigation.

3.  **Governance Concerns:**
    *   **Lack of Strong Consensus:** The absence of strong consensus across multiple established explanation methods (SHAP, LIME, Surrogate) is a significant governance concern. It implies that there is no single, universally agreed-upon explanation for the model's behavior, hindering transparency and accountability.
    *   **Over-reliance Risk:** Relying on the FE_LSDV model for critical decisions, especially long-term forecasting, is risky given the current level of explanation ambiguity. The business could be making decisions based on features that are not consistently understood or validated across different analytical perspectives.
    *   **Data Drift and Regulatory Implications:** The analysis presented does not explicitly cover data drift between DEV and OOS. However, the divergence in feature importance between these datasets (e.g., `August`'s prominence in Surrogate for OOS D9/D10 vs. its lower rank in DEV) warrants investigation. If data drift is present and impacts key drivers, it could lead to inaccurate forecasts and potential non-compliance with regulations that require models to be robust and well-understood.
    *   **Model Risk Management:** The current state of explanation consensus suggests a medium to high model risk. The model's internal workings are not transparently and consistently understood, increasing the likelihood of unexpected behavior, misinterpretation, and potential biases going unnoticed.

**Next Steps:**

*   Conduct a temporal analysis of feature importance to assess stability.
*   Investigate the divergence in feature importance between the DEV and OOS datasets to identify potential data drift and its impact.
*   Perform deeper dives into the unique findings of each framework, particularly the extreme values observed in the Surrogate pivot for high-demand deciles (D9, D10) in the OOS dataset.
*   Explore interaction effects between features, as these might explain why different methods highlight different aspects of feature influence.
*   Consider additional explanation methods or more granular analysis within deciles to build a more robust understanding of the FE_LSDV model's drivers.

---
