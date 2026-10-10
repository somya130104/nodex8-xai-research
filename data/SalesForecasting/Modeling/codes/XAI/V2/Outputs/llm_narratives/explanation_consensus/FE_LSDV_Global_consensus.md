# Explanation Consensus & Triangulation: FE_LSDV — Global Level
> Champion Model: No  |  Benchmark: DEV
> Frameworks: SHAP, LIME, Surrogate Regression

## Business Executive

## Explanation Consensus & Triangulation Report: FE_LSDV Model (Global Level)

This report analyzes the agreement between three explanation methods (SHAP, LIME, and Surrogate) across different datasets (DEV, OOS, OOT_01) for the FE_LSDV model at a global level. The goal is to understand the reliability of the model's explanations and identify key drivers.

### Consensus Features

These are the features where SHAP, LIME, and Surrogate all agree are among the top drivers. This strong consensus makes them the most trustworthy indicators of what influences the model's predictions.

*   **any_holiday** (Average Rank: 2.0): This feature consistently appears as a top driver across all three methods, indicating it's a significant factor in the model's decisions.
*   **NewYear** (Average Rank: 3.67): Another strong performer, this feature is reliably identified as important.
*   **promo_intensity** (Average Rank: 4.33): While its exact ranking varies, all methods agree it's a key driver.
*   **Christmas** (Average Rank: 5.0): Consistently recognized as important by all frameworks.
*   **September** (Average Rank: 7.33): All three methods acknowledge its influence.
*   **August** (Average Rank: 7.33): Similar to September, this feature is consistently flagged.

### Method Disagreements

Disagreements between methods highlight areas where the model's behavior is complex or where the explanation methods have different perspectives.

*   **Divergent Features:**
    *   **Segment_Effect**: Only SHAP identifies this as a top driver. SHAP measures the marginal contribution of a feature using the entire model. This suggests that once other factors are accounted for, "Segment_Effect" has a notable impact.
    *   **is_year_start**: Only Surrogate flags this as important. Surrogate models build a simpler, interpretable model to mimic the complex one. Its prominence here suggests it has a clear, albeit potentially simplified, global effect.
    *   **VetreansDay**: Only Surrogate identifies this. Similar to "is_year_start," this suggests a global pattern that the Surrogate model captures well.
    *   **promo_x_windex**: Only LIME flags this. LIME explains individual predictions by approximating the model locally. This implies "promo_x_windex" might have a significant impact in specific, localized scenarios.

Understanding these divergences can reveal nuanced aspects of the model. SHAP shows the feature's overall impact. LIME points to its influence in specific situations. Surrogate highlights clear, global relationships.

### Framework-Specific Insights

Each framework can highlight features that others overlook due to their unique analytical approach.

*   **SHAP Uniquely Surfaces:**
    *   **Segment_Effect** (Rank 3 in SHAP): As noted above, this is only identified by SHAP.
    *   **global_t** (Rank 10 in SHAP): This feature is only ranked by SHAP among the top 10.
*   **LIME Uniquely Surfaces:**
    *   **promo_x_holiday** (Rank 3 in LIME): While also noted by SHAP, LIME places it higher, suggesting strong local impact.
    *   **Spring** (Rank 8 in LIME): Only LIME ranks this feature.
    *   **promo_x_windex** (Rank 8 in LIME): This is exclusively highlighted by LIME.
*   **Surrogate Uniquely Surfaces:**
    *   **NewYear** (Rank 1 in Surrogate): While all agree it's important, Surrogate gives it the top spot, indicating a very strong, clear global relationship.
    *   **is_year_start** (Rank 2 in Surrogate): This is only ranked by Surrogate.
    *   **VetreansDay** (Rank 9 in Surrogate): This is exclusively highlighted by Surrogate.

### Confidence Assessment

**Overall Confidence: Medium**

*   **Strong Consensus Features:** 6 features show strong agreement across all three methods. This provides a solid foundation of trust in these key drivers.
*   **Divergent Features:** 2 features are identified by only one method. This signals areas where caution is needed, as the influence is not consistently recognized across different analytical lenses.

While there's good agreement on core drivers, the presence of divergent features suggests that a deeper dive might be needed for a complete understanding of the model's behavior. The level of agreement is comparable between the DEV benchmark and the OOT_01 dataset, with 6 strong consensus features in both. However, OOT_01 shows more divergence (6 features) compared to DEV (2 features).

### Recommendations

1.  **High Confidence Features:** The business can rely with high confidence on **any_holiday**, **NewYear**, **promo_intensity**, **Christmas**, **September**, and **August** as the primary drivers of the FE_LSDV model's predictions globally. These are robustly identified across all explanation methods.
2.  **Features Requiring Caution:**
    *   **Segment_Effect**, **is_year_start**, **VetreansDay**, and **promo_x_windex** require caution. Their influence is not consistently captured by all methods. Further investigation may be needed to understand the specific contexts or types of predictions where these features are most impactful.
    *   Features with **Partial Agreement** (e.g., `is_promo`, `promo_x_holiday`, `global_t`) should be monitored. While not fully divergent, the varying levels of agreement suggest their impact might be more conditional or less pronounced than the strong consensus features.
3.  **Cross-Dataset Comparison:** The agreement on key features between the **DEV benchmark** and the **OOT_01 dataset** is generally consistent. However, the higher number of divergent features in **OOT_01** compared to **DEV** suggests that model behavior or data characteristics might be slightly less predictable in out-of-time scenarios. Therefore, insights derived from **DEV** should be considered the most reliable baseline.

---

## Data Scientist

## Explanation Consensus & Triangulation Report: FE_LSDV Model (Global Level)

This report analyzes the consensus and discrepancies between SHAP, LIME, and Surrogate explanation frameworks for the FE_LSDV model at the Global level, using the DEV dataset as the benchmark.

### Consensus Features

The features exhibiting **Strong Consensus** across all three frameworks (SHAP, LIME, and Surrogate) on the DEV dataset are the most trustworthy drivers of model behavior. These features are consistently identified as important regardless of the explanation methodology employed.

*   **any_holiday**: This feature holds the top rank with an **average rank of 2.0**. Its consistent high importance across all frameworks suggests it's a primary, robust predictor.
*   **NewYear**: With an **average rank of 3.67**, `NewYear` is also strongly supported.
*   **promo_intensity**: This feature achieves an **average rank of 4.33**. While its rank varies (SHAP: 2, LIME: 1, Surrogate: 10), its presence in all three top-ranked lists indicates a significant, albeit potentially complex, influence.
*   **Christmas**: Averaging a rank of **5.0**, `Christmas` is another consistently recognized important feature.
*   **September**: This feature has an **average rank of 7.33**.
*   **August**: Also with an **average rank of 7.33**, `August` is consistently flagged.

The strong agreement on these features suggests they are fundamental to the FE_LSDV model's predictions and can be relied upon for business decisions.

### Method Disagreements

**Divergent** features are those flagged by only one framework, highlighting unique insights or potential limitations of each method.

*   **Segment_Effect**: This feature is only identified as important by **SHAP (rank 3)**. SHAP measures the marginal contribution of a feature to the prediction based on the full model. Its appearance solely in SHAP might indicate that its impact is subtle, cumulative, or non-linear in a way that local approximations (LIME) or global surrogate models struggle to capture effectively.
*   **VetreansDay**: This feature appears only in the **Surrogate model (rank 9)**. The Surrogate model fits a global, interpretable model. Its flagging by Surrogate but not SHAP or LIME suggests it might have a more consistent, linear effect across the dataset that a global interpretable model can easily identify, but which doesn't manifest as a significant marginal contribution in SHAP or a strong local explanation in LIME.
*   **promo_x_windex**: This feature is only flagged by **LIME (rank 8)** in the OOT_01 dataset, not on DEV. Its absence in DEV's top-ranked lists from SHAP and Surrogate, but appearing in LIME for OOT_01, suggests its impact might be highly localized and context-dependent, or specific to the distributional shifts observed in OOT_01.

These divergent findings reveal that:
*   **SHAP's** unique identification of `Segment_Effect` suggests a nuanced, potentially complex contribution at the model level.
*   **Surrogate's** sole flagging of `VetreansDay` points to a consistent, interpretable global effect.
*   **LIME's** (and its absence in DEV) may highlight features with localized, instance-specific importance that doesn't translate to global or marginal significance.

### Framework-Specific Insights

Each framework surfaces unique aspects of the FE_LSDV model's behavior:

*   **SHAP-specific**: **`Segment_Effect`** (rank 3 on DEV) is uniquely identified by SHAP. This indicates a feature whose marginal contribution, when considered across all possible feature combinations from the full model, is significant.
*   **LIME-specific**: On the **DEV** dataset, no feature appears exclusively in LIME's top 10. However, LIME uniquely highlights **`promo_x_holiday`** (rank 3) and **`promo_intensity`** (rank 1) as significantly more important relative to other methods on DEV compared to Surrogate.
*   **Surrogate-specific**: **`VetreansDay`** (rank 9 on DEV) is exclusively identified by the Surrogate model. This suggests a consistent, global linear effect that the Surrogate model can easily model, but which is not a primary driver of marginal contribution (SHAP) or local explanations (LIME).

### Confidence Assessment

The overall explanation confidence for the FE_LSDV model at the Global level, based on the DEV benchmark, is **Medium**.

On DEV, we observe:
*   **6 features** in **Strong Consensus** (`any_holiday`, `NewYear`, `promo_intensity`, `Christmas`, `September`, `August`).
*   **5 features** in **Partial Agreement** (`is_promo`, `promo_x_holiday`, `is_year_start`, `global_t`, `Spring`).
*   **2 features** in **Divergent** categories (`Segment_Effect`, `VetreansDay`).

The presence of a strong consensus on a good number of features (6 out of 10 top features) provides a solid foundation for trust. However, the significant number of features in partial agreement (5) and the presence of divergent features (2) indicate that not all aspects of the model's decision-making are uniformly understood across all explanation frameworks. This suggests that while core drivers are clear, there's room for interpretation and potential for differing explanations in specific scenarios.

**Cross-Dataset Comparison (DEV vs. OOT_01):**
When comparing the level of agreement between DEV and OOT_01, we see a notable decrease in consensus for OOT_01.
*   **DEV**: 6 Strong Consensus, 5 Partial, 2 Divergent.
*   **OOT_01**: 6 Strong Consensus, 3 Partial, 6 Divergent.

This indicates that the explanation frameworks **agree less on OOT_01 than on DEV**. The increase in divergent features from 2 on DEV to 6 on OOT_01, and a decrease in partial agreement from 5 to 3, suggests that distributional shifts or changes in data characteristics in OOT_01 are leading to more varied interpretations by SHAP, LIME, and Surrogate.

### Recommendations

Based on the analysis, here are the recommendations:

**High Confidence Features:**
*   **`any_holiday`**, **`NewYear`**, **`Christmas`**: These are unequivocally important drivers, consistently ranked highly across all frameworks and datasets (DEV, OOS, OOT_01). Business decisions related to holiday promotions and planning should heavily rely on these.
*   **`promo_intensity`**: While its rank varies (LIME ranks it highest on DEV), it consistently appears in the top 3-4 across all frameworks on DEV. Its strong presence across methods justifies high confidence, though understanding its exact mechanism might require further deep dives.

**Features Requiring Caution:**
*   **`is_promo`**: This feature shows **Partial Agreement** on DEV, meaning it's not consistently ranked top-tier by all methods. While it's ranked 5th by SHAP and 4th by LIME on DEV, it's absent from Surrogate's top 10. This suggests its importance might be more context-dependent or non-linear.
*   **`promo_x_holiday`**: Similar to `is_promo`, this feature has **Partial Agreement** on DEV (SHAP: 8, LIME: 3, Surrogate: null). Its absence in Surrogate's top 10 on DEV warrants caution.
*   **`is_year_start`**: This feature shows **Partial Agreement** on DEV (SHAP: null, LIME: 9, Surrogate: 2). Its strong flagging by Surrogate (rank 2) but absence from SHAP's top 10 suggests its impact might be more globally linear than marginal.
*   **`global_t`**: Exhibiting **Partial Agreement** on DEV (SHAP: 10, LIME: null, Surrogate: 7), its importance is not consistently captured.
*   **`Segment_Effect`**: As a **Divergent** feature on DEV, exclusively highlighted by SHAP, its importance should be considered with the understanding that LIME and Surrogate do not independently confirm this magnitude.
*   **`VetreansDay`**: This is another **Divergent** feature on DEV, identified only by Surrogate. Its consistent global linear effect is noted, but its absence from SHAP and LIME means its impact might not be as pronounced in marginal or local interpretations.

The observation that explanation consensus is **lower on OOT_01 than on DEV** is critical. Features that show strong consensus on DEV might be less reliable on OOT_01, and vice-versa. This indicates potential distributional shifts in OOT_01 that the FE_LSDV model is responding to differently, or that the explanation methods are interpreting differently. Further investigation into specific feature distributions and model behavior on OOT_01 is recommended.

---

## Risk Manager

## Explanation Consensus & Triangulation Report: FE_LSDV Model (Global Level)

This report analyses the consensus among SHAP, LIME, and Surrogate Regression explainability frameworks for the FE_LSDV model at the Global level, with explicit reference to the benchmark DEV dataset.

### Consensus Features

The following features exhibit **Strong Consensus** across all three frameworks (SHAP, LIME, and Surrogate) on the **DEV** benchmark dataset, indicating they are the most trustworthy drivers of the model's predictions. Their consistent identification across different methodological approaches suggests robustness and a high likelihood of representing genuine underlying relationships.

*   **any_holiday**: Average Rank: 2.0 (SHAP: 1, LIME: 2, Surrogate: 3)
*   **NewYear**: Average Rank: 3.67 (SHAP: 4, LIME: 6, Surrogate: 1)
*   **promo_intensity**: Average Rank: 4.33 (SHAP: 2, LIME: 1, Surrogate: 10)
*   **Christmas**: Average Rank: 5.0 (SHAP: 6, LIME: 5, Surrogate: 4)
*   **September**: Average Rank: 7.33 (SHAP: 7, LIME: 10, Surrogate: 5)
*   **August**: Average Rank: 7.33 (SHAP: 9, LIME: 7, Surrogate: 6)

These features are deemed reliable drivers because their importance is consistently recognized irrespective of whether the explanation method focuses on marginal feature contributions (SHAP), local linear approximations (LIME), or a global interpretable model (Surrogate).

### Method Disagreements

Divergent features are those flagged by only one framework, highlighting the unique perspectives each method offers. Understanding these differences is crucial for a nuanced interpretation of model behaviour.

*   **Segment_Effect**: Identified only by SHAP (Rank 3) and Surrogate (Rank 10) on **DEV**.
    *   **SHAP's insight**: Suggests that `Segment_Effect` has a significant marginal contribution to predictions when considered in isolation.
    *   **Surrogate's insight**: Indicates that a global linear model captures some importance for `Segment_Effect`.
    *   **LIME's absence**: LIME did not flag this feature in its top 10 for DEV. This could mean that its local linear approximations around specific predictions do not consistently highlight `Segment_Effect` as a key driver, or its importance is distributed across many local explanations.
*   **is_year_start**: Identified by LIME (Rank 9) and Surrogate (Rank 2) on **DEV**.
    *   **LIME's insight**: Suggests that `is_year_start` is important in local prediction contexts.
    *   **Surrogate's insight**: The global interpretable model finds `is_year_start` to be the second most important feature.
    *   **SHAP's absence**: SHAP did not rank `is_year_start` in its top 10 for DEV. This implies that while it might have a local impact or be important in a global linear approximation, its marginal contribution, when averaged across all instances, might be lower than other features.
*   **VetreansDay**: Identified only by Surrogate (Rank 9) on **DEV**.
    *   **Surrogate's insight**: The global interpretable model assigns some importance to `VetreansDay`.
    *   **SHAP and LIME's absence**: Neither SHAP nor LIME highlighted this feature in their top 10 for DEV. This suggests its impact might be highly contextual (and thus not captured by global methods like SHAP's average marginal contribution) or its linear relationship is not strong enough to be picked up by LIME's local approximations.

These disagreements highlight that model behaviour can be perceived differently depending on the explanatory lens. SHAP's focus on marginal contributions can reveal features important in isolation, LIME's local approximations can highlight features that are critical for specific predictions, and Surrogate's global linear model can identify features with a consistent linear trend.

### Framework-Specific Insights

Examining features that appear in only one framework's top 10 provides unique insights:

*   **SHAP Unique**:
    *   **Segment_Effect** (Rank 3): This feature is only identified by SHAP as a top driver on the **DEV** benchmark. This suggests it has a significant marginal impact on the model's output, even if not consistently captured by local approximations or global linear models.

*   **LIME Unique**:
    *   No features are uniquely in LIME's top 10 for the **DEV** benchmark. Features like `promo_intensity` (Rank 1) and `any_holiday` (Rank 2) are also highly ranked by SHAP, and `is_promo` (Rank 4), `Christmas` (Rank 5), `NewYear` (Rank 6), and `August` (Rank 7) have strong consensus or are highly ranked by other methods.

*   **Surrogate Unique**:
    *   **NewYear** (Rank 1): While highly ranked by all methods, its top rank by Surrogate on **DEV** indicates a strong linear relationship captured by the global interpretable model.
    *   **is_year_start** (Rank 2): As noted in Method Disagreements, its strong ranking by Surrogate suggests a significant linear influence.
    *   **VetreansDay** (Rank 9): This feature is only identified by Surrogate on **DEV**, suggesting it has some global linear predictive power not captured by SHAP or LIME's top drivers.

### Confidence Assessment

Overall explanation confidence for the FE_LSDV model at the Global level is **Medium**.

On the **DEV** benchmark dataset, there are 6 features with **Strong Consensus** (all 3 frameworks agree) and 2 features with **Divergent** consensus (only one framework flags them). While the presence of 6 strongly agreed-upon features provides a solid foundation of trust, the 2 divergent features, alongside 5 features with only partial agreement, indicate areas where model interpretability is less certain. The shift in consensus between **DEV** (6 strong, 5 partial, 2 divergent) and **OOT_01** (6 strong, 3 partial, 6 divergent) is concerning. The significant increase in divergent features and decrease in partial agreement on **OOT_01** compared to **DEV** suggests that the model's feature importance is less stable in out-of-time data, potentially due to data drift.

This level of agreement suggests that while key drivers are identifiable, there are nuances and potential inconsistencies in how different interpretability methods capture the model's behaviour, especially when extrapolating to new data distributions (as seen in **OOT_01** compared to **DEV**).

### Recommendations

**High Confidence Features (for business reliance):**

*   **any_holiday**: Consistently ranked as a top driver across all frameworks and datasets. This is a robust indicator of its importance.
*   **NewYear**: Shows strong consensus across frameworks on **DEV** and is still highly ranked in OOS and OOT_01.
*   **promo_intensity**: While its rank fluctuates, it consistently appears in the top 2 across SHAP and LIME on **DEV** and is highly ranked in OOS. Its consistent importance across methods and datasets makes it a reliable feature.
*   **Christmas**: Exhibits strong consensus on **DEV** and remains a significant factor in other datasets.

**Features Requiring Caution (due to method disagreement or cross-dataset instability):**

*   **Segment_Effect**: Its inconsistent flagging across methods (only SHAP and Surrogate on **DEV**) and its disappearance from LIME's top 10 on **OOT_01** warrants caution. Further investigation into why SHAP highlights its marginal contribution while LIME does not locally is recommended.
*   **is_year_start**: While Surrogate ranks it highly on **DEV**, its absence in SHAP's top 10 for **DEV** and its partial agreement on other datasets suggest its impact might be context-dependent or not uniformly captured by all methods.
*   **VetreansDay**: Its sole appearance in the Surrogate's top 10 on **DEV** and its subsequent absence from top lists on other datasets suggest it might be a spurious correlation or a feature whose importance is highly sensitive to the specific data distribution and explanation method.
*   **global_t**: This feature shows partial agreement on **DEV** and **OOS**, but its rank and agreement level are inconsistent. Its importance might be diminishing or changing over time.
*   **promo_x_holiday**: Similar to `global_t`, this feature shows partial agreement on **DEV** and **OOS**, but its rank varies. Its importance might be sensitive to specific promotion types or timing.

**Governance Concerns:**

1.  **Model Sensitivity to High-Weighted Features & Over-Reliance Risk**: The model heavily relies on features like `any_holiday` and `promo_intensity`. While these are intuitively understandable, an over-reliance on them without understanding the nuances of their interaction (e.g., `promo_x_holiday`) could lead to brittle predictions if holiday patterns or promotional strategies change significantly. The strong consensus on these features provides some comfort, but their dominance requires ongoing monitoring.
2.  **Temporal Stability & Regulatory Implications of Data Drift**: The analysis shows a noticeable decrease in framework consensus and an increase in divergent features when moving from the **DEV** benchmark to **OOT_01**. This suggests potential data drift. Regulatory bodies are increasingly scrutinizing models for stability and fairness across different data distributions. A significant shift in feature importance between training/validation data (**DEV**) and out-of-time data (**OOT_01**) could flag the model as unstable and potentially non-compliant if the drift leads to biased or inaccurate outcomes. This instability raises concerns about the model's reliability for long-term forecasting.
3.  **Reliability for Long-Term Forecasting**: The observed decrease in consensus and increase in divergent features on out-of-time data (**OOT_01**) directly impacts the reliability of this model for long-term forecasting. If the feature importance landscape shifts unpredictably, forecasts based on the current model may become increasingly inaccurate over time. The model's behaviour in **OOT_01** is a red flag for its suitability for extended forecasting horizons without re-validation or recalibration.
4.  **Methodological Discrepancies**: The differences in feature importance surfaced by SHAP, LIME, and Surrogate highlight that the "true" drivers are not always clear-cut. While consensus on key features is good, divergent features like `Segment_Effect` and `VetreansDay` (on **DEV**) introduce uncertainty. Governance requires a clear protocol for addressing these discrepancies, especially when they involve features with potential business or ethical implications. The lack of consensus on `Segment_Effect` and `VetreansDay` on **DEV** needs further investigation to understand if these are genuine but subtle influences or artifacts of the explanation methods.

---
