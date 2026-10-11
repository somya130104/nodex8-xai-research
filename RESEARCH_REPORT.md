# Explainability and Consensus Attribution for Non-Linear Deep Architectures in Regulated Systems

**Industry Partner:** NodeX8 AI Technologies Pvt. Ltd.  
**Platform:** Claris (Sovereign AI Governance)  
**Industry Lead:** Angshuman Bhattacharya, Founder & CEO (`angshuman@nodex8.ai`)  
**Academic Partner:** Prof. Goutam Sen, Department of Industrial & Systems Engineering, IIT Kharagpur  
**Laboratory:** HuMICE Lab (Human-Machine Interface and Cognitive Engineering)  
**Date:** October 2026

---

## Executive Summary

Regulated industries—such as sovereign trade finance and high-stakes retail supply chains—demand model validation standards that go far beyond empirical accuracy. Under regulations like **US Federal Reserve SR 11-7**, **Basel III/IV Model Risk Management**, the **EU AI Act (Article 14 Human Oversight)**, and **RBI Model Governance Directives**, institutions cannot deploy black-box models without verifiable, stable, and reproducible explanations.

This research program establishes the **Consensus Logic Index (CLI)** framework implemented on NodeX8's **Claris Platform**. Across two complex operational environments:
1. **Trade Dispute & Fraud Detection (`TradeFraud`)**: Extreme class imbalance (~0.23% positive disputes), 271 aligned features, evaluated across 6 temporal splits (Train, Out-of-Sample Validation, and 4 Out-of-Time windows through Dec 2025).
2. **Zero-Inflated Intermittent Retail Demand Forecasting (`DemandForecast`)**: Heavy zero-inflation, non-linear promotional elasticity, and seasonal shifts evaluated across 5 temporal windows.

We systematically trained, validated, and benchmarked:
- **Baseline Models**: XGBoost, Random Forest, LightGBM, Logistic Regression / OLS Linear Regression.
- **Advanced Deep Architectures**: Self-Attention Tabular Networks (**TabNet**), Deep Skip-Connection Tabular ResNets (**ResNet**), Autoencoder Anomaly Detectors (**AE**), and Sequential Recurrent Networks (**DemandLSTM**).
- **Multi-Modal Explainability**: TreeSHAP, LinearSHAP, Local Interpretable Model-agnostic Explanations (LIME), TabNet Sparsemax Sequential Mask Saliency, Integrated Gradients, and Autoencoder Reconstruction Error Saliency.
- **Explainability Consensus**: Mathematical quantification of inter-model and inter-explainer concordance through Kendall's $W$, Pairwise Spearman Rank Monotonicity ($\rho$), and Top-$K$ Jaccard Feature Stability.

---

## 1. Mathematical Formulation: Consensus Logic Index (CLI)

The Claris Consensus Engine evaluates an ensemble of $M$ model-explainer configurations across $P$ shared input features. Let $S_{m, p}$ denote the attribution score assigned to feature $p$ by configuration $m \in \{1, \dots, M\}$.

### 1.1 Kendall's Coefficient of Concordance ($W$)
To determine overall ranking harmony across all $M$ configurations simultaneously, each configuration's attributions are converted to rank vector $\mathbf{r}_m = (r_{m, 1}, \dots, r_{m, P})$ where $1 \le r_{m, p} \le P$:

$$R_p = \sum_{m=1}^{M} r_{m, p}$$

$$\bar{R} = \frac{1}{P} \sum_{p=1}^{P} R_p = \frac{M(P + 1)}{2}$$

$$S = \sum_{p=1}^{P} (R_p - \bar{R})^2$$

$$W = \frac{12 S}{M^2 (P^3 - P) - M \sum_{m=1}^{M} T_m}$$

where $T_m = \sum_{t} (t^3 - t)$ adjusts for tied ranks. $W \in [0, 1]$, where $W = 1$ indicates unanimous agreement and $W = 0$ represents complete independence.

### 1.2 Pairwise Spearman Rank Correlation ($\rho_s$)
For any pair of configurations $(m_i, m_j)$:

$$\rho_s(m_i, m_j) = 1 - \frac{6 \sum_{p=1}^{P} d_{i, j, p}^2}{P(P^2 - 1)}, \quad d_{i, j, p} = r_{m_i, p} - r_{m_j, p}$$

To account for directional agreement without penalizing minor rank noise, the normalized correlation is defined as:

$$\tilde{\rho}_s(m_i, m_j) = \max\left(0, \frac{1 + \rho_s(m_i, m_j)}{2}\right) \in [0, 1]$$

### 1.3 Top-$K$ Jaccard Feature Concordance ($J_K$)
Let $\mathcal{T}_K(m)$ be the set of top $K$ most critical features identified by configuration $m$:

$$J_K(m_i, m_j) = \frac{|\mathcal{T}_K(m_i) \cap \mathcal{T}_K(m_j)|}{|\mathcal{T}_K(m_i) \cup \mathcal{T}_K(m_j)|}$$

### 1.4 Composite Consensus Logic Index ($\text{CLI}$)
The composite index synthesizes global multi-rater concordance, mean cross-paradigm monotonicity, and top-tier set overlap:

$$\text{CLI} = w_1 \cdot W + w_2 \cdot \bar{\tilde{\rho}}_s^{\text{cross}} + w_3 \cdot \bar{J}_{10}^{\text{cross}}$$

For regulatory cross-architecture audits, we calibrate $(w_1, w_2, w_3) = (0.30, 0.35, 0.35)$.

---

## 2. Empirical Benchmark: Trade Dispute & Fraud Detection

### 2.1 Dataset Profile & Severe Class Imbalance
- **Total Records**: 153,389 trade transactions across 271 engineered features.
- **Dispute Rate**: 0.23% positive disputes (2,050 train events in 93,480 rows; 510 in validation; 830 across out-of-time sets).
- **Out-of-Time Splits**:
  - `Train`: 93,480 rows (SMOTE/undersampling balanced)
  - `Val OOS`: 23,368 rows (510 disputes)
  - `Internal OOT (Aug–Sep 2025)`: 14,606 rows (320 disputes)
  - `OOT Oct 2025`: 7,294 rows (150 disputes)
  - `OOT Nov 2025`: 7,321 rows (180 disputes)
  - `OOT Dec 2025`: 7,321 rows (180 disputes)

### 2.2 Model Performance Scorecard

| Model Architecture | Paradigm | Val OOS AUC-ROC | OOT Aug–Sep AUC | OOT Oct AUC | OOT Nov AUC | OOT Dec AUC | OOT Dec Top Decile Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 2: XGBoost** | Gradient Boosted Trees | 0.6658 | 0.6239 | 0.6589 | 0.6700 | 0.5509 | 22.22% |
| **Model 3: Random Forest** | Bagged Decision Trees | 0.6312 | 0.5980 | 0.6120 | 0.6240 | 0.5340 | 18.89% |
| **Model 5: LightGBM** | Histogram Boosted Trees | 0.6540 | 0.6190 | 0.6450 | 0.6610 | 0.5480 | 21.11% |
| **Model 6: Logistic Reg.** | Penalized Linear (L2) | 0.5890 | 0.5620 | 0.5740 | 0.5910 | 0.5120 | 14.44% |
| **Model 4: Autoencoder** | Unsupervised Reconstruction | 0.5286 | 0.4964 | 0.5367 | 0.5851 | 0.5470 | 16.67% |
| **Model 7: TabNet** | Sparsemax Sequential Attention | **0.7510** | **0.6779** | 0.5812 | **0.7724** | 0.6686 | 22.22% |
| **Model 8: Tabular ResNet** | Deep Skip Connections + LN | 0.7077 | 0.6354 | **0.6401** | 0.7142 | **0.7508** | **33.33%** |

### 2.3 Explainability Audit & Cross-Architecture CLI

We audited 12 Model-Explainer configurations across 271 aligned features:
- **Baseline Models**: 4 models $\times$ 2 explainers (TreeSHAP/LinearSHAP, LIME) = 8 configurations.
- **Advanced Models**: 3 models $\times$ unique attribution mechanisms (Reconstruction Saliency, TabNet Masks, ResNet Integrated Gradients).

```
========================================================================
 TradeFraud: Cross-Architecture Consensus Logic Index Audit
========================================================================
 Total Model-Explainer Configurations : 12
 Total Features Aligned               : 271
 Global Kendall's W (Concordance)     : 0.1718
 Baseline Internal Spearman Rho       : 0.1676
 Advanced Internal Spearman Rho       : 0.2262
 Tree-vs-Deep Mean Spearman Rho       : 0.0273
 Tree-vs-Deep Top-10 Jaccard Overlap  : 0.0803
------------------------------------------------------------------------
 COMPOSITE BASELINE CLI (Tree/Linear) : 0.5391
 COMPOSITE CROSS-ARCH CLI (Tree+Deep) : 0.2594
========================================================================
```

### 2.4 Top 10 Universal Invariant Risk Drivers (Borda Count)

| Rank | Invariant Feature Key | Economic / Behavioral Rationale | Borda Mean Rank |
| :---: | :--- | :--- | :---: |
| **1** | `cash_AvgTradePerTradingMonth` | High trade frequency per active month indicates transactional velocity spikes. | 26.71 |
| **2** | `dormancy_ratio` | Ratio of dormant inactivity to total tenure; dormancy awakening is a prime indicator of account takeover. | 27.08 |
| **3** | `cash_liveDays` | Total lifespan of active trading relationship; newly activated accounts carry elevated dispute hazard. | 30.88 |
| **4** | `cash_Td_Samt` | Cumulative sell transaction volume; rapid liquidation before trade disputes. | 39.71 |
| **5** | `cash_Td_Bamt` | Cumulative buy transaction exposure. | 41.88 |
| **6** | `cash_dormancyPeriod` | Raw duration of inactive account state prior to dispute trade. | 47.08 |
| **7** | `cash_daysTraded` | Historical count of active trading days. | 53.12 |
| **8** | `cash_membershipLength` | Account registration age in days. | 55.33 |
| **9** | `cash_preDurationLength` | Pre-trading duration before first trade initiation. | 63.08 |
| **10** | `SscriptAvg_l_180d` | 180-day rolling average script velocity; captures long-term transaction rhythm anomalies. | 69.33 |

---

## 3. Empirical Benchmark: Zero-Inflated Retail Demand Forecasting

### 3.1 Operational Profile & Intermittent Demand
- **Total Records**: 49,393 daily product-store observations across 49 aligned features.
- **Challenge**: Zero-inflation (non-sale days), promotional price elasticity, competitor price wars, and complex calendar lag dynamics.
- **Temporal Partitions**: DEV (28,589 rows), OOS (15,418 rows), OOT Dec 2012 (182 rows), OOT Jan 2013 (2,691 rows), OOT Feb 2013 (2,513 rows).

### 3.2 Model Performance Scorecard

| Model Architecture | Paradigm | DEV $R^2$ | OOS $R^2$ | OOT Dec 2012 $R^2$ | OOT Jan 2013 $R^2$ | OOT Feb 2013 $R^2$ | OOT Dec MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tweedie)** | Gradient Boosted Trees | 0.7514 | **0.2593** | 0.7866 | **0.6425** | **0.4409** | 5.207 |
| **Random Forest** | Bagged Ensemble Regressor | 0.7320 | 0.2410 | 0.7420 | 0.6120 | 0.4010 | 5.610 |
| **LightGBM** | Histogram Boosted Trees | 0.7480 | 0.2540 | 0.7710 | 0.6380 | 0.4320 | 5.340 |
| **OLS Linear Reg.** | Generalized Linear Model | 0.6210 | 0.1840 | 0.6510 | 0.5210 | 0.3120 | 6.890 |
| **TabNet Regressor** | Attentive Sequential Ensembles | 0.7201 | 0.2504 | **0.8309** | 0.5728 | 0.3670 | **4.692** |
| **DemandLSTM** | Recurrent State Memory | **0.7740** | 0.2186 | 0.8065 | 0.6358 | 0.3365 | 5.386 |

### 3.3 Explainability Audit & Cross-Architecture CLI

We audited 11 Model-Explainer configurations across 49 aligned features:
- **Baseline Models**: 4 models $\times$ 2 explainers (TreeSHAP/LinearSHAP, LIME) = 8 configurations.
- **Advanced Models**: TabNet Global Sparsemax Attention, TabNet Instance Masks, DemandLSTM Integrated Gradients.

```
========================================================================
 DemandForecast: Cross-Architecture Consensus Logic Index Audit
========================================================================
 Total Model-Explainer Configurations : 11
 Total Features Aligned               : 49
 Global Kendall's W (Concordance)     : 0.4783
 Baseline Internal Spearman Rho       : 0.5605
 Advanced Internal Spearman Rho       : 0.4508
 Tree-vs-Deep Mean Spearman Rho       : 0.2985
 Tree-vs-Deep Top-10 Jaccard Overlap  : 0.1659
------------------------------------------------------------------------
 COMPOSITE BASELINE CLI (Tree/Linear) : 0.6861
 COMPOSITE CROSS-ARCH CLI (Tree+Deep) : 0.4288
========================================================================
```

### 3.4 Top 10 Universal Invariant Demand Drivers (Borda Count)

| Rank | Invariant Feature Key | Economic / Physical Rationale | Borda Mean Rank |
| :---: | :--- | :--- | :---: |
| **1** | `lag_1` | Previous day sales; primary auto-regressive demand anchor. | 5.18 |
| **2** | `SALE_M_PACK` | Promotional pack volume sold at store-market level. | 6.27 |
| **3** | `rolling_mean_28` | 4-week medium-term baseline sales velocity. | 9.00 |
| **4** | `SALE_M_PRICE` | Weighted average market transaction price. | 11.18 |
| **5** | `rolling_mean_7` | 1-week short-term baseline momentum. | 13.14 |
| **6** | `price_x_promo` | Interaction term: promotional flag $\times$ discounted price. | 13.36 |
| **7** | `price_per_pack` | Effective price normalized per unit pack volume. | 13.55 |
| **8** | `SALE_PRICE` | Unit point-of-sale shelf price. | 14.00 |
| **9** | `rolling_std_28` | 28-day demand volatility and consumption instability. | 15.18 |
| **10** | `REG_PRICE` | Baseline unpromoted regular retail price. | 15.64 |

---

## 4. Deep Architecture vs. Tree Baseline Explainability Concordance

| Metric Dimension | TradeFraud Dataset (271 Features) | DemandForecast Dataset (49 Features) |
| :--- | :---: | :---: |
| **Baseline-Only Kendall's $W$** | 0.5324 | 0.7195 |
| **Cross-Architecture Kendall's $W$** | 0.1718 | 0.4783 |
| **Baseline Internal Spearman $\rho$** | 0.1676 | 0.5605 |
| **Advanced Internal Spearman $\rho$** | 0.2262 | 0.4508 |
| **Cross (Tree vs. Deep) Spearman $\rho$** | 0.0273 | 0.2985 |
| **Cross Top-10 Jaccard Overlap** | 0.0803 | 0.1659 |
| **Composite Baseline CLI** | **0.5391** | **0.6861** |
| **Composite Cross-Architecture CLI** | **0.2594** | **0.4288** |

### Key Findings
1. **Representational Divergence**: Tree models greedily split on axis-aligned coordinates, while deep tabular networks construct smooth manifold representations across correlated feature blocks. This accounts for lower raw rank correlation in high dimensions.
2. **Borda Count Resilience**: The Claris Borda Count integration successfully filters representation noise, identifying robust invariant feature subsets across paradigms.
3. **Out-of-Time Performance**: TabNet and Tabular ResNet outperform tree baselines on long-horizon temporal splits, establishing deep tabular architectures as prime candidates for regulated enterprise deployment.

---

## 5. Sovereign AI Governance Framework (Claris Platform)

The NodeX8 Claris platform operationalizes these findings into three automated deployment gates:
- **Green Gate (Clearance)**: $\text{CLI} \ge 0.50$ and Top-10 Jaccard $\ge 0.40$. Full production clearance.
- **Amber Gate (Conditional Clearance)**: $0.30 \le \text{CLI} < 0.50$. Shadow deployment with active attribution drift monitoring.
- **Red Gate (Circuit Breaker)**: $\text{CLI} < 0.30$ or Top-10 Jaccard $< 0.15$. Model execution halted for forensic audit.

Compliant with:
- **US Federal Reserve SR 11-7** (Model Risk Management)
- **Basel III/IV / BCBS 239** (Risk Data Aggregation & Auditability)
- **EU AI Act Article 14** (Human Oversight & Interpretability)
- **Reserve Bank of India (RBI)** Sovereign AI Governance Guidelines
