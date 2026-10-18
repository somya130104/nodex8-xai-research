# ARCHITECTURE — Decisions Taken So Far

**Project:** Explainability and Consensus Attribution for Non-Linear Deep Architectures in Regulated Financial Systems
**Partners:** NodeX8 AI Technologies Pvt. Ltd. × IIT Kharagpur (HuMICE Lab, Dept. of Industrial & Systems Engineering)
**Status:** Baseline stage. The core research methods are not built yet — see [What Is Deliberately Not Here Yet](#6-what-is-deliberately-not-here-yet).

This document records **the decisions we have actually made and can defend**, and why we made them.
It is the honest record of the project. If a number or a claim is not in this file, it is not something
we are standing behind yet.

A companion document, [FLOW.md](FLOW.md), explains how to read and run the code.

---

## 1. What the project is trying to do

Banks increasingly use complex models (autoencoders, attention networks) for fraud detection and
forecasting. Regulators — notably the **RBI 2026 Model Risk Management draft guidance** and the
**FREE-AI framework** — require that such models not be unexplainable black boxes.

The problem is that the standard explainability tools break on these models:

1. **Autoencoders produce no probability.** They output a single reconstruction-error number
   ("how unusual is this?"). Standard KernelSHAP has to call the model thousands of times to explain
   that, which is far too slow for a real-time banking pipeline.
2. **LIME invents impossible customers.** It explains a prediction by randomly nudging the input.
   But financial features are correlated — a client's 30-day trade volume cannot exceed their 365-day
   volume. Ignoring that produces explanations built on data that could never exist.
3. **Explainers contradict each other** (the "Rashomon effect"). SHAP may say dormancy drove a
   fraud flag while LIME says trade velocity did. In an audit you cannot present two different stories.

The research proposal asks us to build three explainers that work on these models
(**Adapted SHAP, Manifold-LIME, and Latent Reconstruction Decomposition / LRD**), and then a
**Consensus Logic Index (CLI)** that measures, for one single transaction, how much those three
explainers agree — flagging the transaction for human review when they do not.

**None of those four things are built yet.** Everything in this repository today is the groundwork:
clean data splits, baseline models to compare against, and conventional SHAP/LIME.

---

## 2. The three datasets

| | **TradeFraud** | **DemandForecast** | **SalesForecasting** |
|---|---|---|---|
| **Problem type** | Binary classification | Regression | Regression |
| **Target** | `Dispute` (`"D"` / `"N"`) | `QUANTITY` (units sold) | `UnitSales` |
| **Unit of a row** | One client (snapshot) | One product × one day | One customer × one week |
| **Size** | 150,339 clients, 271 features | 49,393 rows, 131 products, 1 store | 56,144 rows, 5 segments |
| **Period** | Apr 2024 – Dec 2025 | Jul 2011 – Feb 2013 | 2023 onward |
| **Main difficulty** | Extreme class imbalance (~0.22% disputes) | ~20% of days have zero sales | Weekly panel structure |
| **Role here** | **Priority 1** — the autoencoder lives here, and the autoencoder is the main LRD target | Priority 2 | **Reference template only** — we do not modify it |

**Decision: TradeFraud first.** It is the cleaner dataset, it has all the time splits already defined,
and the autoencoder family (AE/VAE) is the primary target of the LRD method, which is the
project's novel contribution.

**Decision: SalesForecasting is the template, not a work item.** It is a pre-existing, fully built
pipeline (feature engineering → models → a manifest-driven XAI pipeline producing global, decile
and segment explanations → a validation package with hash-chained audit logs). We studied it and
are mirroring its conventions. We do not change its files.

---

## 3. Data splitting decisions

The rule we follow everywhere: **never let information from the future, or from the same customer,
leak into the training set.** A model that has seen the answer will look brilliant and then fail in
production.

### TradeFraud

| Split | Rows | Dispute rows | **Unique disputes** | Period |
|---|---:|---:|---:|---|
| `train.csv` | 93,480 | 2,050 | **205** | Apr 2024 – Jul 2025 |
| `val_oos.csv` | 23,368 | 510 | **51** | Apr 2024 – Jul 2025 |
| `internal_oot.csv` | 14,606 | 320 | **32** | Aug – Sep 2025 |
| `oot_202510.csv` | 7,294 | 150 | **15** | Oct 2025 |
| `oot_202511.csv` | 7,321 | 180 | **18** | Nov 2025 |
| `oot_202512.csv` | 7,321 | 180 | **18** | Dec 2025 |

- **OOS (Out-of-Sample)** = *different clients, same period.* We verified there is **zero client
  overlap** between train and validation.
- **OOT (Out-of-Time)** = *future months the model has never seen.* This is the honest test of
  whether a model survives in production as fraud patterns change.

> **Important caveat we must flag to NodeX8.** In the data as supplied, **each dispute row is
> duplicated 10 times**. The "2,050 disputes" in training are 205 distinct clients repeated ten times
> each. This matters a great deal: with only 15–18 *real* disputes in each out-of-time month, the
> confidence interval around any AUC figure is roughly ±0.13. Most apparent differences between
> models on those months are statistical noise, not real differences. It also explains why the tree
> models report a training AUC of 1.000 — they are memorising repeated rows.
> **We have asked Angshuman whether this duplication is intentional.** Until we hear back, every
> OOT number in this repository should be read as indicative only.

### DemandForecast

| Split | Rows | Period |
|---|---:|---|
| `dev.csv` | 28,589 | up to Oct 2012 |
| `oos.csv` | 15,418 | same period, held-out brands |
| `oot_01_201212.csv` | 182 | Dec 2012 |
| `oot_02_201301.csv` | 2,691 | Jan 2013 |
| `oot_03_201302.csv` | 2,513 | Feb 2013 |

- OOS is split by **entity (brand × store)**, so the model is tested on products it has not seen.
- OOT is the final calendar months.

> **Two caveats.** (a) `oot_01` covers only **two days** (182 rows), because the cut-off was set as
> "last 60 days" rather than on calendar-month boundaries — any metric on it is unreliable.
> (b) There is only **one store** and **ten brands** in this data, so a 70/30 split by brand means
> OOS rests on just 3 brands. Both are on the fix list.

### Leakage check we ran and passed

The lag and rolling features in DemandForecast (`lag_1`, `rolling_mean_7`, `rolling_mean_28`, …) were
verified to use **only past days** — they match a shifted rolling window, not one that includes the
current day. So there is no look-ahead leakage in the features themselves. This was worth checking:
a rolling average that accidentally includes today's sales would make the model look near-perfect
and be worthless in production.

---

## 4. Modelling decisions

### Why these baseline models

Angshuman's instruction was explicit: *"You can use existing models… the actual work is the
explainability of the advanced models. I'm ok with few baseline models,"* and *"each data set has
3–4 baseline models before the advanced models."*

So we deliberately built **four solid baselines per dataset and stopped tuning**. Their job is not to
win an accuracy contest. Their job is to be a trustworthy point of comparison for the explainability
work, and to prove the data pipeline is sound.

**TradeFraud (classification)** — each uses a *different* strategy for the class imbalance, on purpose,
so that we can later ask whether the imbalance strategy changes the explanation:

| Model | Imbalance strategy | Why included |
|---|---|---|
| XGBoost | `scale_pos_weight=44.6` | Industry-standard gradient boosting; exact TreeSHAP available |
| Random Forest | SMOTE (ratio 0.2) | Different ensemble family (bagging, not boosting) |
| LightGBM | `is_unbalance=True` | Faster boosting; cross-check against XGBoost |
| Logistic Regression | `class_weight='balanced'`, elastic-net | Transparent linear baseline; needed as the reference for LIME |

**DemandForecast (regression):** XGBoost (Tweedie), Random Forest, LightGBM, OLS. **Tweedie loss**
was chosen specifically because ~20% of days have zero sales, and Tweedie is designed for
zero-inflated non-negative data where ordinary squared error does badly.

### Why these metrics

**Decision: for TradeFraud we report AUC-ROC, AUC-PR, Gini, KS and Top-Decile Recall — never accuracy.**
With a 0.22% event rate, a model that predicts "no fraud" for every single client is 99.78% accurate
and completely useless. Top-Decile Recall is the one that matters operationally: *if an investigator
reviews only the riskiest 10% of accounts, what share of all disputes do they catch?*

For DemandForecast: MAE, RMSE, MAPE and R².

### Current baseline results (preliminary)

TradeFraud, AUC-ROC:

| Model | Validation (OOS) | OOT Dec 2025 |
|---|---:|---:|
| Logistic Regression | 0.716 | 0.586 |
| Random Forest | 0.686 | 0.607 |
| XGBoost | 0.666 | 0.551 |
| LightGBM | 0.665 | 0.565 |
| Autoencoder | 0.529 | 0.547 |
| TabNet* | 0.751 | 0.669 |

DemandForecast, R²:

| Model | DEV | OOS |
|---|---:|---:|
| Random Forest | 0.834 | 0.226 |
| LightGBM | 0.823 | 0.235 |
| XGBoost | 0.751 | 0.259 |
| LSTM* | 0.774 | 0.219 |
| TabNet* | 0.720 | 0.250 |
| OLS | 0.626 | 0.244 |

\* **Not yet comparable.** TabNet and the LSTM were early-stopped using the validation set they are
then scored on, which flatters them. They will be retrained with a validation slice taken from
training data only. Until then, do not use these to argue that deep models beat the baselines.

**We are not claiming any model is better than another.** Given the duplicated disputes, the tiny
real sample sizes, and the early-stopping issue above, that claim is not currently supportable. All
numbers are read directly from the `performance_scorecard.csv` files.

### Why the autoencoder

The autoencoder is trained **only on non-dispute clients**. It learns what "normal" looks like and
then scores how badly it fails to reconstruct a new client — high reconstruction error means unusual.

**Decision: we keep it even though its AUC (~0.53) is weak.** This is deliberate, and worth stating
clearly because it will look odd otherwise:

- It is the architecture family the proposal names first, and it is the primary target for LRD.
- The research question is *"can we explain this model's reasoning quickly and faithfully?"* — an
  explainer is judged on whether it faithfully describes what the model did, not on whether the
  model is accurate.
- That said, we know *why* it is weak: it is trained on raw standardised values of heavily
  skewed financial amounts, so a handful of large-valued columns dominate the reconstruction error.
  Log-transforming the amount columns is on the fix list.

---

## 5. Explainability decisions so far

Angshuman's requirement: *"For each model use SHAP and LIME for first level explainability. Our
target is to produce global, local, and at decile/cohort level (segments in data)."*

**Decision: match the explainer to the model type.** Using the slow general-purpose explainer when
an exact fast one exists would be wasteful and less accurate:

- **TreeSHAP** for XGBoost / Random Forest / LightGBM — exact and fast for tree models.
- **LinearSHAP** for Logistic Regression and OLS.

**Decision: three levels of aggregation**, as required:

| Level | What it answers | Status |
|---|---|---|
| **Global** | Across everyone, which features matter most on average? | Done (both datasets) |
| **Decile / Cohort** | Within each risk decile and each customer segment, which features matter? | Done for TradeFraud; pending for DemandForecast |
| **Local** | For *this one* client, why were they flagged? (the regulatory adverse-action notice) | **Not yet done** |

For TradeFraud the cohorts are the segment fields that come with the data: `Segment`
(Gentry, Heartlanders, Novices, …), `Activeness_Segment` (Dormant … Power Trader), and risk decile.

---

## 6. What is deliberately NOT here yet

This section exists so that nobody — including us — mistakes groundwork for the finished research.

| Required by the proposal | Status |
|---|---|
| **Adapted SHAP** (DeepSHAP / Integrated Gradients on the autoencoder's reconstruction error, with a study of how the choice of background reference changes the answer) | Not built |
| **Manifold-LIME** (perturbations constrained by the Mahalanobis covariance so they stay on the real financial data manifold) | Not built |
| **LRD — Latent Reconstruction Decomposition** (back-projecting reconstruction error through the bottleneck to input features, target < 50 ms) | Not built |
| **CLI(x)** — per-transaction consensus using **weighted Kendall's τ**, per the proposal's Equation (1) | Not built |
| **Uncertainty gate** — a calibrated τ threshold that routes low-agreement cases to a human | Not built |
| **VAE** (probabilistic latent space; richer LRD target) | Not built |
| **Local / per-instance explanations** | Not built |
| **Faithfulness, stability and latency benchmarks** | Not built |
| **GNN family** | Blocked — the data has no client-to-client links (see below) |

### Things that were previously in this repository and have been removed

An earlier working session produced several artefacts that went beyond the proposal and, in places,
conflicted with it. They were removed in the cleanup commit. **Nothing is lost — the full previous
state is preserved in git history at commit `769b51b`** and can be recovered at any time.

| Removed | Why |
|---|---|
| `run_cross_architecture_cli.py`, `Consensus_Outputs/` | Measured whether *different models* rank features similarly. That is a different question from the proposal's CLI (do *three explainers* agree about *one transaction*?). Keeping the "CLI" name made it look as though the core deliverable was finished when it had not been started. |
| The composite formula `CLI = 0.30·W + 0.35·ρ + 0.35·J` in `consensus_engine.py` | Not in the proposal. The proposal specifies one formula — weighted Kendall's τ. The three weights were never derived or calibrated. |
| `train_model8_resnet.py`, `Model8_ResNet/` | Tabular ResNet is not one of the four architecture families in the proposal (AE/VAE, TabNet/FT-Transformer, GNN, LSTM/temporal). |
| Borda Count "universal invariant drivers" | Not requested, and contaminated by a LIME bug on DemandForecast (see limitations). |
| `RESEARCH_REPORT.md` | Contained performance figures that did not match our own scorecard CSVs, uncalibrated Green/Amber/Red deployment gates, and a regulatory-compliance table asserting conformance with SR 11-7, Basel III/IV, the EU AI Act and RBI. None of that was substantiated. It will be rewritten, generated directly from the result files. |
| Autoencoder "Reconstruction Error Saliency" in `run_advanced_xai.py` | Described as a gradient-based attribution, but it actually computed the *average error gap between dispute and non-dispute clients* — which uses the labels and produces one global number. That is not an explanation of any individual decision. |
| Output paths pointing into `~/.gemini/antigravity-ide/brain/…` | An AI tool's internal scratch directory had been written into project source, so scripts would not run on anyone else's machine. |
| `Model_Outputs/requirements.txt` | A raw `pip freeze` of an entire Jupyter environment (190 packages including `openai` and `google-generativeai`), in the wrong folder. A curated dependency list will replace it. |

What was **kept** from that work: TabNet's native attention masks, now in
`extract_tabnet_masks.py` and clearly labelled as a *native model signal* rather than one of the
proposal's three explainers. The proposal does ask us to compare the Tri-Explainer against TabNet's
built-in masks, so this is a legitimate reference point.

---

## 7. Known limitations (open items)

Recorded honestly so they are fixed deliberately rather than discovered by a reviewer.

1. **Duplicated disputes (TradeFraud).** 10× duplication of positive rows. Awaiting confirmation
   from NodeX8; our default will be to train and evaluate on unique rows with class weights.
2. **LIME feature-mapping bug (DemandForecast).** LIME returns conditions as text
   (`"lag_1 <= 0.00"`), and the current code matches those by substring. Because `lag_1` is a
   substring of `lag_14`, and `SALE_PRICE` of `C_SALE_PRICE`, weights are credited to the wrong
   feature. **Fifteen features have a LIME value of exactly zero for every model**, including every
   competitor (`C_*`) feature. TradeFraud is unaffected (no feature name is a substring of another).
3. **The DemandForecast "LSTM" is not temporal.** It is fed sequences of length 1, which makes it a
   plain feed-forward network. A genuine 28-day sequence model is needed for the proposal's
   temporal architecture family. (SalesForecasting's LSTM already does this properly and can be
   adapted.)
4. **TabNet / LSTM early-stopped on the evaluation set.** Their validation scores are optimistic.
5. **Decision thresholds are re-tuned on every test set,** including out-of-time sets. This makes
   precision/recall/F1 look better than they would in production. One threshold, fixed on
   validation, must be applied everywhere.
6. **No confidence intervals.** Essential given 15–18 real disputes per OOT month.
7. **Autoencoder preprocessing.** Heavy-tailed amount columns are not log-transformed, so a few
   columns dominate the reconstruction error.
8. **`oot_01_201212` is 2 days long.**
9. **Hosmer-Lemeshow calibration charts exist but should not be interpreted.** Every model
   deliberately distorts its probabilities (`scale_pos_weight`, SMOTE, `class_weight='balanced'`), and
   the autoencoder's "probability" is a min-max-scaled reconstruction error, so a calibration test
   on it is not meaningful.
10. **Attribution outputs are not version-controlled.** `.gitignore` excludes `*.csv`, so
    `shap_global.csv` and friends exist only locally. They are our evidence and should be committed.
11. **No automated tests, and no pinned dependency list.** The proposal requires ≥85% test coverage
    for the final SDK.
12. **Model numbering skips Model 1** (models run 2–7), a leftover from an earlier numbering scheme.

---

## 8. Open questions for NodeX8

1. **Duplicated disputes** — is the 10× repetition intentional oversampling, or should we train and
   evaluate on unique rows?
2. **LRD specification** — the proposal describes LRD as NodeX8's proprietary framework but gives
   only a one-line description. Does NodeX8 have an existing written specification or pseudocode we
   should implement, or is designing the precise algorithm part of our research contribution?
3. **GNN family** — the proposal asks for a graph model over account networks (mule accounts, fraud
   rings), but the TradeFraud data has one independent row per client and no columns linking clients
   to each other. There is nothing to build a graph from. Should we defer this, use a public graph
   dataset (e.g. Elliptic) to demonstrate the method, or can NodeX8 share link data such as shared
   devices or counterparties?
4. **Publication data** — partner data is unlikely to be publishable. Should we additionally run on
   the public benchmarks named in the proposal's Work Package 1 (IEEE-CIS, Kaggle Credit)?

---

## 9. Where the work goes next

In order. Nothing later should start before the earlier items, because every consensus number
depends on clean inputs.

| Phase | Work |
|---|---|
| **0. Fix and freeze** | Pin dependencies; resolve the dispute duplication; stop early-stopping on evaluation sets; fix one threshold per model; add bootstrap confidence intervals; fix the LIME substring bug; fix the DemandForecast OOT months. |
| **1. Finish baseline explainability** | Local (per-instance) SHAP and LIME for high-risk, borderline and misclassified cases; DemandForecast cohort level; per-instance τ between SHAP and LIME as the *baseline* consensus. |
| **2. Advanced models** | Rebuild the autoencoder with proper preprocessing; add the VAE; retrain TabNet cleanly; build a genuine sequence LSTM. |
| **3. The Tri-Explainer** | Adapted SHAP (with background-sensitivity study), Manifold-LIME, LRD — plus latency measurement and a synthetic ground-truth test. |
| **4. CLI** | Weighted Kendall's τ, per-instance CLI per the proposal's Equation (1), aggregation by decile/segment/time, and calibration of the uncertainty gate. |
| **5. Benchmarks** | Faithfulness, monotonicity, stability, latency, memory. |
| **6. SDK** | `claris_deep_xai`, PEP-8, ≥85% test coverage. |
| **7. Paper & handover** | Report regenerated from result files; manuscript; final presentation. |
