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
> models on those months are statistical noise, not real differences.
> **We have asked Angshuman whether this duplication is intentional.** Until we hear back, our
> default (applied throughout this repository as of Phase 0) is to **drop the duplicates** before
> training or evaluating anything: every model now trains and is scored on the 205/51/32/15/18/18
> unique clients per split, with class weighting used instead to handle the imbalance. This is why a
> training AUC of 1.000 (seen in the original, undeduplicated runs — the tree models were simply
> memorising repeated rows) is no longer observed.

### DemandForecast

| Split | Rows | Period |
|---|---:|---|
| `dev.csv` | 26,803 | up to Nov 2012 |
| `oos.csv` | 14,461 | same period, held-out brands |
| `oot_01_201212.csv` | 2,925 | Dec 2012 (full month) |
| `oot_02_201301.csv` | 2,691 | Jan 2013 (full month) |
| `oot_03_201302.csv` | 2,513 | Feb 2013 (through the 27th — the dataset ends there) |

- OOS is split by **entity (brand × store)**, so the model is tested on products it has not seen.
- OOT is the last three full calendar months in the data.

> **Fixed in Phase 0.** The OOT cutoff used to be "max date − 60 days", which landed mid-month and
> left `oot_01` covering only the last 2 days of December. It is now the start of the last three
> full calendar months, so `oot_01` is a genuine full month (2,925 rows). All DemandForecast models
> were retrained on the corrected splits; the numbers in Section 4 reflect this.
>
> **Still open:** there is only **one store** and **ten brands** in this data, so a 70/30 split by
> brand means OOS rests on just 3 brands. Splitting OOS by product (131 UPCs) instead would give a
> far more stable estimate — this remains on the fix list for Phase 1.

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

### Current baseline results (post Phase 0 fixes)

All numbers below are read directly from `performance_scorecard.csv` in each model's output folder,
after: deduplication, a decision threshold chosen once on validation and reused unchanged everywhere
(not re-optimised per dataset), early stopping moved onto an internal split of training data only
(never the evaluation set), and — for TradeFraud — a bootstrap 95% confidence interval on every AUC.

TradeFraud, AUC-ROC [95% CI]:

| Model | Validation (OOS) | OOT Dec 2025 |
|---|---:|---:|
| Logistic Regression | 0.716 [0.63, 0.80] | 0.586 [0.43, 0.73] |
| XGBoost | 0.715 [0.62, 0.81] | 0.620 [0.46, 0.77] |
| Random Forest | 0.711 [0.62, 0.80] | 0.604 [0.45, 0.74] |
| TabNet | 0.677 [0.59, 0.77] | 0.672 [0.51, 0.82] |
| LightGBM | 0.575 [0.47, 0.68] | 0.548 [0.39, 0.71] |
| Autoencoder | 0.537 [0.46, 0.61] | 0.556 [0.42, 0.69] |

Every confidence interval above is roughly 0.15–0.25 AUC points wide — exactly the effect of the
15–18 real out-of-time disputes described above. **Read the point estimates with that in mind:**
LightGBM's drop after deduplication (previously 0.665/0.565, now 0.575/0.548) is a real consequence
of early stopping on an internal split with only ~20 positive examples, not a bug — it is itself a
demonstration of the small-sample problem this fix exists to surface. No model's OOT Dec 2025
confidence interval is clearly separated from any other's; none of these differences should be
presented as "Model X beats Model Y" without saying so.

DemandForecast, R² (retrained on the corrected calendar-month OOT splits, Section 3):

| Model | DEV | OOS | OOT Dec 2012 (full month) |
|---|---:|---:|---:|
| Random Forest | 0.836 | 0.220 | — |
| LSTM | 0.773 | 0.264 | 0.507 |
| LightGBM | 0.785 | 0.227 | — |
| XGBoost | 0.767 | 0.224 | — |
| TabNet | 0.694 | 0.093 | 0.551 |
| OLS | 0.627 | 0.228 | — |

**We are not claiming any model is better than another.** OOS R² is low and similar for every model
(0.09–0.26), which — as before — looks like a structural effect of only 3 brands being held out,
not a model failure. TabNet's distinctly lower OOS R² (0.093) is worth a closer look in Phase 1
rather than being read as "TabNet is worse": it may be more sensitive to the small number of
held-out brands than the tree models are.

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

**Decision: three levels of aggregation**, as required — **all three are now done for both
datasets** (Phase 1, see ROADMAP.md):

| Level | What it answers | Status |
|---|---|---|
| **Global** | Across everyone, which features matter most on average? | ✅ Done (both datasets) |
| **Decile / Cohort** | Within each risk decile / segment / brand / day-type, which features matter? | ✅ Done (both datasets) |
| **Local** | For *this one* client, why were they flagged? (the regulatory adverse-action notice) | ✅ Done (both datasets) |

For TradeFraud the cohorts are the segment fields that come with the data: `Segment`
(Gentry, Heartlanders, Novices, …), `Activeness_Segment` (Dormant … Power Trader), and risk decile.
For DemandForecast, which has no equivalent segment field, the cohorts are **Brand**, **DayType**
(one mutually-exclusive bucket per row, in priority order Holiday > Promotion > Weekend > Weekday —
see `data/DemandForecast/scripts/cohort_utils.py`), and predicted-demand decile.

### Local (per-instance) explanations — Phase 1

For each of the 4 baseline models per dataset, **15 representative instances** are picked from
Validation OOS and explained with both SHAP and LIME on the exact same 15 cases, so the two can be
compared instance-by-instance (`run_local_explanations.py` in each dataset's `scripts/`):

- **TradeFraud** (classification): 5 highest-risk, 5 borderline (closest to that model's own fixed
  `Threshold_Used` from Phase 0 — not 0.5), 5 misclassified (a mix of false positives and false
  negatives, since these are more informative than confidently-correct cases).
- **DemandForecast** (regression, which has no decision threshold or misclassification concept):
  5 highest-predicted-demand, 5 borderline (closest to the median prediction), 5 highest-error
  (largest |actual − predicted|) — see `src/local_explain_utils.py` for the exact selection logic and
  why each bucket is regression's honest equivalent of the classification one.

Each instance also gets a short, templated plain-English sentence generated from its top-3 SHAP
features (`local_explanations.csv`) — this is the actual adverse-action-notice text, e.g.:
*"This case was classified as Dispute (predicted probability 0.585 against a decision threshold of
0.325), primarily because: (1) `cash_AvgTradePerTradingMonth` increased the predicted risk; (2)
`cash_AvgTradePerTradingDay` increased the predicted risk; (3) `cash_Td_Samt` increased the predicted
risk."*

**A same-model, same-instance SHAP-vs-LIME agreement number** is also computed for each of the 15
cases (`baseline_instance_cli.csv`, via `src/eval_utils.baseline_instance_tau`). This is explicitly
a **placeholder, not the proposal's CLI** — it uses `scipy.stats.weightedtau` rather than a
from-scratch, unit-tested weighted Kendall's τ, and compares only one pair of explainers (SHAP, LIME)
rather than the three the proposal specifies (Adapted SHAP, Manifold-LIME, LRD). It exists only as an
early rehearsal; Phase 4 builds the real thing. Observed baseline agreement: TradeFraud's four models
average τ ≈ 0.25–0.28 across the three case types; DemandForecast's average τ ≈ 0.46–0.53 — consistent
with the global-level finding (Section 4) that DemandForecast's explainers agree with each other more
than TradeFraud's do.

### `feature_hierarchy.csv` — Phase 1

Both datasets now have a `feature_hierarchy.csv` in `XAI_Outputs/`, mapping every model feature to a
Sub-Group → Group with equal weights within each, in the same format as SalesForecasting's reference
file. This is what will let later work report "Promotional Activity: 31%" instead of 51 or 271
individual feature rows:
- **TradeFraud**: built directly from `Data/variable_dictionary.csv`'s existing Group/Sub_Group
  columns (`build_feature_hierarchy.py`) — all 271 model features mapped, no manual categorisation
  needed.
- **DemandForecast**: no such reference file exists, so the mapping is hand-built from the feature
  names (`build_feature_hierarchy.py`) into 5 groups (Pricing, Competitor, Promotion, Calendar &
  Seasonality, Demand History) — all 51 model features mapped.

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

### Fixed in Phase 0

1. ~~**Duplicated disputes (TradeFraud).**~~ **Fixed.** Every TradeFraud training, SHAP, LIME and
   TabNet-mask script now drops the 10× duplicated rows before touching the data
   (`src/eval_utils.dedupe_rows`), and trains/evaluates on the 205/51/32/15/18/18 unique clients per
   split. **Still awaiting confirmation from NodeX8** that this is the right call rather than
   intentional oversampling — see Section 8.
2. ~~**LIME feature-mapping bug (DemandForecast).**~~ **Fixed.** The substring-matching approach
   (`if f in feat_name_exp`) is replaced with `exp.as_map()`, which gives LIME's own
   (feature_index, weight) pairs with no text parsing. Re-running confirmed the fix: of the 15
   features that used to read exactly zero for every model, 13 now have real values; the remaining
   2 (`REG_M_PRICE`, `count`) are genuinely constant columns in this sample, so zero is correct for
   them. The same fix was applied to TradeFraud's LIME script too, even though no TradeFraud feature
   name happened to collide this way.
3. ~~**`oot_01_201212` is 2 days long.**~~ **Fixed.** `create_samples.py` now cuts OOT on a
   calendar-month boundary instead of "max date − 60 days"; `oot_01` is a full December 2012
   (2,925 rows). All DemandForecast models were retrained on the corrected splits.
4. ~~**TabNet early-stopped on the evaluation set (TradeFraud).**~~ **Fixed.** TradeFraud's
   `train_model7_tabnet.py` now carves its early-stopping split from training data only, the same
   pattern XGBoost and LightGBM already used. (DemandForecast's TabNet and LSTM scripts were checked
   and turned out to already do this correctly — they were never affected.)
5. ~~**Decision thresholds re-tuned on every test set.**~~ **Fixed for TradeFraud.** Every TradeFraud
   classification/anomaly script now chooses one F1-maximising threshold on Validation OOS only
   (`select_f1_threshold`) and reuses it unchanged on every other split (`metrics_at_threshold`).
6. ~~**No confidence intervals.**~~ **Fixed for TradeFraud.** Every AUC-ROC in a TradeFraud scorecard
   now ships with a bootstrap 95% CI (`bootstrap_auc_ci`, 1,000 resamples). The CIs are wide — see
   Section 4 — which is the point: it makes the real uncertainty impossible to miss.
7. ~~**No pinned dependency list.**~~ **Fixed.** `requirements.txt` at the repo root pins every
   package to the version verified working together (see its header comment for a macOS-specific
   `libomp` / `scipy` note we hit and resolved while doing this work).

### Still open

8. **The DemandForecast "LSTM" is not temporal.** It is fed sequences of length 1, which makes it a
   plain feed-forward network. A genuine 28-day sequence model is needed for the proposal's temporal
   architecture family. (SalesForecasting's LSTM already does this properly and can be adapted.)
   This is Phase 2 work, not Phase 0 — it is a model-architecture change, not a correctness fix.
9. **Autoencoder preprocessing.** Heavy-tailed amount columns are not log-transformed, so a few
   columns dominate the reconstruction error. Also Phase 2 work.
10. **Hosmer-Lemeshow calibration charts exist but should not be interpreted.** Every model
    deliberately distorts its probabilities (`scale_pos_weight`, SMOTE, `class_weight='balanced'`),
    and the autoencoder's "probability" is a min-max-scaled reconstruction error, so a calibration
    test on it is not meaningful.
11. **Attribution outputs are not version-controlled.** `.gitignore` excludes `*.csv`, so
    `shap_global.csv` and friends exist only locally. They are our evidence and should be committed.
12. **No automated tests.** The proposal requires ≥85% test coverage for the final SDK;
    `src/eval_utils.py` has been exercised against synthetic data by hand but has no pytest suite yet.
13. **Model numbering skips Model 1** (models run 2–7), a leftover from an earlier numbering scheme.
14. **DemandForecast OOS is still split by brand (3 held-out brands), not by product.** The
    calendar-month fix (item 3) did not change this — it remains the most likely explanation for
    OOS R² being low and similar across every model (Section 4).

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
| **0. Fix and freeze** | ✅ **Done** (Section 7). Pinned dependencies; deduplicated disputes everywhere; moved TabNet's early stopping off the evaluation set; fixed one threshold per model; added bootstrap confidence intervals; fixed the LIME substring bug; fixed the DemandForecast OOT months and retrained everything on the corrected splits. |
| **1. Finish baseline explainability** | ✅ **Done** (Section 5). Local (per-instance) SHAP and LIME for high-risk/high-value, borderline and misclassified/high-error cases; DemandForecast cohort level (Brand/DayType/Decile); per-instance placeholder τ between SHAP and LIME; `feature_hierarchy.csv` for both datasets. |
| **2. Advanced models** | Rebuild the autoencoder with proper preprocessing; add the VAE; retrain TabNet cleanly; build a genuine sequence LSTM. |
| **3. The Tri-Explainer** | Adapted SHAP (with background-sensitivity study), Manifold-LIME, LRD — plus latency measurement and a synthetic ground-truth test. |
| **4. CLI** | Weighted Kendall's τ, per-instance CLI per the proposal's Equation (1), aggregation by decile/segment/time, and calibration of the uncertainty gate. |
| **5. Benchmarks** | Faithfulness, monotonicity, stability, latency, memory. |
| **6. SDK** | `claris_deep_xai`, PEP-8, ≥85% test coverage. |
| **7. Paper & handover** | Report regenerated from result files; manuscript; final presentation. |
