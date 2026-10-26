# ARCHITECTURE — Decisions Taken So Far

**Project:** Explainability and Consensus Attribution for Non-Linear Deep Architectures in Regulated Financial Systems
**Partners:** NodeX8 AI Technologies Pvt. Ltd. × IIT Kharagpur (HuMICE Lab, Dept. of Industrial & Systems Engineering)
**Status:** The three explainers (Adapted SHAP, Manifold-LIME, LRD) are built and have been put through identical studies on five models (Phase 3). The consensus index (CLI), its uncertainty gate and the formal benchmark suite are **not** built yet — see [What Is Deliberately Not Here Yet](#6-what-is-deliberately-not-here-yet).

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

**Decision: we keep it even though its AUC (~0.48–0.54) is weak.** This is deliberate, and worth
stating clearly because it will look odd otherwise:

- It is the architecture family the proposal names first, and it is the primary target for LRD.
- The research question is *"can we explain this model's reasoning quickly and faithfully?"* — an
  explainer is judged on whether it faithfully describes what the model did, not on whether the
  model is accurate.

### Phase 2.1 — what we tried, and the honest result

The original hypothesis (stated in an earlier version of this document) was that a handful of
large-valued, heavily skewed amount columns were dominating the autoencoder's mean-squared
reconstruction error. We tested this directly: `src/autoencoder_preprocessing.py` signed-log1p
transforms every column with |skew| > 2 on normal training rows (263 of 271 — almost every feature in
this dataset turned out to be heavily skewed, not just a handful), then standardises and clips to
`[-5, 5]`, and the autoencoder's bottleneck layer was made a plain, deterministic `nn.Linear` (no
BatchNorm/Dropout) with `encode()`/`decode()` exposed separately — both needed for Phase 3's LRD,
which backpropagates through exactly this layer.

**What actually happened, measured, not assumed:**

| | Before (Phase 0) | After (Phase 2.1) |
|---|---:|---:|
| Best validation-split reconstruction MSE | 0.2433 | **0.0812** |
| Validation OOS AUC-ROC [95% CI] | 0.537 [0.459, 0.612] | 0.478 [0.398, 0.555] |

The preprocessing dramatically improved the autoencoder's actual job — reconstructing normal clients
faithfully (MSE down ~3×, with training curves showing no instability). **It did not improve, and if
anything slightly lowered, the AUC** — but the old AUC sits comfortably inside the new model's 95% CI,
so this difference is statistical noise, not a real regression. The likely mechanism: compressing
heavy tails makes the model fit the *bulk* of normal clients better, but some of the fraud signal in
this dataset may specifically be "has an extreme, uncompressed value" — exactly what the log transform
smooths over. This is a genuine, reportable trade-off, not a bug, and we are not re-tuning the
threshold to chase a better AUC number — doing so would be exactly the kind of after-the-fact
cherry-picking this cleanup exists to avoid. The architectural fix (clean, deterministic bottleneck;
separate `encode()`/`decode()`) stands on its own merit for Phase 3 regardless of this AUC result.

### Phase 2.2 — the VAE (new model, `Model9_VAE`)

Built per the proposal (Architecture Family 1 explicitly names AE **and** VAE). Same preprocessing
as the plain autoencoder (so the two are directly comparable), 32-dimensional latent space, trained
on normal rows only, scored deterministically using `z = mu` (not a random sample — otherwise the
same client could get a different score on every run, which would also break Phase 3's LRD, which
needs a reproducible `z`).

**KL annealing** (ramping the KL loss weight from 0 to 1 over the first 20% of training) was used to
avoid posterior collapse — the well-known VAE failure mode where the model learns to ignore its input
entirely and output a generic `mu=0, sigma²=1` for everything. We checked for this directly: final
validation KL stayed at ≈9.7 (not collapsed to ≈0), and reconstruction loss fell by more than half
over training (47.2 → 20.3) — the model is genuinely using its latent space.

The VAE's AUC came out modestly, consistently higher than the plain autoencoder's across every split
(e.g. Validation OOS 0.502 vs 0.478; OOT Dec 0.565 vs 0.519 — see
`Model_Outputs/Model9_VAE/vae_vs_ae_comparison.csv`), though with only ~51 disputes in validation
these differences are also well within noise and should not be read as "the VAE is better."

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

### Phase 3 — the Tri-Explainer (Adapted SHAP, Manifold-LIME, LRD)

All code is in `src/explainers/`; 33 tests (`python3 -m pytest tests -q`) check the maths against
cases with known answers (closed-form IG, finite-difference checks of both LRD gradient steps,
exact recovery of a linear black box by Manifold-LIME, and a hand-built autoencoder that cannot "see"
some features). One driver (`studies.py`) puts **every** model through the *same* procedures, so
results are comparable across models: the autoencoder and VAE (TradeFraud, 251 clients: all 51
validation disputes + 200 random normals), TabNet (same 251 clients), DemandForecast TabNet (250
random out-of-sample rows) and the 28-day LSTM (120 of those rows; its Manifold-LIME runs in 1,428
dimensions and is ~25× costlier per client).

**What is explained.** Autoencoder/VAE: the reconstruction error L(x)=Σⱼ(xⱼ−x̂ⱼ)² ("which features make
this client look anomalous to the model?"). TabNet (TradeFraud): the dispute log-odds. TabNet
(DemandForecast): predicted demand. LSTM: the output head's score *before* its final ReLU (explaining the
clamped value would give an all-zero explanation for any client predicted ≤ 0). TabNet was trained on raw
values, but "realistic neighbours" are only well defined after the log/standardise/clip preprocessing, so
all explainers work in that preprocessed space and explain TabNet evaluated at the inverse transform; this
reproduces the real TabNet closely (correlation 0.995, mean |log-odds difference| 0.019 against a spread of 1.13).

**Chosen before any results were seen:** the *primary* variant of each explainer — Adapted SHAP =
Integrated Gradients averaged over 10 k-means baselines; Manifold-LIME = manifold perturbations
(Ledoit-Wolf covariance, scale 0.3, range clip, integer snapping); LRD = λ=0.5 — so the headline cannot be
cherry-picked. Everything else is reported as an ablation.

**Where we deliberately departed from the roadmap, and why**

1. *LRD's Step D could not be satisfied as written.* The roadmap asked for λ chosen so that the
   attributions sum to L. With the latent term un-normalised (total S ≠ L) the constraint λL+(1−λ)S=L
   has only the solution λ=1 — the trivial per-feature-error baseline. Normalising the latent term to
   total L makes completeness hold *by construction* for any λ (verified: max relative error 10⁻⁶), so
   λ became a free blend that is fixed in advance and openly ablated over {0, 0.5, 1}.
2. *The "feature-wise mean" baseline* is the arithmetic mean of the **raw** values pushed through the
   preprocessing. In the standardised space the mean is the origin, which would just duplicate the zeros
   baseline. (DemandForecast has no log transform, so there the training *median* is used instead.)
3. *DeepLIFT was excluded:* captum's rules cover `nn.Module` nonlinearities only; the squared
   reconstruction error would be silently mis-attributed. IG and GradientSHAP are exact for any
   differentiable scalar.
4. *Integer snapping was added to Manifold-LIME* (96 of TradeFraud's 271 columns and 31 of
   DemandForecast's 51 are integer-valued; a promotion flag of 0.37 is not a real day), with a
   no-rounding variant kept so each realism constraint's effect can be separated.
5. *The faithfulness check uses two replacement operators, and that was added after seeing the first
   one's results.* Replacing 5 features with another client's values raised reconstruction error by ~10
   (mean L is 22.6) **even for a random ranking** — grafting one client's values onto another creates an
   off-manifold client. A "repair" operator (set the features to the typical value, 0) was added. Both are
   reported; conclusions are drawn only where they agree. Each has its own bias (below).
6. *Bugs found by the tests and fixed:* captum's GradientSHAP draws from numpy's *global* RNG, so seeding
   torch alone did not make it reproducible; and a 3-operand `einsum` made Manifold-LIME ~4× slower than
   needed (the first run was killed and redone).

**Latency** (median ms for one client, CPU; KernelSHAP from only 1–5 clients so rough):

| Explainer | AE | VAE | TabNet (TradeFraud) | TabNet (Demand) | LSTM |
|---|---:|---:|---:|---:|---:|
| IG, zero baseline (50 steps) | 2.6 | 2.9 | 27 | 10 | 15 |
| **IG, 10 k-means baselines (primary)** | 27 | 29 | 263 | 105 | 151 |
| GradientSHAP | 1.3 | 1.6 | 13 | 6.5 | 14 |
| **Manifold-LIME (primary)** | 80 | 84 | 170 | 57 | 367 |
| vanilla LIME | 30 | 34 | 138 | 59 | 349 |
| **LRD** | **1.7** | **1.9** | n/a | n/a | n/a |
| KernelSHAP, 10-row background | 96 | 116 | 1,068 | 558 | 2,375 |
| KernelSHAP, 100-row background | 476 | 636 | 9,235 | 4,969 | 21,068 |

- **LRD meets its <50 ms target by a wide margin (≈2 ms)** and is roughly 55–60× faster than KernelSHAP
  with a 10-row background and ≈280–335× faster with 100 rows.
- **The proposal's ">20 seconds per transaction" for KernelSHAP is only reproduced in the extreme case**
  (the 1,428-column LSTM, 100-row background: 21 s, one client). On the autoencoders — a 271-input network
  of ~100k parameters — KernelSHAP takes 0.1–0.6 s. The speed advantage is large but the motivating
  number depends on model size and background size; the paper should say so rather than quote 20 s generically.
- The proposal's ≤100 ms system budget is met by single-baseline IG, GradientSHAP, LRD and native TabNet
  masks on every model; the *primary* IG (k-means) and Manifold-LIME exceed it on TabNet and the LSTM (and
  Manifold-LIME sits at ~80 ms on the autoencoders).

**Adapted SHAP — how much does the answer depend on the baseline?** (mean weighted-τ between the rankings
two baselines produce for the same client; 1 = identical)

| | AE | VAE | TabNet (TF) | TabNet (Demand) | LSTM |
|---|---:|---:|---:|---:|---:|
| zeros vs k-means | 0.23 | 0.22 | 0.83 | 0.64 | 0.87 |
| zeros vs raw-mean | −0.04 | −0.01 | 0.82 | 0.70 | 0.79 |
| zeros vs nearest-neighbour | 0.28 | 0.22 | 0.23 | 0.53 | 0.44 |
| k-means vs GradientSHAP | 0.45 | 0.52 | 0.80 | 0.70 | 0.84 |

On the autoencoders the explanation is **strongly baseline-dependent** (top-10 overlap 1–25%); on the
supervised models it is stable across population-level baselines, but the instance-specific
nearest-neighbour baseline is consistently the outlier (it explains "difference from the closest normal
client", a different question). This is a result about the problem, not numerical noise: at 400 vs 50 steps
the autoencoder rankings change by only τ=0.984 (value residual 7.6% → 0.8%). Two cautions: the raw-mean
baseline is a poor reference — its top-5 features are *less* unusual than average (mean |z| 0.48× the
all-feature average, versus 1.8–2.9× for every other variant) and its completeness residual is 39% — **we
recommend not using it**; and **DemandForecast TabNet's IG is only moderately converged** (24% residual at
50 steps, 8% at 400; ranking τ=0.87 vs 400 steps), because its sparsemax-gated, discretely-driven response is
hard for IG's quadrature — treat those attributions as approximate.

**Manifold-LIME against vanilla LIME** (same marginal spread; differs in dependence structure, kernel and
rounding):

| | fidelity R² (manifold / no-round / vanilla) | stability τ over 5 seeds (manifold / no-round / vanilla) | distance of perturbations to nearest real row (manifold / vanilla; real-to-real) |
|---|---|---|---|
| AE | 0.84 / 0.78 / 0.77 | 0.77 / 0.50 / 0.79 | 4.81 / 5.46; 3.01 |
| VAE | 0.85 / 0.88 / 0.77 | 0.78 / 0.48 / 0.79 | same geometry |
| TabNet (TF) | 0.66 / 0.82 / 0.79 | 0.66 / 0.51 / 0.61 | same geometry |
| TabNet (Demand) | 0.76 / 0.71 / 0.67 | 0.75 / 0.62 / 0.73 | 2.78 / 2.98; 3.10 |
| LSTM (1,428 columns) | 0.71 / 0.99 / 1.00 | 0.71 / 0.62 / 0.74 | 26.8 / 27.1; 10.9 |

The honest reading: **a modest plausibility gain, and no consistent stability or fidelity gain.**
Manifold perturbations sit 3–20% closer to real clients than independent ones (the gap widens with
perturbation scale), but even at the chosen scale they are *farther* from any real client than real
clients are from each other in 271 dimensions (4.8 vs 3.0), and in 1,428 dimensions the three modes are
indistinguishable. Stability: the no-rounding variant is consistently the *least* stable (correlated
perturbations make the local regression ill-conditioned, so credit is split arbitrarily among correlated
features); with rounding it matches vanilla in the main diagnostic (0.77 vs 0.79). In a sweep with a smaller
sample budget (500 perturbations, 12 clients) the manifold version is more stable at small scale or stronger
ridge (e.g. scale 0.1: 0.85 vs 0.78; ridge 10: 0.78 vs 0.66) — so any stability advantage depends on the
sample budget and is not a robust property. Integer snapping *hurts* local fit for the piecewise models
(TabNet R² 0.82 → 0.66; LSTM 0.99 → 0.71), where discrete jumps break local linearity. About a third of
perturbed entries are clipped to the realistic range in every mode, because many features sit at their
minimum (a zero count) so half of any perturbation falls below it.

**LRD — what the evidence does and does not show.** Completeness holds (10⁻⁶) and attributions are
non-negative magnitudes. The λ ablation (autoencoder shown; the VAE is the same within noise; the
VAE-specific σ-weighting changed nothing material):

| faithfulness check: how much more do top-5 features matter than random-5? (top/random) | donor replacement | zero ("repair") replacement | clients where top-5 matters more than bottom-5 |
|---|---:|---:|---:|
| LRD λ=0 (pure latent path) | 1.7 | 3.0 | 82% (100% zero) |
| LRD λ=0.5 (**pre-declared primary**) | 0.9 | 1.4 | 81% (100% zero) |
| LRD λ=1 (= the trivial per-feature error) | 0.8 | 1.3 | 71% |
| IG, k-means (primary Adapted SHAP) | 1.8 | 1.8 | 59% / 74% |
| Manifold-LIME (primary) | 1.4 | 1.8 | 80% / 94% |
| vanilla LIME | 1.0 | 1.5 | 74% / 94% |
| random ranking (control) | 1.1 | 1.1 | 41% / 51% |

Routing through the latent code *does* add something over the trivial per-feature error (λ=0 beats λ=1 on
both operators), and the **pre-declared primary λ=0.5 sits much closer to the weak end** — the evidence so
far favours λ=0. We are **not** switching the primary after seeing this: the check is itself biased — zero
replacement favours any attribution proportional to the input (IG from the origin, LRD, LIME contributions
are all zero for a feature already at its typical value, so replacing it with zero changes nothing), and
donor replacement is dominated by disruption. In fact only the direct per-feature error (λ=1) shows a
positive *repair* (mean error falls when its top features are set to typical values); every
input-proportional ranking raises error because setting a few large, mutually consistent values to zero
makes the client incoherent. λ is therefore to be settled in Phase 5, using the synthetic ground-truth test
(corrupt known features) on data not used here. Ratios of top to *bottom* effect are not quoted: for TabNet
and the LSTM the bottom effect is exactly zero (sparse attention; constant columns), making them undefined.

**Do the explainers agree?** Placeholder weighted-τ between the three *primary* explainers (this is a
preview, **not** the CLI): autoencoder/VAE — Adapted SHAP–Manifold-LIME 0.18, Adapted SHAP–LRD 0.27,
Manifold-LIME–LRD 0.52; TabNet (TradeFraud) — 0.16 between the two explainers, 0.43 / 0.12 against TabNet's
own attention masks; TabNet (Demand) — 0.41, 0.57, 0.37; LSTM — 0.41 on the 1,428 time×feature cells, 0.34
after summing over time to the 51 features. The explainers disagree substantially, most in the 271-feature
problems. That disagreement is exactly what Phase 4 is meant to measure and gate on.

**Sanity check (the roadmap's "do the top features look unusual?").** `show_tri_explainer_case.py` reports,
for each variant, how unusual its top-5 features are (mean |z| in the preprocessed space; random ≈ 0.70).
Every variant except the raw-mean baseline scores 1.8–2.9× above average. A worked example: for the
dispute with the highest reconstruction error all three primary explainers converge on
`net_trade_amt_l_003d` (a net trade of −15,041, in the bottom 1.5% of normal clients) and on trade-count
features in the 99th percentile; for a dispute the model barely notices (error 3.4, i.e. typical) they pick
`cash_dormancyPeriod` (98th percentile) — the model is simply not very anomalous about it, consistent with
its weak AUC. Note that in the explained sample the autoencoder's mean error is *higher for normal clients
(23.8) than for disputes (17.6)*: these are faithful explanations of a model whose anomaly signal points the
wrong way on this data.

**Model-specific observations.** TabNet's sparse attention gives many features an exactly-zero attribution,
so its explanations are short lists. The LSTM's IG attribution is concentrated on the most recent day: 26%
of total mass at t-0, 48% in the last three days, 15% across the first fourteen — the 28-day window
matters, but mostly through the final day, whose features already include the engineered lag and rolling
statistics.

---

## 6. What is deliberately NOT here yet

This section exists so that nobody — including us — mistakes groundwork for the finished research.

| Required by the proposal | Status |
|---|---|
| **Adapted SHAP** (IG / GradientSHAP on the autoencoder's reconstruction error, with a study of how the baseline changes the answer) | ✅ Built and studied (Phase 3) |
| **Manifold-LIME** (covariance-shaped perturbations that stay on the data manifold) | ✅ Built and studied (Phase 3) — honest result: modest plausibility gain, no consistent stability/fidelity gain |
| **LRD — Latent Reconstruction Decomposition** (back-projecting reconstruction error through the bottleneck, target < 50 ms) | ✅ Built (≈2 ms); **provisional** — formulation is ours, awaiting NodeX8's confirmation (Section 8, Q2); λ to be settled in Phase 5 |
| **VAE** (probabilistic latent space; richer LRD target) | ✅ Built (Phase 2) |
| **Local / per-instance explanations** | ✅ Built (Phase 1) |
| **CLI(x)** — per-transaction consensus using **weighted Kendall's τ**, per the proposal's Equation (1) | Not built (a placeholder-τ *preview* exists; it is not the CLI) |
| **Uncertainty gate** — a calibrated τ threshold that routes low-agreement cases to a human | Not built |
| **Formal faithfulness / stability / adversarial / latency benchmark suite** | Partly: latency, a first faithfulness check and Manifold-LIME stability are done; the synthetic ground-truth test, deletion/insertion curves and adversarial robustness are Phase 5 |
| **Adapting LRD to TabNet / LSTM** (roadmap stretch goal) | Not attempted — no autoencoder bottleneck; those models get Adapted SHAP + Manifold-LIME |
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

### Fixed in Phase 2

8. ~~**The DemandForecast "LSTM" is not temporal.**~~ **Fixed.** `train_lstm.py` was rebuilt to use a
   genuine 28-calendar-day lookback window per product (UPC), adapted from SalesForecasting's
   reference LSTM windowing convention. A sanity check was added and passed: shuffling the order of
   the 27 history days within a sequence changes the model's prediction by a mean of 1.51 units — the
   old length-1 model would have shown exactly 0, since it had nothing to shuffle. See ROADMAP.md
   Phase 2.4.
9. ~~**Autoencoder preprocessing.**~~ **Addressed, with an honest caveat.** Heavy-tailed columns are
   now signed-log1p-transformed, scaled, and clipped (`src/autoencoder_preprocessing.py`). This
   measurably improved reconstruction fidelity (MSE 0.243 → 0.081) but did **not** improve AUC — see
   Section 5's Phase 2.1 write-up for the full honest result and why we are not re-tuning to chase a
   better number.

### Still open

10. **Hosmer-Lemeshow calibration charts exist but should not be interpreted.** Every model
    deliberately distorts its probabilities (`scale_pos_weight`, SMOTE, `class_weight='balanced'`),
    and the autoencoder's "probability" is a min-max-scaled reconstruction error, so a calibration
    test on it is not meaningful.
11. **Attribution outputs are not version-controlled.** `.gitignore` excludes `*.csv`, so
    `shap_global.csv` and friends exist only locally. They are our evidence and should be committed.
12. **No automated tests.** The proposal requires ≥85% test coverage for the final SDK;
    `src/eval_utils.py` has been exercised against synthetic data by hand but has no pytest suite yet.
15. **LRD is our formulation, not NodeX8's.** If NodeX8 has a specification it overrides this one. The
    primary λ=0.5 was fixed in advance but the ablation favours λ=0; settle it in Phase 5 on independent data.
16. **The Phase 3 faithfulness checks are biased by their replacement operators** (see Section 5), so they
    support only coarse statements (every explainer beats random; the raw-mean baseline does not). They
    are not a ranking of the explainers.
17. **DemandForecast TabNet's Adapted SHAP is only moderately converged** (24% IG completeness residual at
    50 steps; ranking τ=0.87 against 400 steps).
18. **Explanations are computed on small samples** (251 TradeFraud clients incl. all 51 unique disputes; 250
    Demand rows; 120 for the LSTM), KernelSHAP latency from 1–5 clients, and LSTM Manifold-LIME diagnostics
    on 20 rows without sweeps. Differences between explainers of a few hundredths should not be read as real.
19. **Perturbations are not truly "on the manifold" in high dimension.** At the chosen scale Manifold-LIME's
    copies are farther from real clients than real clients are from each other (271-d) and no closer than
    vanilla's in 1,428-d; Gaussian-plus-rounding is only an approximation of the data's structure.
20. **The explained models are weak detectors** (autoencoder/VAE AUC ≈ 0.5, error higher for normals than
    disputes in the sample). The explainers describe what the models do, not what fraud looks like.
13. **Model numbering skips Model 1 and Model 8** (models are 2–7, then 9 for the VAE), a leftover
    from an earlier numbering scheme — harmless, left as-is rather than renumbering everything.
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
   (**Update:** a concrete formulation is now implemented in `src/explainers/lrd.py` with one correction to the
   roadmap's Step D, explained in Section 5. It is provisional until NodeX8 confirms or replaces it.)
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
| **2. Advanced models** | ✅ **Done** (Section 5). Autoencoder rebuilt with signed-log1p preprocessing and a clean deterministic bottleneck (honest result: reconstruction fidelity improved 3×, AUC did not); VAE built with KL annealing (no posterior collapse); DemandForecast LSTM rebuilt with real 28-day sequences (sanity-checked). |
| **3. The Tri-Explainer** | ✅ **Done** (Section 5). Adapted SHAP, Manifold-LIME and LRD implemented, tested (33 tests) and put through identical studies on five models: latency, baseline sensitivity, Manifold-LIME diagnostics, a first faithfulness check, and an agreement preview. Several honest negative/neutral results (Manifold-LIME's benefit is modest; LRD's λ unsettled; KernelSHAP's ">20 s" only reproduced in the extreme case). |
| **4. CLI** | Weighted Kendall's τ, per-instance CLI per the proposal's Equation (1), aggregation by decile/segment/time, and calibration of the uncertainty gate. |
| **5. Benchmarks** | Faithfulness, monotonicity, stability, latency, memory. |
| **6. SDK** | `claris_deep_xai`, PEP-8, ≥85% test coverage. |
| **7. Paper & handover** | Report regenerated from result files; manuscript; final presentation. |
