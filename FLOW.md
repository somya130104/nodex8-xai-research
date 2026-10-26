# FLOW — A Beginner's Walkthrough of This Codebase

If you have just been handed this repository and have never seen it before, read this file top to
bottom. It assumes you know a little Python but nothing about this project.

For *why* each decision was made, read [ARCHITECTURE.md](ARCHITECTURE.md) afterwards.

---

## 1. The project in one paragraph

A bank wants to use complicated AI models to catch fraud. Regulators say: you cannot use a model
you cannot explain. So this project (a) trains some models, (b) runs tools that explain *which
features drove each prediction*, and (c) — this is the research part, **not built yet** — invents
better explanation methods for the hardest models, plus a score called **CLI** that measures
whether different explanation methods agree with each other. When they disagree too much, a human
should review the case instead of trusting the machine.

---

## 2. Vocabulary you need first

Six terms. Everything else in the repo is built from these.

| Term | Plain meaning |
|---|---|
| **Feature** | One column of input. "Days since last trade" is a feature. We have 271 of them for fraud. |
| **Target** | The column we are trying to predict. Here: `Dispute` (fraud yes/no) or `QUANTITY` (units sold). |
| **Attribution** | One number per feature, for one prediction, saying how much that feature pushed the answer. The output of "explaining" is a list of 271 such numbers. |
| **SHAP** | An explanation method borrowed from game theory — it splits the prediction fairly among the features, like splitting a bill. `TreeSHAP` is the fast exact version for tree models. |
| **LIME** | A different explanation method. It creates many slightly-altered copies of your input, watches how the model reacts, and fits a simple straight-line model to that local behaviour. The line's slopes are the explanation. |
| **Autoencoder** | A model that learns to squash an input down to a few numbers and then rebuild it. Train it only on *normal* clients, and when it later fails badly at rebuilding someone, that person is unusual → possibly fraud. The "failure size" is called **reconstruction error**. |

Two more you will meet in file names:

- **OOS** = *Out-of-Sample* — different customers, same time period. Tests: does this work on people we have not seen?
- **OOT** = *Out-of-Time* — future months. Tests: does this still work next quarter, as behaviour drifts? This is the harder and more honest test.

---

## 3. Map of the repository

```
nodex8/
├── ARCHITECTURE.md       ← the decisions we made and why (read second)
├── FLOW.md               ← this file (read first)
│
├── requirements.txt      ← pinned package versions (pip install -r requirements.txt)
├── src/                  ← shared code. Currently just eval_utils.py (Phase 0 fixes);
│                           reserved for the real explainer library (claris_deep_xai) later
│
└── data/
    ├── TradeFraud/            ← PRIORITY 1. Fraud detection. Classification.
    │   ├── Data/                   raw data as supplied by NodeX8 (do not edit)
    │   ├── Modeling_Data/          raw data after splitting into train/val/OOT
    │   ├── Modeling_Data_Enriched/ ← the splits WITH engineered features. Models use THIS one.
    │   ├── scripts/                all the code
    │   ├── Model_Outputs/          one folder per trained model: scorecards, saved weights, charts
    │   └── XAI_Outputs/            the SHAP and LIME results
    │
    ├── DemandForecast/        ← PRIORITY 2. Retail demand. Regression.
    │   ├── features.csv            the dataset with features already built in
    │   ├── samples/                dev / oos / the three OOT months
    │   ├── scripts/
    │   ├── Model_Outputs/
    │   └── XAI_Outputs/
    │
    └── SalesForecasting/      ← REFERENCE ONLY — do not modify
        └── Modeling/               a complete, finished pipeline we are copying conventions from
```

> **The single most common mistake:** TradeFraud has two similar folders,
> `Modeling_Data/` and `Modeling_Data_Enriched/`. **Every model uses the `_Enriched` one** — it has
> the 91 extra engineered features, for 271 total. If you load the other one, your saved models
> will crash or give nonsense.

---

## 4. How the code flows

Data moves left to right. Each box is a script; each arrow is a file written to disk.

```
   RAW DATA                 PREPARE                   TRAIN                  EXPLAIN
  ──────────              ───────────              ──────────             ───────────

  Data/Data.csv  ──►  preprocess_raw_data.py
                           │
                           ▼
                      feature_engineering.py   ──►  Modeling_Data_Enriched/
                           │                              │
                      split_modeling_data.py              │
                           │                              ▼
                           │                     train_model2_xgboost.py  ─┐
                           │                     train_model3_rf.py        │
                           │                     train_model5_lightgbm.py  ├─► Model_Outputs/
                           │                     train_model6_logreg.py    │    <Model>/
                           │                     train_model4_autoencoder.py│    ├ performance_scorecard.csv
                           │                     train_model7_tabnet.py    ─┘    ├ model_config.json
                           │                                                     └ the saved model
                           │                              │
                           │                              ▼
                           └──────────────────►  run_baseline_shap.py  ──► XAI_Outputs/shap_global.csv
                                                 run_baseline_lime.py  ──►            shap_cohort.csv
                                                                                      lime_global.csv
                                                                                      lime_cohort.csv
                                                 extract_tabnet_masks.py ─► tabnet_masks.csv
```

DemandForecast is the same shape with different names: `create_samples.py` makes the splits, then
`train_xgboost.py` / `train_rf.py` / `train_lightgbm.py` / `train_ols.py` / `train_tabnet.py` /
`train_lstm.py`, then the same two `run_baseline_*.py` scripts.

**Important: the arrows only run in one direction, and the order matters.** You cannot run
`run_baseline_shap.py` before the models exist, because it loads the saved model files.

---

## 5. Reading order — the 30-minute tour

Follow this exact order and the codebase will make sense. Start with TradeFraud; it is the priority.

### Step 1 — Look at the data (5 min)

```bash
cd data/TradeFraud
head -2 Modeling_Data_Enriched/train.csv | cut -c1-400   # the column names
wc -l Modeling_Data_Enriched/*.csv                        # how big each split is
head -5 Data/variable_dictionary.csv                      # what each feature MEANS
```

`variable_dictionary.csv` is the most useful file in the repo for a newcomer — it describes all 277
columns in plain English, with their group, sub-group and how they were calculated.

### Step 2 — Read one training script end to end (10 min)

Open **`scripts/train_model2_xgboost.py`**. It is the clearest one, and all the others follow the
same five-part shape:

1. **Config block at the top** — paths, which files to evaluate on, hyperparameters.
2. **`load_features()`** — read a CSV, convert `Dispute` from `"D"`/`"N"` to 1/0, drop ID columns.
3. **`evaluate()`** — compute AUC, KS, Gini, Top-Decile Recall for one dataset.
4. **The training call** — fit the model.
5. **The loop over `eval_sets`** — score on train, validation, and each OOT month, then write
   `performance_scorecard.csv`.

Once you have read this one, the other five are variations: Random Forest uses SMOTE, LightGBM uses
`is_unbalance`, Logistic Regression adds a scaler, TabNet is a neural network, and the autoencoder
is the odd one out (see Step 4).

### Step 3 — Look at a result (3 min)

```bash
cat Model_Outputs/Model2_XGBoost/performance_scorecard.csv
```

One row per dataset. Read along the `AUC_ROC` column: high on train, lower on validation, lower
still on the out-of-time months. That downward slope is normal and expected — it is the model
ageing as fraud patterns change. Then read `Top_Decile_Recall`: that is the business number, "what
fraction of all fraud do we catch if we investigate only the riskiest 10%?"

> Be careful with the `Precision` / `Recall` / `F1` columns right now — each dataset currently uses
> its own tuned threshold, so they are not comparable across rows. This is a known issue
> (ARCHITECTURE.md, limitation 5).

### Step 4 — Understand the autoencoder, because it is different (7 min)

Open **`scripts/train_model4_autoencoder.py`**. Every other model answers *"what is the probability
this is fraud?"*. This one answers *"how weird is this client?"*, and that difference is the whole
reason this research project exists.

The key steps in the file:

1. Keep **only non-dispute clients** from training (`X_normal = X_train_all[y_train_all == 0]`).
2. Train the network to rebuild its own input — input and target are the same thing.
3. For any new client, the score is the **reconstruction error**: how far the rebuilt version is
   from the original.
4. Squash that error into a 0–1 range so it can be scored like a probability.

Because the output is an error value rather than a probability, SHAP and LIME cannot be pointed at
it in the ordinary way. **Finding the right way to explain this model is the core research task.**

### Step 5 — Read the explainability scripts (5 min)

Open **`scripts/run_baseline_shap.py`**. Note the three levels it produces:

- **Global** — average the absolute SHAP value for each feature across everyone → `shap_global.csv`
- **By segment** — the same average, computed separately within each customer segment
- **By decile** — the same, within each 10% risk band → both go to `shap_cohort.csv`

Then `run_baseline_lime.py` does the same with LIME.

The output files are in "long" format — one row per (model, feature) rather than a wide grid:

```
Model,Feature,Mean_Abs_SHAP
Model2_XGBoost,cash_dormancyPeriod,0.0142
```

This shape is deliberate: it makes it easy to stack results from many models and compare them.

---

## 6. Running things yourself

### Set-up

```bash
cd /Users/somyabhadada/nodex8
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **On macOS:** `xgboost` and `lightgbm` need the OpenMP runtime, which their wheels do not bundle.
> Run `brew install libomp` once before `pip install`, or the import will fail with a `libomp.dylib
> not loaded` error. See the comment at the top of `requirements.txt` for a scipy-wheel issue we also
> hit (and the exact pinned version that avoids it) on very new macOS builds.

### Running a script

Every script has its paths hard-coded at the top and takes no arguments, so you just run it:

```bash
python data/TradeFraud/scripts/train_model2_xgboost.py
```

Rough timings on a laptop: tree models take a couple of minutes, the autoencoder and TabNet take
five to fifteen. **LIME is the slow one** — it builds thousands of perturbed samples per explained
client, so expect tens of minutes.

### Sensible order if starting from scratch

```
1. train_model2_xgboost.py        (and 3, 5, 6 — the baselines)
2. train_model4_autoencoder.py    (and 7 — TabNet)
3. run_baseline_shap.py
4. run_baseline_lime.py            ← slow, run it last
5. extract_tabnet_masks.py
```

The data splits in `Modeling_Data_Enriched/` already exist, so you do not need to re-run the
preprocessing scripts unless you intend to change the feature engineering.

---

## 7. What each file is for

### Shared code — `src/`

| File | What it does |
|---|---|
| `eval_utils.py` | The Phase 0 fixes, factored out once instead of copy-pasted into six scripts: `dedupe_rows()` drops the 10×-duplicated dispute rows; `select_f1_threshold()` / `metrics_at_threshold()` implement "pick a threshold on validation only, reuse it everywhere"; `bootstrap_auc_ci()` / `bootstrap_auc_pr_ci()` add confidence intervals. Every TradeFraud script imports from here via `sys.path.append("/Users/somyabhadada/nodex8"); from src.eval_utils import ...`. |

### TradeFraud — `data/TradeFraud/scripts/`

**Data preparation** (already run; only re-run if changing features)

| File | What it does |
|---|---|
| `preprocess_raw_data.py` | Cleans the raw supplied data |
| `feature_engineering.py` | Builds the 91 extra features (dormancy ratios, buy/sell imbalance, …) |
| `split_modeling_data.py` | Cuts the data into train / validation / four OOT months |
| `generate_customer_master.py`, `generate_consent_log.py` | Build the customer and consent reference tables |
| `generate_variable_dictionary.py`, `update_variable_dictionary.py` | Maintain the plain-English feature dictionary |

**Models** — each writes to its own `Model_Outputs/<name>/` folder

| File | Model |
|---|---|
| `train_model2_xgboost.py` | XGBoost (gradient-boosted trees) |
| `train_model3_rf.py` | Random Forest, with SMOTE |
| `train_model4_autoencoder.py` | Autoencoder anomaly detector — **the key research target** |
| `train_model5_lightgbm.py` | LightGBM |
| `train_model6_logreg.py` | Logistic Regression, elastic-net |
| `train_model7_tabnet.py` | TabNet (attention network for tables) |
| `train_model9_vae.py` | Variational Autoencoder — the richer sibling of Model 4 (no Model 8; see FLOW.md §8) |

**Explainability and charts**

| File | What it does |
|---|---|
| `run_baseline_shap.py` | TreeSHAP / LinearSHAP at global, segment and decile level |
| `run_local_explanations.py` | Local (per-instance) SHAP + LIME on 15 representative cases per model — the adverse-action-notice level |
| `build_feature_hierarchy.py` | Builds `XAI_Outputs/feature_hierarchy.csv` from `Data/variable_dictionary.csv` |
| `run_baseline_lime.py` | The same three levels using LIME |
| `extract_tabnet_masks.py` | Pulls out TabNet's *built-in* attention masks. **Note:** this is a signal the model gives for free, not one of the three explainers the research requires |
| `plot_roc_all.py`, `plot_hosmer_lemeshow.py` | Charts. See ARCHITECTURE.md limitation 9 before interpreting the calibration chart |

### DemandForecast — `data/DemandForecast/scripts/`

| File | What it does |
|---|---|
| `create_samples.py` | Builds dev / oos / three OOT months from `features.csv` |
| `train_xgboost.py` | XGBoost with Tweedie loss (suited to the many zero-sales days) |
| `train_rf.py`, `train_lightgbm.py`, `train_ols.py` | The other three baselines |
| `train_tabnet.py`, `train_lstm.py` | The deep models — `train_lstm.py` builds real 28-day sequences (Phase 2), using `features.csv` directly for lookback history |
| `run_baseline_shap.py`, `run_baseline_lime.py` | SHAP and LIME, now including cohort output (Brand/DayType/Decile) via `cohort_utils.py` |
| `run_local_explanations.py` | Local (per-instance) SHAP + LIME on 15 representative cases per model |
| `build_feature_hierarchy.py` | Builds `XAI_Outputs/feature_hierarchy.csv` (hand-built groups — no variable dictionary exists for this dataset) |
| `cohort_utils.py` | Shared `assign_day_type()` used by both the SHAP and LIME scripts |
| `extract_tabnet_masks.py` | TabNet's native masks |

### SalesForecasting — reference only

`data/SalesForecasting/Modeling/` is a **finished** pipeline from an earlier project. Do not change
it. Worth studying as the template we are copying:

- `codes/XAI/V2/run_pipeline_regression.py` — a manifest-driven XAI pipeline producing global,
  decile and segment explanations, SHAP + LIME + a surrogate model, and drift (PSI) checks.
- `codes/XAI/feature_hierarchy.csv` — maps each feature to a Sub-Group and Group, so importance can
  be reported as "Promotional Activity: 31%" instead of as 271 individual rows. We need an
  equivalent file for both of our datasets.
- `codes/validation/` — reproducibility checks with a hash-chained audit log, plus unit tests.

---

## 8. Things that will confuse you (so they do not)

1. **Models are numbered 2–7, then 9. There is no Model 1 or Model 8.** Left over from an earlier
   numbering scheme; harmless, not worth renumbering everything to fix.
2. **`tabnet_model.zip.zip`** — a genuine double extension caused by a save bug. The loading code
   has a fallback for it. Harmless, on the tidy-up list.
3. **Disputes look 10× more common than they are — and every script now fixes this itself.** In the
   supplied data each fraud row is duplicated ten times. Every loader calls `dedupe_rows()` (from
   `src/eval_utils.py`) before doing anything else, so you will see a line like
   `[dedupe_rows] removed 1854 duplicate rows (93480 -> 91626)` printed every time you run a
   TradeFraud script. That is expected, not an error. See ARCHITECTURE.md §3 and §7.
4. **Validation AUC around 0.6–0.7, with a wide confidence interval, is not a broken model.** With a
   0.22% fraud rate on out-of-time data and only 15–18 real disputes per month, that is a normal and
   useful result — and the CI (now reported alongside every AUC) is there specifically so you don't
   over-read small differences between models as real. Accuracy is a meaningless metric here — a
   model that says "never fraud" is 99.78% accurate.
5. **A model's threshold is fixed once, on Validation OOS, and reused everywhere.** You will see one
   `Threshold_Used` value repeated down the whole `performance_scorecard.csv`, including on OOT
   months. That is intentional — see ARCHITECTURE.md §7, item 5.
6. **`train_lstm.py` now builds real 28-day sequences** (fixed in Phase 2 — it used to feed
   sequences of length 1, making it a plain feed-forward network in disguise). It needs the full
   `features.csv`, not just the split files, because a sequence near the start of an OOT month needs
   history from before that month (which lives in `dev.csv` or `oos.csv` for that product).
7. **`Dispute` is the strings `"D"` and `"N"`, not 1 and 0.** Every script converts it with
   `(df["Dispute"] == "D").astype(int)`. If you write new code and forget this, every row silently
   becomes "not fraud" and your model will look perfect.
8. **Result CSVs are not in git.** `.gitignore` excludes `*.csv`, so `shap_global.csv` and the other
   outputs live only on the local machine. You must re-run the scripts to regenerate them.
9. **You will not find CLI, LRD or Manifold-LIME anywhere.** They are not built yet — that is the
   research still to come. An earlier session produced something *named* CLI that measured a
   different thing; it was removed to avoid confusion. See ARCHITECTURE.md §6.

---

## 9. Where to go next

| You want to… | Go to |
|---|---|
| Understand why things are the way they are | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Know what is missing and in what order to build it | ARCHITECTURE.md §6 and §9 |
| Know what not to trust yet | ARCHITECTURE.md §7 (known limitations) |
| See what a finished pipeline looks like | `data/SalesForecasting/Modeling/codes/XAI/V2/` |
| Understand a specific feature | `data/TradeFraud/Data/variable_dictionary.csv` |
