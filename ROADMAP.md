# ROADMAP — What's Left To Build (Phase 1 → Phase 7)

This is the implementation plan for everything after Phase 0. Phase 0 (data fixes, dedup,
thresholds, confidence intervals) is **done** — see [ARCHITECTURE.md](ARCHITECTURE.md) Section 7.
This file picks up from there.

**Who this is for:** someone who knows a little Python and a little machine learning, but has never
seen this project before, and needs to know exactly what to build, in what order, and why.

**How to read this file:** each phase has (1) a plain-English explanation of *why* it exists, (2) the
exact deliverables, (3) step-by-step implementation notes with enough detail to start coding, and
(4) how to know you're done. Phases are in dependency order — **do not skip ahead**. Phase 3 (the
Tri-Explainer) cannot be built correctly without Phase 2's cleaned-up models; Phase 4 (CLI) is
mathematically meaningless without Phase 3's three explainers.

---

## Before you start: the one-paragraph mental model

Three things are being built, and they build on each other like a staircase:

1. **Three ways to explain a prediction** (Phase 3) — ordinary SHAP/LIME break on an autoencoder (no
   probability output) and create unrealistic data points (LIME's perturbations). So the proposal
   asks for three fixed-up explainers: **Adapted SHAP**, **Manifold-LIME**, and a brand-new method
   called **LRD**.
2. **A score that says whether those three explainers agree** (Phase 4) — called the **CLI
   (Consensus Logic Index)**. If they agree, trust the explanation. If they don't, send it to a
   human.
3. **Proof that all of this actually works** (Phase 5) — benchmarks showing the explanations are
   fast, faithful (they point at the right features), and stable (they don't change wildly for no
   reason).

Everything in Phases 1–2 is getting the ground ready for step 1. Phases 6–7 are packaging and
writing it up once steps 1–3 exist.

---

## Table of Contents

1. [Phase 1 — Finish Baseline Explainability](#phase-1--finish-baseline-explainability)
2. [Phase 2 — Advanced Models, Done Properly](#phase-2--advanced-models-done-properly)
3. [Phase 3 — The Tri-Explainer Framework](#phase-3--the-tri-explainer-framework)
4. [Phase 4 — The Consensus Logic Index (CLI)](#phase-4--the-consensus-logic-index-cli)
5. [Phase 5 — The Benchmark Suite](#phase-5--the-benchmark-suite)
6. [Phase 6 — The `claris_deep_xai` SDK](#phase-6--the-claris_deep_xai-sdk)
7. [Phase 7 — Paper, Thesis, and Handover](#phase-7--paper-thesis-and-handover)
8. [Quick-Reference Timeline](#quick-reference-timeline)

---

## Phase 1 — Finish Baseline Explainability ✅ Done

**Duration estimate:** ~2 weeks
**Depends on:** Phase 0 (done)
**Status:** implemented and verified by actually running every script — see ARCHITECTURE.md
Section 5 for what was built, the real example output, and the baseline instance-CLI numbers
observed. The checklist and "how to know you're done" sections below are kept as-written for
reference; every item in the checklist is checked off.
**Why this phase exists:** Angshuman's explicit instruction was *"For each model use SHAP and LIME
for first level explainability. Our target is to produce global, local, and at decile/cohort level."*
Global and cohort levels are done. **Local (per-instance) is missing entirely**, and this is the
level regulators actually care about — it's what produces the "why was I declined" explanation for
one specific customer.

### 1.1 Local (per-instance) SHAP and LIME

**What "local" means:** instead of "on average, which features matter" (global) or "within this
segment, which features matter" (cohort), local means "for *this one* transaction, why did the model
say what it said." The output is one attribution number per feature, for one row.

**What to build**, for each TradeFraud model (XGBoost, RF, LightGBM, LogReg) and each DemandForecast
model:

1. Pick **15 representative cases** from Validation OOS:
   - **5 highest-risk** (highest predicted probability of fraud / highest predicted demand)
   - **5 borderline** — closest to that model's own fixed threshold (`Threshold_Used` in its
     scorecard, from Phase 0), *not* closest to 0.5. A model whose threshold is 0.72 has its
     genuinely uncertain cases near 0.72, not near 0.5.
   - **5 misclassified** — false positives and false negatives (more informative than "5 lowest-risk",
     because they show you *what went wrong*, not just what the model is confident about)
2. For each of the 15 cases, compute the full attribution vector (one number per feature) using the
   same SHAP/LIME explainer already built for that model in the baseline scripts.
3. Save to `shap_local_samples.csv` and `lime_local_samples.csv`, with columns:
   `Model, Instance_ID, Case_Type (high_risk/borderline/misclassified), True_Label, Predicted_Score,
   <feature_1>, <feature_2>, ..., <feature_N>`
4. Produce a short human-readable text explanation per case — this is the actual "adverse action
   notice" regulators require, e.g.: *"This client was flagged primarily because (1) sell-side
   activity in the last 30 days was 3× the historical average, (2) 245 days of dormancy preceded this
   sudden activity, (3) the buy/sell ratio was heavily skewed."* Generate this by taking the top-3
   features by absolute attribution and writing a templated sentence.

**Where this goes:** new scripts `run_local_shap.py` / `run_local_lime.py` in each dataset's
`scripts/` folder, reusing the model-loading code already in `run_baseline_shap.py`.

### 1.2 DemandForecast cohort level (currently missing)

TradeFraud has segment/decile cohort SHAP; DemandForecast only has global. Add:
- **By BRAND** (10 brands exist in the data)
- **By day type** (weekday / weekend / holiday / promo day — all these columns already exist:
  `is_weekend`, `holiday`, `PROMOTION_FLG`)
- **By risk decile** (split predictions into 10 buckets, same pattern as TradeFraud)

### 1.3 Baseline consensus — done *correctly* this time

The proposal's CLI is **per-instance**: for one specific case, do the K explainers agree on feature
ranking? Before any cross-architecture or cross-model comparison is attempted (that comes later), do
the simplest correct version: for every baseline model, for every one of the 15 local cases above,
compute the weighted Kendall's τ between its SHAP ranking and its LIME ranking **for that one case
on that one model**. This is "baseline CLI" — a sanity check that even simple tree models show
disagreement between explainers sometimes, and it's the simplest possible rehearsal for Phase 4's
full CLI.

> Note: a correct `weighted_tau()` function doesn't exist yet in this repo — you'll build it properly
> in Phase 4.1. For this phase, `scipy.stats.weightedtau` is good enough as a placeholder; just note
> in the code that it gets replaced.

### Phase 1 deliverables checklist

- [ ] `shap_local_samples.csv`, `lime_local_samples.csv` for both datasets
- [ ] Template-generated plain-English explanation for each of the 15×N cases
- [ ] `shap_cohort.csv` / `lime_cohort.csv` for DemandForecast, with brand/day-type/decile levels
- [ ] A short per-instance baseline-CLI table (SHAP vs. LIME agreement, per case, per model)
- [ ] `feature_hierarchy.csv` for both datasets (map every feature to a Sub-Group → Group, copying
      the convention from `data/SalesForecasting/Modeling/codes/XAI/feature_hierarchy.csv`) — needed
      so importance can later be reported as "Promotional Activity: 31%" instead of 271 rows

### How to know Phase 1 is done

Run the local-explanation script for one model and manually read 3 of the 15 generated text
explanations. Do they make intuitive sense given the raw feature values for that client? If a
"highest-risk" explanation cites a feature that's actually at a normal value for that client,
something is wrong in the attribution → text pipeline — fix it before moving on.

---

## Phase 2 — Advanced Models, Done Properly ✅ Done

**Duration estimate:** ~3 weeks
**Depends on:** Phase 0, Phase 1 (helpful but not strictly blocking)
**Why this phase exists:** Phase 3 explains *models*. If the models are broken or badly set up, the
explanations built on top of them are explaining nothing meaningful. This phase fixes the two
documented weak spots (the autoencoder, and DemandForecast's non-temporal "LSTM") and adds the one
model family that's missing entirely (VAE).
**Status:** implemented and verified — see ARCHITECTURE.md Section 5 for the real numbers, including
an honest negative result (the autoencoder preprocessing fix improved reconstruction fidelity 3× but
did not improve AUC — reported as-is, not re-tuned to chase a better number) and the LSTM sanity
check (shuffling sequence order changes predictions, confirming real temporal memory is used). 2.3
needed no new work (TabNet was already fixed in Phase 0); 2.5 (FT-Transformer) was skipped as the
optional stretch goal it always was.

### 2.1 Rebuild the Autoencoder properly (TradeFraud)

**The problem today:** the autoencoder's AUC is ~0.53 (barely better than a coin flip). ARCHITECTURE.md
already diagnoses why: heavy-tailed amount columns (`cash_Td_Samt`, `Trade_totalAmt_l_365d`, etc.) are
fed in as raw standardized values, so a handful of large-valued columns dominate the reconstruction
error and drown out everything else.

**What to do:**
1. **Log-transform skewed columns before scaling.** Identify amount/count columns (anything with
   `_Amt`, `Samt`, `Bamt`, `script`, `numbers` in the name) and apply `np.log1p(x)` to them *before*
   fitting `StandardScaler`. This compresses the heavy tail so no single feature dominates the loss.
2. **Clip extreme values** after scaling, e.g. to `[-5, 5]`, so a handful of outlier clients can't
   single-handedly blow up the reconstruction error for everyone else.
3. **Re-check the architecture.** Keep the bottleneck (32 units) but make sure the final bottleneck
   layer is a clean `nn.Linear` with **no Dropout and no BatchNorm** on it — Phase 3's LRD method
   needs a well-defined, deterministic latent vector `z` to back-project through. Dropout on the
   bottleneck would make `z` different on every forward pass, which breaks LRD.
4. **Expose `encode()` and `decode()`** as separate callable methods on the model class, not just a
   combined `forward()` — Phase 3 needs to call them independently.
5. **Retrain and re-evaluate.** Don't chase a specific AUC target — report whatever it is, honestly,
   and say explicitly in the write-up: *"the autoencoder's job in this research is to be explained,
   not to maximize fraud-catching AUC; an explainer's faithfulness is judged against what the model
   actually does, not against the label."*

### 2.2 Build the VAE (Variational Autoencoder) — new model

**What a VAE adds over a plain autoencoder:** instead of squashing input into one fixed point `z`,
it squashes input into a *distribution* — a mean `μ` and a spread `σ²`. This means the model can
express "I'm not sure what this looks like" (wide `σ²`), which is valuable for fraud detection: an
unusual client should have both high reconstruction error *and* high uncertainty.

**Architecture:**
```
Input (271 features, after the same log1p + scaling as 2.1)
  → Dense(512, ReLU) → Dense(256, ReLU)
  → two parallel heads: μ = Dense(32), log_σ² = Dense(32)
  → reparameterization trick: z = μ + ε * exp(log_σ²/2), where ε ~ N(0,1)
  → Dense(256, ReLU) → Dense(512, ReLU) → Output (271 features, no activation)

Loss = Reconstruction Loss (MSE) + β * KL_divergence
KL_divergence = -0.5 * mean(1 + log_σ² - μ² - exp(log_σ²))
```

**The one real risk, and how to avoid it:** a badly-scaled VAE loss causes "posterior collapse" — the
KL term dominates and the model just learns to output `μ=0, σ²=1` for everything, ignoring the input
completely. Avoid this by:
- Using a `β` weight on the KL term, starting at `β=0.0` for the first few epochs and "annealing" it
  up to `β=1.0` over the first ~20% of training (this is a standard, well-documented fix called KL
  annealing — search for "KL annealing VAE" for reference implementations).
- Watching the reconstruction loss during training — if it stays flat near its initial value while
  the KL loss drops to near zero immediately, that's collapse; lower `β` further.

Train on normal (non-dispute) rows only, exactly like the plain autoencoder. Save to
`Model_Outputs/Model9_VAE/` (there's no Model 1 either, so this gap is consistent with the existing
numbering quirk — don't worry about renumbering).

### 2.3 Retrain TabNet cleanly

TradeFraud's TabNet is already correctly fixed in Phase 0 (no more leakage). Nothing new needed here
— just make sure Phase 3's explainer work uses the already-fixed model, not an older cached one.

### 2.4 Build a genuine sequence LSTM for DemandForecast

**The problem:** DemandForecast's current `train_lstm.py` feeds sequences of length 1, which makes it
mathematically identical to a feed-forward network — there's no "memory" of past days being used.

**What to do:**
1. **Build real sequences.** For each product (UPC), sort by date, and for each day create a 28-day
   window of that product's past feature values (price, promo flags, past quantities) as the input
   sequence, with the next day's `QUANTITY` as the target.
2. **Reuse SalesForecasting's LSTM as the template** — it already does exactly this correctly, with
   13-week sequences per customer segment. The code is at
   `data/SalesForecasting/Modeling/codes/lstm_model.py`. Adapt its windowing logic rather than
   starting from scratch.
3. Keep the rest of the architecture similar (2-layer LSTM, 64 hidden units) but now the hidden state
   genuinely carries information forward across the 28-day window.
4. This becomes the one true "temporal" model family (Architecture Family 4 in the proposal) and the
   main target for the temporal variant of the Tri-Explainer in Phase 3.

### 2.5 Optional stretch: FT-Transformer (if time allows)

The proposal also names FT-Transformer alongside TabNet as an option for the "sparse attention
tabular network" family. If time permits after 2.1–2.4, add one for TradeFraud or DemandForecast
using the `rtdl` Python package, which has a ready-made `FTTransformer` class. Not required — TabNet
already covers this architecture family.

### Phase 2 deliverables checklist

- [ ] Autoencoder retrained with log1p + clipping; `encode()`/`decode()` exposed separately
- [ ] VAE built, trained on normal rows only, with KL-annealing to avoid collapse
- [ ] Both saved to `Model_Outputs/`, with scorecards reported honestly (no target AUC to hit)
- [ ] DemandForecast LSTM rebuilt with real 28-day sequences, evaluated the same way as before
- [ ] (Optional) FT-Transformer for one dataset

### How to know Phase 2 is done

For the autoencoder/VAE: plot the reconstruction error distribution for normal vs. dispute clients
(a simple histogram). You should see *more separation* between the two distributions than before —
even a small visible gap is a meaningful improvement over near-total overlap. For the LSTM: confirm
that shuffling the order of days within a sequence changes the model's prediction (if it doesn't,
the "memory" isn't actually being used, and something is still wrong).

---

## Phase 3 — The Tri-Explainer Framework ✅ Done

**Duration estimate:** ~4–5 weeks (this is the largest phase — the proposal says to spend ~80% of
remaining time here and in Phase 4)
**Depends on:** Phase 2 models
**Why this phase exists:** this is the actual research contribution. Three explainers, each solving
one of the three problems named in the proposal (slow KernelSHAP, unrealistic LIME perturbations, no
fast method for autoencoders).

### 3.1 Vocabulary check before starting

- **Background / baseline reference:** what gradient-based explainers compare your input *against*.
  SHAP-family methods compute "how different is this input from a typical/neutral input, and which
  features caused that difference." The choice of "typical/neutral" changes the answer.
- **Manifold:** the subspace of "realistic" data. Real clients' features are correlated (someone with
  a high 30-day trade volume usually also has a somewhat-high 90-day volume). A random perturbation
  that ignores this creates a financially impossible, unrealistic fake client.
- **Latent space / bottleneck:** the compressed 32-number representation inside an autoencoder.

### 3.2 Explainer 1 — Adapted SHAP / Integrated Gradients

**What it is:** gradient-based attribution. Instead of randomly sampling (slow), it uses the model's
own gradients (fast — a handful of backward passes).

**Implementation steps:**
1. **Define the scalar being explained**, per model:
   - Autoencoder/VAE: `L(x) = sum((x - x_hat)^2)` — the total reconstruction error (not any single
     feature's error, the overall score).
   - TabNet / supervised models: the model's output logit (pre-sigmoid), not the final probability
     — gradients behave better on logits.
   - LSTM: the model's final-day prediction.
2. **Wrap the scalar as a tiny PyTorch module.** Write a thin `nn.Module` whose `forward(x)` returns
   exactly that one scalar. This makes it pluggable into the `captum` library's explainers directly.
3. **Run three captum methods** for comparison:
   - `captum.attr.IntegratedGradients` (50 steps, standard path-integral from baseline to input)
   - `captum.attr.GradientShap` (a SHAP-flavored version of the above, using multiple random baselines)
   - `captum.attr.DeepLift` (a faster single-pass alternative)
4. **The required background-sensitivity study** (explicitly asked for in the proposal — this
   becomes its own small section of the final paper): run the SAME explanation with 4 different
   baselines and compare results using weighted Kendall's τ (built in Phase 4.1, or `scipy`'s version
   as a placeholder for now):
   - all-zeros vector
   - the feature-wise mean of the training data
   - k-means cluster centers of normal training rows (k=10)
   - the single nearest normal-client neighbor of the instance being explained
   Report a table: how much does the top-10 feature list change depending on which baseline was used?
5. **Measure latency.** Time 1,000 single-instance explanations and report p50/p95 milliseconds. This
   is the number that gets compared against the ">20 seconds for KernelSHAP" claim in the proposal
   — actually measure KernelSHAP on the autoencoder once, for real, instead of just quoting the
   20-second figure from the proposal text.

**Output file:** `adapted_shap_<model>.csv`, columns `Instance_ID, Feature, Attribution, Baseline_Used`.

### 3.3 Explainer 2 — Manifold-LIME

**What it is:** standard LIME, but perturbations are generated so that correlated features move
together, instead of being randomly and independently jittered.

**Implementation, step by step** (this is subtle — read carefully, a naive implementation breaks on
exactly the cases it's meant to help with):

1. **Estimate the feature covariance matrix `Σ`** from *normal* training rows (not the instance being
   explained, and not disputes — you want "what does a typical realistic feature combination look
   like"). Use `sklearn.covariance.LedoitWolf` instead of a raw `np.cov()` — the raw covariance of
   271 correlated features is often singular/unstable; Ledoit-Wolf shrinkage fixes this cleanly.
2. **Compute the Cholesky factor** `L` such that `Σ = L @ L.T`.
3. **Generate perturbations correlated around the instance `x` being explained** — NOT around the
   overall training mean. This is the single most common mistake people make implementing this:
   ```
   for each of ~1000 perturbations:
       z = x + scale * L @ epsilon         # epsilon ~ N(0, I)
   ```
   where `scale` is a small tunable number (start around 0.3, tune once). Sampling around `x` means
   even an unusual client (like a fraud case) gets locally-realistic neighbors, rather than having
   every neighbor pulled toward the "normal" average and rejected for being too far from it.
4. **Keep perturbations within the training data's realistic range**: clip each feature to its
   observed `[min, max]` in training. Round any genuinely integer-valued columns.
5. **Weight each neighbor** by `w = exp(-mahalanobis_distance(x, z)^2 / width^2)` — closer, more
   plausible neighbors count more when fitting the local surrogate.
6. **Fit a weighted Ridge regression** (`sklearn.linear_model.Ridge`, with `sample_weight=w`) on the
   neighbors' predictions. Report `contribution_j = coefficient_j * (x_j - mean_j)` as the
   attribution — this scales it to be comparable with SHAP's units.
7. **Produce the required diagnostics** (these become real findings in the paper, not just plumbing):
   - **Plausibility check**: for both standard LIME and Manifold-LIME, compute each generated
     neighbor's nearest-neighbor distance to a real training row. Manifold-LIME's neighbors should
     sit much closer to real data than standard LIME's — plot this as a comparison.
   - **Local surrogate fidelity**: report the weighted R² of the Ridge fit — how well does the simple
     linear model actually match the real model's behavior in this neighborhood?
   - **Stability**: rerun with 5 different random seeds, report the average weighted-τ agreement
     between runs, for both standard LIME and Manifold-LIME. Manifold-LIME should be more stable.

**Output file:** `manifold_lime_<model>.csv`, same column shape as the SHAP output.

### 3.4 Explainer 3 — LRD (Latent Reconstruction Decomposition) — the novel method

**⚠️ Before writing any code here, ask Angshuman whether NodeX8 already has a specification for LRD.**
The proposal describes it as "NodeX8's proprietary framework" in one sentence; it may be existing
background IP they want implemented exactly, or it may be the actual open research question handed
to this project. Don't assume — this is listed as an open question in ARCHITECTURE.md Section 8.

**If designing it from scratch, here's a concrete, defensible formulation:**

The idea: an autoencoder squashes input `x` into a small code `z`, then rebuilds it as `x̂`. Instead
of asking "which input features caused the error" the slow way (SHAP-style sampling), LRD asks it
analytically in two fast steps:

1. **Step A — which latent units caused the error?** For the bottleneck vector `z` (32 numbers),
   compute `c_k = z_k * (∂L/∂z_k)` for each latent unit `k` — "how much did unit k's activation level
   contribute to the total reconstruction error." This needs exactly one backward pass from the
   error, stopped at the bottleneck (not all the way to the input).
2. **Step B — which input features activate those latent units?** Compute the encoder's Jacobian:
   `J[k][j] = ∂z_k / ∂x_j` — a 32×271 matrix, one row per latent unit. This can be computed with
   `torch.func.jacrev` in one call, or with 32 small backward passes (one per latent unit).
3. **Step C — combine them.** Each latent unit's "blame" (`c_k` from Step A) gets distributed across
   input features in proportion to how strongly that unit depends on each feature (`J[k][j]`):
   ```
   LRD_j = sum over k of:  c_k * |J[k][j] * x_j| / (sum over j' of |J[k][j'] * x_j'|)
   ```
4. **Step D — completeness check** (this is the kind of property you can actually prove, and it
   belongs in the thesis as a small lemma): combine Step C's `LRD_j` with the direct per-feature
   reconstruction error `(x_j - x̂_j)^2` using a weight `λ`, chosen so that
   `sum_j(LRD_j) ≈ L(x)` (the total error) — i.e., the attributions "add up" to the thing being
   explained, the same completeness property SHAP is famous for.
5. **Why it's genuinely different from Adapted SHAP in 3.2**: it never needs a background/baseline
   reference and never samples or integrates along a path — it's a closed-form combination of one
   backward pass and one Jacobian. This is also exactly why it should be faster.
6. **VAE version**: use `μ` in place of `z`. Optionally scale each latent unit's contribution by
   `1/σ_k` so that latent dimensions the model itself is *uncertain* about count for less — this is
   the "richer attribution" the VAE unlocks.
7. **Measure latency** against a target of under 50ms per instance on CPU — report the real number,
   don't assume it.
8. **Validate it's not just noise**: for each instance, take the top-5 features by LRD score,
   *permute* them (replace with a random value from another row), and check that reconstruction error
   changes a lot. Do the same for the bottom-5 features and confirm error barely changes. This is a
   basic faithfulness sanity check specific to this phase (the full faithfulness suite is Phase 5).

**Output file:** `lrd_<model>.csv`, same column shape as the other two.

### 3.5 Apply all three explainers across every advanced model

For TabNet and the LSTM, LRD doesn't have an obvious "bottleneck" to back-project through. Two
options, in order of effort:
- **Easy/acceptable for now**: treat these two models' native signals (TabNet's attention masks,
  explained already in `extract_tabnet_masks.py`) as a 4th reference point, and run only Adapted SHAP
  + Manifold-LIME as the Tri-Explainer's two general-purpose members for these models.
- **Stretch goal**: adapt LRD's "Step A/B/C" logic using the last hidden layer as a stand-in for the
  bottleneck. Worth attempting after the AE/VAE version is solid and validated, not before.

### Phase 3 deliverables checklist

- [ ] Adapted SHAP implemented for AE/VAE (on reconstruction error) and TabNet/LSTM (on logit)
- [ ] Background-sensitivity study (4 baselines × weighted-τ comparison table)
- [ ] Manifold-LIME implemented, with the plausibility/fidelity/stability diagnostics
- [ ] LRD implemented and validated with the permutation sanity check, benchmarked against a <50ms
      target
- [ ] All three explainers' outputs saved in the same column format so Phase 4 can load them directly
- [ ] A latency table across all three explainers × all advanced models

### How to know Phase 3 is done

Pick one autoencoder instance. Look at the top-5 features from each of the three explainers side by
side. They don't need to be identical (disagreement is expected — that's literally what Phase 4
measures), but each one individually should point at features whose values actually look unusual for
that client when you check the raw data. If an explainer's top feature is something that looks
perfectly ordinary for that client, something in that explainer's implementation is wrong.

---

## Phase 4 — The Consensus Logic Index (CLI)

**Duration estimate:** ~2 weeks
**Depends on:** Phase 3 (needs all three explainers' outputs to exist)
**Why this phase exists:** this is the proposal's actual formal contribution — a mathematically
precise way to say "how much do these three explanations agree, for this one case" and "when they
disagree too much, stop and ask a human."

### 4.1 Implement weighted Kendall's τ correctly

**Why not Spearman?** (the proposal is explicit about this) Spearman treats a swap between the #1 and
#2 most important features the same as a swap between #100 and #101. For a regulatory explanation,
disagreeing about the *top* driver is much more serious than disagreeing somewhere in the middle of
an unimportant tail. Weighted Kendall's τ penalizes top-of-ranking disagreements more heavily.

```python
def weighted_kendall_tau(rank_a, rank_b, top_k_weight=2.0):
    """
    rank_a, rank_b: arrays of ranks (1 = most important) for the same features,
    from two different explainers.
    """
    # scipy.stats.weightedtau implements a close variant of this out of the box —
    # use it as the base, but confirm its weighting scheme matches the proposal's
    # intent (extra penalty concentrated at the top of the ranking) before relying
    # on it as-is. If it doesn't, implement a simple hyperbolic top-k weight
    # (weight ∝ 1/rank) directly over concordant/discordant pairs.
```

**Write a unit test before using this anywhere else** — this is the single most important piece of
math in the whole project, and a silent bug here invalidates every CLI number downstream:
- identical rankings → τ = 1.0
- exactly reversed rankings → τ = -1.0
- a swap at the #1/#2 position should lower τ *more* than a swap at the #99/#100 position — assert
  this explicitly with a small synthetic example

### 4.2 Implement CLI(x) exactly per the proposal's Equation 1

```
CLI(x) = [2 / (K*(K-1))] * sum over all pairs i<j of τ_w(rank(a_i(x)), rank(a_j(x)))
```

Where `K=3` (Adapted SHAP, Manifold-LIME, LRD) and `a_i(x)` is explainer `i`'s attribution vector for
instance `x`. In plain words: take the three explainers, look at every pair of them (3 pairs when
K=3), compute weighted-τ for each pair, average the three numbers.

**This must be computed for ONE model, ONE instance, across its OWN three explainers.** This is
exactly where the previous implementation (before this session's cleanup — see ARCHITECTURE.md
Section 6) went wrong: it averaged agreement across 12 *different model/explainer combinations*
instead. That measures something else entirely (do different models rely on the same features) and
is kept in this codebase as a separate, clearly-labeled secondary metric — never call it CLI.

```python
def cli(attributions_dict, top_k=10, tau_threshold=0.6):
    """
    attributions_dict: {"adapted_shap": array, "manifold_lime": array, "lrd": array}
                        — all three for the SAME model and the SAME single instance
    Returns: pairwise τ_w matrix, the averaged CLI score, and a GREEN/AMBER/RED flag
    """
```

### 4.3 Aggregate and analyze

Compute CLI(x) for every explained instance (reuse the same instances from Phase 1's local
explanations, plus a larger random sample from OOT). Then aggregate:
- **Global CLI** — the full distribution (not just the mean) per model
- **Segment-level CLI** — does agreement differ between "Power Trader" and "Dormant" clients?
- **Decile-level CLI** — is agreement *lower* in the highest-risk decile, i.e. exactly where
  disagreement matters most for a real decision?
- **OOT stability** — does CLI degrade across the OOT months as the model and data drift?

### 4.4 Calibrate the uncertainty gate

`τ_threshold` is explicitly called out in the proposal as "a calibrated value (part of the
research)" — meaning picking it IS research, not a detail to skip. Three ways to calibrate it, in
order of rigor (do at least the first):

1. **Quantile-based**: set the threshold at, say, the 10th percentile of CLI scores observed on
   Validation OOS — i.e., flag the worst 10% of explanations for review by construction.
2. **Faithfulness-linked**: using Phase 5's faithfulness tests, find the CLI level below which the
   consensus top-3 features start failing a permutation faithfulness check.
3. **Small human-rating study** (fits well with HuMICE Lab's cognitive-ergonomics angle, and is
   explicitly mentioned in the proposal): have 2–3 people independently rate ~50 generated
   explanations as "makes sense" / "doesn't make sense," and find the CLI value below which human
   agreement with the explanation drops off.

Implement the actual routing logic: any instance with `CLI(x) < tau_threshold` gets a `"REVIEW"` flag
in its output row, with a short note on *which pair of explainers disagreed most* and *on which
features* — this is what gets handed to a human reviewer.

### 4.5 Keep the old "model disagreement" metric, correctly labeled

The cross-model Kendall's W / Spearman / Jaccard analysis that was in the codebase before (see
ARCHITECTURE.md Section 6) answers a real and useful question — "do different model *architectures*
rely on the same features" — it is just not the CLI. Rebuild it cleanly under an explicit name like
`model_rashomon_index` so nobody confuses it with the real CLI again.

### Phase 4 deliverables checklist

- [ ] `weighted_kendall_tau()` with passing unit tests
- [ ] `cli()` producing a per-instance score + GREEN/AMBER/RED flag, computed from 3 same-model
      explainers
- [ ] Global, segment, decile, and OOT-stability CLI tables
- [ ] A calibrated `tau_threshold` with a written justification for how it was chosen
- [ ] Divergence analysis: for low-CLI cases, which feature(s) do the explainers disagree about most
- [ ] The old cross-model metric, kept but clearly renamed and documented as a separate thing

### How to know Phase 4 is done

Take 3 instances with CLI near 1.0 and 3 with CLI near 0 (or your threshold). For the high-CLI cases,
confirm the three explainers' top-5 feature lists genuinely overlap a lot. For the low-CLI cases,
confirm they genuinely don't. If the numbers don't match this basic sanity check, there's a bug in
the ranking or the τ computation, not a real finding.

---

## Phase 5 — The Benchmark Suite

**Duration estimate:** ~2 weeks (can overlap with the tail end of Phase 4)
**Depends on:** Phase 3, Phase 4
**Why this phase exists:** so far, nothing has *proven* the explanations are any good — only that
they can be computed and that they agree or disagree with each other. This phase is where "is this
explanation actually correct" gets tested.

### 5.1 Faithfulness

**Deletion/insertion (AOPC) test:** take an instance, rank features by attribution, then progressively
zero out (delete) the top-ranked features one at a time, recording how fast the model's output
changes. A faithful explainer's top features should cause the output to change *fast* when removed.
Repeat in reverse (insert features from zero, most important first) and confirm the output recovers
quickly. Do this for every explainer, and compare curves.

### 5.2 A synthetic ground-truth test (the strongest single experiment to include)

Build a small synthetic dataset where you, the author, *know* the true answer:
1. Take 500 real normal rows.
2. For each, corrupt exactly 3 specific, known features (e.g., multiply an amount column by 10×).
3. Run every explainer on these corrupted rows and compute **precision@3** — what fraction of the
   time are the 3 known-corrupted features actually in the explainer's reported top-3?

This directly answers "does LRD actually find the right features" without relying on circular
reasoning (you can't validate an explainer using the same model it's explaining, you need an
independent check — this is it).

### 5.3 Stability and adversarial robustness

- **Stability**: add small Gaussian noise (e.g. 1% of each feature's standard deviation) to an
  instance, recompute the explanation, and measure weighted-τ agreement with the original. A fragile
  explainer whose top features flip under tiny noise is not trustworthy.
- **Adversarial robustness**: deliberately search (a simple hill-climbing perturbation search is
  enough, doesn't need to be sophisticated) for a tiny input change that keeps the model's prediction
  the same but completely changes the explanation. Report how easy or hard this is to find for each
  explainer — Manifold-LIME should be somewhat harder to fool than standard LIME, since its
  perturbations are already constrained to the realistic manifold.

### 5.4 Runtime and memory

For every (model, explainer) pair: time 1,000 single-instance explanations, report median and
95th-percentile milliseconds, and peak memory (`tracemalloc` or similar). Compare against the
proposal's two explicit targets: LRD <50ms, general sub-100ms latency for the whole system.

### 5.5 Assemble the final matrix

One big table: rows = (model × explainer), columns = faithfulness (AOPC + synthetic precision@3),
stability, adversarial robustness score, latency (p50/p95), memory. This table is a direct
deliverable — it's "the Comprehensive Benchmark Evaluation Suite" named in the original proposal.

### Phase 5 deliverables checklist

- [ ] Deletion/insertion faithfulness curves for every (model, explainer) pair
- [ ] Synthetic ground-truth precision@3 table
- [ ] Stability-under-noise scores
- [ ] A basic adversarial-robustness comparison
- [ ] Latency (p50/p95) and memory table
- [ ] The final combined benchmark matrix

### How to know Phase 5 is done

The synthetic precision@3 numbers should all be clearly above random chance (random chance with 271
features and picking 3 of the top-3 is near 0%; even a mediocre explainer should land well above
that). If any explainer is scoring at or near random-chance level, go back and recheck its Phase 3
implementation before concluding it "doesn't work" — it's more likely a bug than a real negative
result.

---

## Phase 6 — The `claris_deep_xai` SDK

**Duration estimate:** ~2 weeks
**Depends on:** Phases 3, 4, 5 (packages what already works — don't start this before those are
solid, or you'll be repackaging constantly)
**Why this phase exists:** this is NodeX8's actual deliverable to plug into their Claris platform —
a real, installable, tested Python library, not a pile of one-off scripts.

### 6.1 Directory structure

```
claris_deep_xai/
├── pyproject.toml
├── README.md
├── claris_deep_xai/
│   ├── __init__.py
│   ├── explainers/
│   │   ├── adapted_shap.py       # from Phase 3.2
│   │   ├── manifold_lime.py      # from Phase 3.3
│   │   └── lrd.py                # from Phase 3.4
│   ├── consensus/
│   │   ├── weighted_kendall.py   # from Phase 4.1
│   │   ├── cli.py                # from Phase 4.2
│   │   └── uncertainty_gate.py   # from Phase 4.4
│   ├── benchmarks/
│   │   ├── faithfulness.py       # from Phase 5.1-5.2
│   │   └── stability.py          # from Phase 5.3
│   ├── rollup/
│   │   └── hierarchy.py          # feature_hierarchy.csv rollup, from Phase 1
│   └── utils/
│       └── data_utils.py
└── tests/
    ├── test_weighted_kendall.py
    ├── test_cli.py
    ├── test_lrd.py
    └── ...
```

### 6.2 What "production-ready" means here specifically

- **PEP-8 compliant**: run `ruff check` (or `flake8`) and fix everything it flags.
- **≥85% test coverage** (explicit proposal requirement): run `pytest --cov=claris_deep_xai`. Every
  module above needs real unit tests, not just the ones already written in Phase 4.1 for
  `weighted_kendall_tau`.
- **Type hints** on every public function signature.
- **One clear entrypoint per dataset**: a single command or function call that runs the whole
  pipeline (load model → run all 3 explainers → compute CLI → flag for review) for one dataset, so
  it's trivial for someone at NodeX8 to try it.
- **`pip install -e .`** should work cleanly from the repo root.

### Phase 6 deliverables checklist

- [ ] Package structure above, installable with `pip install -e .`
- [ ] ≥85% test coverage, verified with `pytest --cov`
- [ ] PEP-8 clean (`ruff check` passes)
- [ ] A short `README.md` with a working code example
- [ ] One end-to-end entrypoint per dataset

### How to know Phase 6 is done

Hand the repo to someone who hasn't touched it (or simulate this by deleting your local venv and
starting fresh): can they `pip install -e .`, import the package, and run the single entrypoint on
one dataset, getting a CLI score back, using only the README? If not, the packaging isn't done yet.

---

## Phase 7 — Paper, Thesis, and Handover

**Duration estimate:** ~3 weeks
**Depends on:** everything above

### 7.1 Regenerate the research report, this time from code

Write one script that reads every scorecard, every CLI output, and every benchmark table directly
from their CSV/JSON files and assembles the markdown report — never hand-type a number into a report
again (this is exactly the mistake that caused the discrepancies ARCHITECTURE.md documents from the
earlier version of this project). Keep the structure similar to the old `RESEARCH_REPORT.md` but with
every claim traceable to a file.

### 7.2 Paper draft

Target venue: IEEE TKDE or ACM TIST (both named in the proposal). Suggested structure, matching the
proposal's own framing:

1. **Introduction** — the regulatory motivation (RBI 2026 MRM, FREE-AI), the Rashomon-effect problem,
   with one real example from this project's own data
2. **Related work** — SHAP, LIME, DeepLIFT, Integrated Gradients, prior autoencoder-explanation work
   (search for this before claiming LRD is novel — a quick literature check, e.g. for
   "autoencoder anomaly explanation SHAP reconstruction error," is worth doing properly rather than
   assuming no one has tried this)
3. **Problem formulation** — formally define "the explainability consistency problem"
4. **Tri-Explainer framework** — Adapted SHAP, Manifold-LIME, LRD, each with its formula, from
   Phase 3
5. **CLI formalization** — the weighted-τ formula and the uncertainty gate, from Phase 4
6. **Experiments** — both datasets, all model families, from Phases 1–3
7. **Results** — faithfulness, latency, CLI distributions, from Phase 5
8. **Conclusion** — what was found, what's next (GNN, more datasets)

### 7.3 Public-data validation (recommended, ask Angshuman first)

Since partner data likely can't be published, consider also running the Tri-Explainer and CLI on one
public dataset named in the proposal's Work Package 1 (IEEE-CIS Fraud Detection, or the Kaggle Credit
dataset) so the paper's results can be independently reproduced by reviewers.

### 7.4 Thesis chapters

For the M.Tech thesis specifically: mathematical proofs/bounds for LRD's completeness property
(Phase 3.4 Step D), convergence behavior of the VAE training, and the human-in-the-loop validation
experiment from Phase 4.4 if that calibration route was taken.

### 7.5 Final handover

Code review with NodeX8, a project presentation, and a final walkthrough of the SDK.

### Phase 7 deliverables checklist

- [ ] `RESEARCH_REPORT.md` regenerated entirely from result files by script
- [ ] Paper draft, ready for NodeX8 co-author review
- [ ] (If pursued) results on one public dataset
- [ ] Thesis chapters
- [ ] Final presentation and code walkthrough with NodeX8

---

## Quick-Reference Timeline

| Phase | What | Duration | Depends on | Status |
|---|---|---|---|---|
| 0 | Fix and freeze | — | — | ✅ **Done** |
| 1 | Local explanations, DemandForecast cohorts | ~2 weeks | Phase 0 | ✅ **Done** |
| 2 | Autoencoder rebuild, VAE, real LSTM | ~3 weeks | Phase 0 | ✅ **Done** |
| 3 | Adapted SHAP, Manifold-LIME, LRD | ~4–5 weeks | Phase 2 | ✅ **Done** |
| 4 | CLI, weighted τ, uncertainty gate | ~2 weeks | Phase 3 | |
| 5 | Faithfulness, stability, latency benchmarks | ~2 weeks | Phases 3–4 | |
| 6 | `claris_deep_xai` SDK packaging | ~2 weeks | Phases 3–5 | |
| 7 | Paper, thesis, handover | ~3 weeks | Everything | |

**Total: roughly 18–19 weeks** from here, which fits comfortably inside the proposal's 6-month term
(with Phase 0 already behind you) and leaves some buffer. Phases 1 and 2 can run in parallel if more
than one person is working on this. Phase 3 is the one phase that should not be rushed or
parallelized carelessly — it's the actual research contribution everything else depends on.

**One rule that matters more than the schedule:** if a later phase's numbers don't make intuitive
sense (an explainer scoring at random chance, a CLI gate flagging either 0% or 100% of cases, a
latency number that seems impossibly fast), stop and check for a bug before writing it up as a
finding. This project's history (see ARCHITECTURE.md Section 6) already includes one round of
reporting numbers that turned out not to hold up — the fix each time was going back to the file that
produced the number and checking it by hand.
