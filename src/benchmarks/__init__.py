"""
Benchmark suite for the Tri-Explainer (ROADMAP.md Phase 5).

  suite.py         one uniform interface over every explainer variant, so every test below is
                   applied identically (explain(name, U, seed) -> attributions)
  faithfulness.py  5.1  deletion / insertion curves (AOPC) against a random-ranking control
  synthetic.py     5.2  ground-truth tests: (a) 3 known corrupted features on real normal rows
                        (autoencoder family); (b) a planted-signal MLP whose true drivers are known
                        (stand-in for supervised models, where "the right answer" is otherwise unknowable)
  robustness.py    5.3  stability under 1% noise, and a hill-climbing adversarial search
  runtime.py       5.4  latency (p50/p95) over many single-instance explanations, and peak memory

DECISION RULES FIXED BEFORE ANY PHASE 5 RESULT WAS SEEN (written into this file first)
------------------------------------------------------------------------------------
Everything below was decided after only a PILOT of the corruption design itself (does corrupting 3
features by 10x move the model, and is the trivial "largest |z|" baseline neither perfect nor useless?
-> error rose in 100% of rows, median shift 1.3 sigma, trivial baseline 46% vs chance 1.1%). No
explainer was evaluated in that pilot.

 R1  LRD's blend lambda (open since Phase 3): the synthetic corruption rows are split at random into a
     TUNING half and an EVALUATION half. lambda in {0, 0.5, 1} is chosen as the one with the highest
     precision@3 on the TUNING half at 10x severity; if the best is within 0.02 of lambda=0.5, keep 0.5.
     AE and VAE are decided separately; if they pick different values the project default stays 0.5 and
     the disagreement is reported. All numbers quoted for the choice are from the EVALUATION half.
 R2  Stochastic integer rounding in Manifold-LIME (candidate fix for the low-count blind spot found in
     Phase 4): recommended for adoption only if, at 10x severity on the TUNING half, it raises
     precision@3 by >= 0.03 over deterministic rounding on BOTH the autoencoder and the VAE. The Phase 3
     primary variants are NOT changed in this phase regardless; the outcome is a recommendation.
 R3  Sanity gate: every explainer's precision@3 must be well above chance (3/p). Anything within 2x
     chance is investigated as a probable bug before being reported as a result.
 R4  Hypothesis to test, either outcome reported: Manifold-LIME is harder to fool than vanilla LIME
     (roadmap 5.3).
 R5  Severities: the headline is the roadmap's 10x multiplier; 100x is reported alongside as the
     clearly-detectable case. Both are always reported.
 R6  Adversarial search constraints. Prediction change <= 2% of the standard deviation of the explained scalar
     across the explained instances; per-feature perturbation <= eps_in, where eps_in is the LARGEST of
     (0.05, 0.02, 0.01) preprocessed units for which at least 20% of random proposals satisfy the prediction
     constraint on the median instance. That choice looks only at the model's output, never at any explainer.
     Same budget (30 iterations x 4 candidates) and same constraints for every explainer of a model.
 R8  Planted-signal driver features must have no near-twin: |rho| < 0.8 with every other feature. If a dataset has
     fewer than 10 such candidates (DemandForecast has 5, because its features are highly redundant), the bound is
     relaxed to 0.9 for that dataset. Decided from feature redundancy alone, before any explainer was run on it.
 R7  Known biases of the replacement operators (documented in ARCHITECTURE.md, Phase 3) are carried
     forward: faithfulness is read against the random control and across BOTH operators, never from one.
"""
