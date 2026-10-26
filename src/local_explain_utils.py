"""
Shared helpers for Phase 1's local (per-instance) explainability work
(see ROADMAP.md, Phase 1.1).

Two things live here, used identically by both datasets so the selection
logic and the text-explanation wording can't drift apart between them:

1. Case selection — picking 15 representative instances to explain in
   detail, instead of explaining either "everything" (too slow, too much to
   read) or an arbitrary random sample (uninformative — you want the cases
   that actually test the explanation, not typical ones).
2. Turning a feature-attribution vector into a short, human-readable
   sentence — the actual "why was this flagged" text a regulator or a
   customer-facing adverse-action notice would show.
"""
import numpy as np


def select_classification_cases(pred_prob, true_label, threshold, n_each=5):
    """
    Pick representative instances for a binary classifier (TradeFraud).

    Three buckets, no instance repeated across buckets:
      - "high_risk":     the n_each instances with the highest predicted
                          probability. "Is the model's most confident call
                          sensible?"
      - "borderline":    the n_each instances whose predicted probability is
                          closest to THIS MODEL'S OWN decision threshold
                          (not 0.5 — a model whose threshold sits at 0.72 has
                          its genuinely uncertain cases near 0.72). "Do the
                          explainers still agree when the model itself is
                          unsure?"
      - "misclassified": false positives and false negatives at that same
                          threshold, mixed as evenly as the data allows.
                          "What did the model get wrong, and why?" — more
                          informative than a 4th bucket of confident correct
                          predictions would be.

    Parameters
    ----------
    pred_prob, true_label : 1-D array-like, same length, aligned by position
    threshold : float — this model's Threshold_Used from Phase 0 (chosen
                 once on Validation OOS; see ARCHITECTURE.md Section 7)
    n_each : how many instances per bucket (default 5, for 15 total)

    Returns
    -------
    dict of {bucket_name: np.ndarray of positional indices into pred_prob}
    """
    pred_prob = np.asarray(pred_prob, dtype=float)
    true_label = np.asarray(true_label, dtype=int)
    n = len(pred_prob)
    pred_label = (pred_prob >= threshold).astype(int)
    used = set()

    def take(order, limit):
        out = []
        for i in order:
            i = int(i)
            if i in used:
                continue
            out.append(i)
            used.add(i)
            if len(out) == limit:
                break
        return np.array(out, dtype=int)

    high_risk = take(np.argsort(-pred_prob), n_each)
    borderline = take(np.argsort(np.abs(pred_prob - threshold)), n_each)

    fp_idx = [i for i in range(n) if pred_label[i] == 1 and true_label[i] == 0 and i not in used]
    fn_idx = [i for i in range(n) if pred_label[i] == 0 and true_label[i] == 1 and i not in used]
    # False negatives are usually much rarer than false positives in a
    # fraud-style imbalanced problem, so take them first; fill the rest
    # with false positives, then with any remaining false negatives.
    n_fn_first = min(len(fn_idx), (n_each + 1) // 2)
    misclassified = list(fn_idx[:n_fn_first])
    remaining = n_each - len(misclassified)
    misclassified.extend(fp_idx[:remaining])
    remaining = n_each - len(misclassified)
    if remaining > 0:
        misclassified.extend(fn_idx[n_fn_first:n_fn_first + remaining])
    misclassified = np.array(misclassified[:n_each], dtype=int)
    used.update(misclassified.tolist())

    return {"high_risk": high_risk, "borderline": borderline, "misclassified": misclassified}


def select_regression_cases(pred_value, actual_value, n_each=5):
    """
    Pick representative instances for a regressor (DemandForecast).
    The classification concepts above don't map directly onto regression
    (there's no decision threshold, no false positive/negative), so the
    three buckets here are regression's closest honest equivalents:

      - "high_value":  the n_each instances with the highest PREDICTED
                        demand. "Does the model's biggest call make sense?"
      - "borderline":   the n_each instances whose prediction is closest to
                        the median prediction across the dataset — the
                        "typical, ambiguous" case, playing the same role
                        the classification threshold plays above.
      - "high_error":   the n_each instances with the largest |actual -
                        predicted| error — the regression equivalent of
                        "misclassified": what did the model get most wrong,
                        and does the explanation reveal why?
    """
    pred_value = np.asarray(pred_value, dtype=float)
    actual_value = np.asarray(actual_value, dtype=float)
    used = set()

    def take(order, limit):
        out = []
        for i in order:
            i = int(i)
            if i in used:
                continue
            out.append(i)
            used.add(i)
            if len(out) == limit:
                break
        return np.array(out, dtype=int)

    high_value = take(np.argsort(-pred_value), n_each)
    median_pred = float(np.median(pred_value))
    borderline = take(np.argsort(np.abs(pred_value - median_pred)), n_each)
    high_error = take(np.argsort(-np.abs(actual_value - pred_value)), n_each)

    return {"high_value": high_value, "borderline": borderline, "high_error": high_error}


def top_features_signed(attribution_vector, feat_cols, k=3):
    """
    Return the top-k (feature_name, signed_attribution) pairs by absolute
    value — the features that mattered most for this one instance, with
    their direction (pushed the prediction up or down) preserved.
    """
    attribution_vector = np.asarray(attribution_vector, dtype=float)
    order = np.argsort(-np.abs(attribution_vector))[:k]
    return [(feat_cols[i], float(attribution_vector[i])) for i in order]


def text_explanation_classification(top_feats, pred_prob, threshold,
                                     positive_label="Dispute", negative_label="Normal"):
    """
    Build the actual "adverse action notice" sentence: which features pushed
    this one case toward or away from being flagged, in order of importance.
    """
    verdict = f"classified as **{positive_label}**" if pred_prob >= threshold \
        else f"classified as **{negative_label}**"
    parts = []
    for i, (feat, val) in enumerate(top_feats, 1):
        direction = f"increased the predicted risk" if val > 0 else f"decreased the predicted risk"
        parts.append(f"({i}) `{feat}` {direction}")
    return (f"This case was {verdict} (predicted probability {pred_prob:.3f} against a "
            f"decision threshold of {threshold:.3f}), primarily because: " + "; ".join(parts) + ".")


def text_explanation_regression(top_feats, pred_value):
    """Build the equivalent sentence for a regression prediction."""
    parts = []
    for i, (feat, val) in enumerate(top_feats, 1):
        direction = "pushed the predicted value up" if val > 0 else "pushed the predicted value down"
        parts.append(f"({i}) `{feat}` {direction}")
    return (f"The predicted value of {pred_value:.3f} was driven primarily by: "
            + "; ".join(parts) + ".")
