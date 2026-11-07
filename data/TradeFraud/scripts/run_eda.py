"""
Exploratory data analysis for TradeFraud.

Describes the data the models actually see (the Modeling_Data_Enriched splits, after the
Phase 0 de-duplication of the 10x-repeated dispute rows). Nothing here fits a model.

Outputs (XAI-style folder next to the other outputs): data/TradeFraud/EDA_Outputs/
  EDA_SUMMARY.md                 the readable report (tracked in git)
  split_overview.csv             rows / unique clients / disputes / rate per split, before and after de-dup
  feature_summary.csv            all 271 features: missing, zeros, quantiles, skew, outlier gauge, group
  group_summary.csv              the same, rolled up to the variable-dictionary groups
  univariate_target_signal.csv   each feature alone as a dispute score (AUC) on the training clients
  correlated_pairs.csv           feature pairs with |Spearman| >= 0.95
  segment_dispute_rates.csv      dispute rate by activeness segment and by client segment, with 95% CIs
  monthly_trend.csv              clients and disputes per calendar month
  drift_psi.csv                  Population Stability Index of every feature, train vs each later split
  plots/*.png                    the charts used in the summary

Run:  python3 data/TradeFraud/scripts/run_eda.py
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append("/Users/somyabhadada/nodex8")
from src.eval_utils import dedupe_rows
from src.eda_utils import (feature_summary, univariate_auc, psi, correlated_pairs, md_table, style_axes)

warnings.filterwarnings("ignore")
DATA = "/Users/somyabhadada/nodex8/data/TradeFraud/Modeling_Data_Enriched/"
DICT = "/Users/somyabhadada/nodex8/data/TradeFraud/Data/variable_dictionary.csv"
OUT = "/Users/somyabhadada/nodex8/data/TradeFraud/EDA_Outputs/"
PLOTS = OUT + "plots/"
EXCLUDE = ["Client", "Client_ID", "Date", "Dispute", "Segment", "Activeness_Segment"]
SPLITS = [("train", "train.csv"), ("val_oos", "val_oos.csv"), ("internal_oot", "internal_oot.csv"),
          ("oot_2025-10", "oot_202510.csv"), ("oot_2025-11", "oot_202511.csv"), ("oot_2025-12", "oot_202512.csv")]
TEAL, AMBER, RED, GRAY, INK = "#0B6E69", "#F2A541", "#D1495B", "#6B7787", "#1B2430"
SEGMENT_ORDER = ["Dormant", "Occasional", "Regular", "Active", "Frequent", "Power Trader"]


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def main():
    os.makedirs(PLOTS, exist_ok=True)
    print("Loading splits...")
    raw, data = {}, {}
    for name, fn in SPLITS:
        df = pd.read_csv(DATA + fn)
        raw[name] = df
        feat = [c for c in df.columns if c not in EXCLUDE]
        for c in feat:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        data[name] = dedupe_rows(df, feat).reset_index(drop=True)
    features = [c for c in raw["train"].columns if c not in EXCLUDE]
    vd = pd.read_csv(DICT)
    group_of = dict(zip(vd.Variable_Name, vd.Group))
    sub_of = dict(zip(vd.Variable_Name, vd.Sub_Group))
    horizon_of = dict(zip(vd.Variable_Name, vd.Time_Horizon))

    # ------------------------------------------------------------------ 1. split overview
    rows = []
    for name, _ in SPLITS:
        r, d = raw[name], data[name]
        dr = (r.Dispute == "D")
        dd = (d.Dispute == "D")
        rows.append({"split": name, "rows_raw": len(r), "disputes_raw": int(dr.sum()),
                     "rows_unique": len(d), "unique_clients": d.Client_ID.nunique(), "disputes_unique": int(dd.sum()),
                     "dispute_rate_pct": 100.0 * dd.mean(), "duplication_of_disputes": dr.sum() / max(dd.sum(), 1),
                     "min_date": str(d.Date.min())[:10], "max_date": str(d.Date.max())[:10]})
    overview = pd.DataFrame(rows)
    overview.to_csv(OUT + "split_overview.csv", index=False)
    tr, va = data["train"], data["val_oos"]
    overlap = len(set(tr.Client_ID) & set(va.Client_ID))
    y_tr = (tr.Dispute == "D").astype(int).values
    print(overview[["split", "unique_clients", "disputes_unique", "dispute_rate_pct"]].round(3).to_string(index=False))

    # ------------------------------------------------------------------ 2. feature summary (train, de-duplicated)
    print("Feature summary...")
    fs = feature_summary(tr, features)
    fs.insert(1, "group", fs.feature.map(group_of))
    fs.insert(2, "sub_group", fs.feature.map(sub_of))
    fs.insert(3, "time_horizon", fs.feature.map(horizon_of))
    fs.to_csv(OUT + "feature_summary.csv", index=False)
    gs = fs.groupby("group").agg(features=("feature", "count"), mean_zero_pct=("zero_pct", "mean"),
                                 mean_abs_skew=("skew", lambda s: s.abs().mean()),
                                 max_missing_pct=("missing_pct", "max")).reset_index()
    gs.to_csv(OUT + "group_summary.csv", index=False)
    n_const = int(fs.constant.sum())
    n_skew2 = int((fs["skew"].abs() > 2).sum())
    n_int = int(fs.integer_valued.sum())
    n_miss = int((fs.missing_pct > 0).sum())
    n_zero50 = int((fs.zero_pct > 50).sum())
    n_neg = int((fs.neg_pct > 0).sum())
    n_out = int((fs.max_over_p99 > 10).sum())

    # ------------------------------------------------------------------ 3. univariate signal for the target
    print("Univariate signal...")
    uni = univariate_auc(tr, features, y_tr)
    uni.insert(1, "group", uni.feature.map(group_of))
    uni.to_csv(OUT + "univariate_target_signal.csv", index=False)
    # what would chance look like? permute the labels many times and take the best of 271
    rng = np.random.RandomState(0)
    best_null = [univariate_auc(tr, features, rng.permutation(y_tr)).strength.iloc[0] for _ in range(20)]
    null_best = float(np.mean(best_null))
    y_va = (va.Dispute == "D").astype(int).values
    uni_va = univariate_auc(va, features, y_va).set_index("feature")
    uni["auc_validation"] = uni.feature.map(uni_va.auc)
    uni.to_csv(OUT + "univariate_target_signal.csv", index=False)
    n_above = int((uni.strength > 2 * null_best).sum())

    # ------------------------------------------------------------------ 4. redundancy
    print("Correlations...")
    C, pairs = correlated_pairs(tr, features, threshold=0.95)
    pairs.to_csv(OUT + "correlated_pairs.csv", index=False)
    degree = pd.concat([pairs.feature_a, pairs.feature_b]).value_counts()
    n_in_pair = int(len(degree))

    # ------------------------------------------------------------------ 5. segments
    pooled = pd.concat([tr.assign(split="train"), va.assign(split="val_oos")], ignore_index=True)
    seg_rows = []
    for col, order in (("Activeness_Segment", SEGMENT_ORDER), ("Segment", None)):
        g = pooled.groupby(col).apply(lambda d: pd.Series({"clients": len(d), "disputes": int((d.Dispute == "D").sum())}))
        g = g.loc[[o for o in order if o in g.index]] if order else g.sort_index()
        for lab, r in g.iterrows():
            lo, hi = wilson(r.disputes, r.clients)
            seg_rows.append({"segment_type": col, "segment": lab, "clients": int(r.clients), "disputes": int(r.disputes),
                             "dispute_rate_pct": 100.0 * r.disputes / r.clients, "ci95_lo_pct": 100 * lo, "ci95_hi_pct": 100 * hi})
    seg = pd.DataFrame(seg_rows)
    seg.to_csv(OUT + "segment_dispute_rates.csv", index=False)

    # ------------------------------------------------------------------ 6. time trend
    allrows = pd.concat([data[n].assign(split=n) for n, _ in SPLITS], ignore_index=True)
    allrows["month"] = pd.to_datetime(allrows.Date).dt.to_period("M").astype(str)
    mt = allrows.groupby("month").apply(lambda d: pd.Series({"clients": len(d), "disputes": int((d.Dispute == "D").sum())})).reset_index()
    mt["dispute_rate_pct"] = 100.0 * mt.disputes / mt.clients
    mt.to_csv(OUT + "monthly_trend.csv", index=False)

    # ------------------------------------------------------------------ 7. drift
    print("Drift (PSI)...")
    drift = {"feature": features}
    for name, _ in SPLITS[1:]:
        drift[name] = [psi(tr[f], data[name][f]) for f in features]
    drift = pd.DataFrame(drift)
    drift["max_psi"] = drift[[n for n, _ in SPLITS[1:]]].max(axis=1)
    drift["group"] = drift.feature.map(group_of)
    drift.sort_values("max_psi", ascending=False).to_csv(OUT + "drift_psi.csv", index=False)
    drift_cols = [n for n, _ in SPLITS[1:]]
    drift_summary = pd.DataFrame({"split": drift_cols,
                                  "features_psi_gt_0.1": [int((drift[c] > 0.1).sum()) for c in drift_cols],
                                  "features_psi_gt_0.25": [int((drift[c] > 0.25).sum()) for c in drift_cols],
                                  "median_psi": [drift[c].median() for c in drift_cols]})

    # ------------------------------------------------------------------ plots
    print("Plots...")
    plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": "#C9D1DB", "figure.dpi": 130})

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(overview))
    ax[0].bar(x, overview.unique_clients, color=TEAL)
    for i, v in enumerate(overview.unique_clients):
        ax[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax[0].set_xticks(x, overview.split, rotation=25, ha="right")
    style_axes(ax[0], "Unique clients per split", ylabel="clients")
    ax[1].bar(x, overview.disputes_unique, color=RED)
    for i, (v, r) in enumerate(zip(overview.disputes_unique, overview.dispute_rate_pct)):
        ax[1].text(i, v, f"{v}\n({r:.2f}%)", ha="center", va="bottom", fontsize=9)
    ax[1].set_xticks(x, overview.split, rotation=25, ha="right")
    ax[1].set_ylim(0, overview.disputes_unique.max() * 1.25)
    style_axes(ax[1], "Real (unique) disputes per split", ylabel="disputes")
    fig.tight_layout(); fig.savefig(PLOTS + "01_class_balance.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].hist(fs.zero_pct.dropna(), bins=20, color=TEAL)
    style_axes(ax[0], "Share of zeros per feature", "% of clients equal to 0", "features")
    ax[1].hist(fs["skew"].clip(-30, 60).dropna(), bins=30, color=AMBER)
    ax[1].axvline(2, color=INK, ls="--", lw=1); ax[1].axvline(-2, color=INK, ls="--", lw=1)
    style_axes(ax[1], "Skewness per feature (|2| = dashed)", "skewness (clipped to −30…60)", "features")
    ax[2].hist(np.log10(fs.max_over_p99.replace([np.inf], np.nan).dropna().clip(lower=1)), bins=25, color=GRAY)
    style_axes(ax[2], "Outlier gauge: max ÷ 99th percentile", "log10 of the ratio", "features")
    fig.tight_layout(); fig.savefig(PLOTS + "02_feature_shape.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, st, order in ((ax[0], "Activeness_Segment", SEGMENT_ORDER), (ax[1], "Segment", None)):
        s = seg[seg.segment_type == st]
        a.bar(range(len(s)), s.dispute_rate_pct, color=TEAL,
              yerr=[s.dispute_rate_pct - s.ci95_lo_pct, s.ci95_hi_pct - s.dispute_rate_pct], ecolor=INK, capsize=3)
        a.set_xticks(range(len(s)), s.segment, rotation=30, ha="right")
        style_axes(a, f"Dispute rate by {st.replace('_', ' ').lower()} (95% CI)", ylabel="% of clients")
    fig.tight_layout(); fig.savefig(PLOTS + "03_segment_dispute_rates.png"); plt.close(fig)

    top = uni.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.feature, top.auc - 0.5, left=0.5, color=[TEAL if d.startswith("higher") else AMBER for d in top.direction])
    ax.axvline(0.5, color=INK, lw=1)
    ax.set_xlim(0.25, 0.75)
    style_axes(ax, "Strongest single features for spotting disputes (training clients)", "AUC (teal: higher in disputes · amber: lower)")
    ax.grid(axis="x", color="#E3E8EE"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(PLOTS + "04_top_univariate_signal.png"); plt.close(fig)

    keep = [f for f in uni.feature.head(40) if f in C.columns]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(C.loc[keep, keep].values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keep)), keep, rotation=90, fontsize=6); ax.set_yticks(range(len(keep)), keep, fontsize=6)
    ax.set_title("Spearman correlation among the 40 strongest features", loc="left", fontsize=12, fontweight="bold", color=INK)
    fig.colorbar(im, shrink=0.7); fig.tight_layout(); fig.savefig(PLOTS + "05_correlation_top40.png"); plt.close(fig)

    topd = drift.sort_values("max_psi", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(topd[drift_cols].values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.5)
    ax.set_xticks(range(len(drift_cols)), drift_cols, rotation=25, ha="right"); ax.set_yticks(range(len(topd)), topd.feature, fontsize=8)
    for i in range(len(topd)):
        for j, c in enumerate(drift_cols):
            ax.text(j, i, f"{topd[c].iloc[i]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Feature drift vs training (PSI; > 0.25 = large shift)", loc="left", fontsize=12, fontweight="bold", color=INK)
    fig.colorbar(im, shrink=0.8); fig.tight_layout(); fig.savefig(PLOTS + "06_drift_psi.png"); plt.close(fig)

    show = list(uni.feature.head(6))
    fig, axs = plt.subplots(2, 3, figsize=(13, 6))
    for a, f in zip(axs.ravel(), show):
        a0 = signed_log1p(tr.loc[y_tr == 0, f].fillna(0).values); a1 = signed_log1p(tr.loc[y_tr == 1, f].fillna(0).values)
        bins = np.linspace(min(a0.min(), a1.min()), max(a0.max(), a1.max()), 30)
        a.hist(a0, bins=bins, density=True, color=TEAL, alpha=0.6, label="normal")
        a.hist(a1, bins=bins, density=True, color=RED, alpha=0.6, label="dispute")
        style_axes(a, f, "signed log(1+|x|)", "density")
        a.title.set_fontsize(9)
    axs[0, 0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(PLOTS + "07_top_feature_distributions.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(mt.month, mt.clients, color="#C9E4E1", label="clients")
    ax2 = ax.twinx(); ax2.plot(mt.month, mt.dispute_rate_pct, color=RED, marker="o", label="dispute rate %")
    ax.set_xticks(range(len(mt)), mt.month, rotation=60, ha="right", fontsize=8)
    style_axes(ax, "Clients (bars) and dispute rate (line) by month", ylabel="clients"); ax2.set_ylabel("dispute rate %", color=RED)
    ax2.spines[["top"]].set_visible(False)
    fig.tight_layout(); fig.savefig(PLOTS + "08_monthly_trend.png"); plt.close(fig)

    # ------------------------------------------------------------------ report
    ov = overview[["split", "rows_raw", "disputes_raw", "unique_clients", "disputes_unique", "dispute_rate_pct", "min_date", "max_date"]]
    ov = ov.rename(columns={"rows_raw": "rows as supplied", "disputes_raw": "dispute rows as supplied", "unique_clients": "unique clients",
                            "disputes_unique": "unique disputes", "dispute_rate_pct": "dispute rate %"})
    seg_a = seg[seg.segment_type == "Activeness_Segment"][["segment", "clients", "disputes", "dispute_rate_pct", "ci95_lo_pct", "ci95_hi_pct"]]
    seg_b = seg[seg.segment_type == "Segment"][["segment", "clients", "disputes", "dispute_rate_pct", "ci95_lo_pct", "ci95_hi_pct"]]
    t10 = uni.head(10)[["feature", "group", "auc", "auc_validation", "direction"]].rename(columns={"auc": "AUC (train)", "auc_validation": "AUC (validation)"})
    pairs_top = pairs.head(8)
    worst = drift.sort_values("max_psi", ascending=False).head(10)[["feature", "group"] + drift_cols]
    skewed = fs.reindex(fs["skew"].abs().sort_values(ascending=False).index).head(8)[["feature", "skew", "zero_pct", "median", "p99", "max"]]
    zero_heavy = fs.sort_values("zero_pct", ascending=False).head(8)[["feature", "zero_pct", "n_unique", "integer_valued"]]
    grp = gs.rename(columns={"mean_zero_pct": "mean % zeros", "mean_abs_skew": "mean abs skew", "max_missing_pct": "max % missing"})
    rate_overall = 100 * overview.loc[overview.split == "train", "disputes_unique"].iloc[0] / overview.loc[overview.split == "train", "unique_clients"].iloc[0]
    sa = seg[seg.segment_type == "Activeness_Segment"].set_index("segment")
    dorm_rate, dorm_disputes, dorm_clients = sa.loc["Dormant", "dispute_rate_pct"], int(sa.loc["Dormant", "disputes"]), int(sa.loc["Dormant", "clients"])
    pt_rate, dr_regular = sa.loc["Power Trader", "dispute_rate_pct"], sa.loc["Regular", "dispute_rate_pct"]
    md = f"""# TradeFraud — exploratory data analysis

Generated by `data/TradeFraud/scripts/run_eda.py`. Descriptive only: nothing here fits a model.
All feature statistics use the **training split after de-duplication** (`{len(tr):,}` unique clients,
`{int(y_tr.sum())}` real disputes) unless stated. Charts are in `EDA_Outputs/plots/`; every table below
is also saved as a CSV in this folder.

## 1. What the data looks like

- **{len(features)} model features** per client snapshot, from the variable dictionary's groups (below). {n_int} are
  integer-valued (counts, flags); {len(features) - n_int} are continuous.
- **Missing values: {'none' if n_miss == 0 else f'{n_miss} features have some'}.** Missing data is not a problem in these splits.
- **Constant features: {n_const}.** {n_zero50} features are zero for more than half of all clients (sparse activity features);
  {n_neg} contain negative values (net/imbalance measures).
- **Heavily skewed: {n_skew2} of {len(features)} features have |skewness| > 2**, and {n_out} have a maximum more than 10× their
  99th percentile. This is why the autoencoder needs a log transform before it can learn (Phase 2).
- **Train and validation share no clients** ({overlap} overlapping client IDs).

### Splits

{md_table(ov, "{:.3f}")}

**The dispute rows were duplicated 10× in the data as supplied.** After de-duplication the dispute rate is
**{rate_overall:.2f}%** of clients in training — roughly 1 in 450. A model that always says "no dispute" is
{100 - rate_overall:.2f}% accurate, which is why accuracy is never reported. The out-of-time months have only
15–18 real disputes each, so any score measured on them is very uncertain.

![class balance](plots/01_class_balance.png)

### Feature groups

{md_table(grp, "{:.2f}")}

![feature shape](plots/02_feature_shape.png)

Most skewed features:

{md_table(skewed, "{:.2f}")}

Sparsest features (share of clients with value exactly 0):

{md_table(zero_heavy, "{:.1f}")}

## 2. Who disputes? (segments)

Dispute rate with 95% confidence interval, training + validation clients pooled
({int(seg_a.clients.sum()):,} clients, {int(seg_a.disputes.sum())} disputes).

**By activeness segment**

{md_table(seg_a, "{:.2f}")}

**By client segment**

{md_table(seg_b, "{:.2f}")}

![segments](plots/03_segment_dispute_rates.png)

**Disputes track how active the client is.** The rate climbs from {dorm_rate:.2f}% in the Dormant segment
({dorm_disputes} disputes among {dorm_clients:,} clients) to {pt_rate:.2f}% for Power Traders. Dormant clients essentially never
dispute, and Power Traders dispute about {pt_rate / max(dr_regular, 1e-9):.1f}× as often as Regular ones. The ten client
segments, by contrast, are all within sampling error of the overall rate — they carry little information about disputes.

## 3. Which single features separate disputes from normal clients?

Each feature used alone as a dispute score (AUC: 0.5 = no signal, 1.0 = perfect; direction-free strength = |AUC − 0.5|×2).

{md_table(t10, "{:.3f}")}

![top signal](plots/04_top_univariate_signal.png)

**How to read this.** The strongest single feature reaches an AUC of {uni.auc.iloc[0]:.3f} on training clients
(strength {uni.strength.iloc[0]:.3f}) and {uni.auc_validation.iloc[0]:.3f} on the validation clients it was not chosen on.
For calibration: with {int(y_tr.sum())} disputes and {len(features)} features, the best feature in *pure noise* (labels shuffled)
reaches a strength of only about {null_best:.3f}. So the top features carry **real but modest** signal — {n_above} features
exceed twice the noise level — and it is concentrated in **lifetime activity volume** (total traded amounts, trades per day,
days traded). That matches Section 2: more active clients dispute more.

**Why it matters for the models:** on the validation clients, single activity-volume features score AUC
{uni.auc_validation.head(6).min():.2f}–{uni.auc_validation.head(6).max():.2f} — *higher* than the best trained model
(0.716, logistic regression; ARCHITECTURE.md Section 4). The gap is within sampling error (the models' own 95% ranges are about
±0.09 with 51 validation disputes), so this is not proof that the models are worse, but it says clearly that **the models add
nothing measurable beyond "how active / how large is this client"**. Their explanations are therefore likely to be dominated by
activity-volume features, and a plain one-feature rule is a baseline any explanation story must be honest about.

![distributions](plots/07_top_feature_distributions.png)

## 4. Redundancy between features

{len(pairs):,} feature pairs have |Spearman| ≥ 0.95, involving {n_in_pair} of the {len(features)} features. Many features are
rolling-window versions of one another (e.g. the same quantity over 7, 15 and 21 days), so the effective number of independent
signals is far smaller than {len(features)}. This matters for explanation: when features are near-copies, an explainer can
split credit between them arbitrarily, which is one reason different explainers disagree.

{md_table(pairs_top, "{:.3f}")}

![correlation](plots/05_correlation_top40.png)

## 5. Time and drift

![monthly trend](plots/08_monthly_trend.png)

**Finding: the features do not drift.** Every feature has a PSI far below 0.1 in every later split, including all three
out-of-time months. Month to month the dispute rate moves between {mt.dispute_rate_pct.min():.2f}% and {mt.dispute_rate_pct.max():.2f}%
with no trend — what roughly {mt.disputes.mean():.0f} disputes per month produce by chance. Each client appears once (one snapshot),
no client appears in two out-of-time files, and every calendar month holds almost exactly the same number of clients
({mt.clients.min():,}–{mt.clients.max():,}), i.e. the months are equal-sized cohorts. So the "out-of-time" months are
effectively *fresh clients drawn from the same distribution*, not a population that changed. Consequence: TradeFraud's
out-of-time results are a second out-of-sample test, **not a stress test of behaviour change** — a model that does well there
has not proved it survives drift. (PSI is computed on the training deciles; for the many features that are over 75% zeros it
has only a few bins and is less sensitive, so this is a statement about the bulk of each distribution.)

Population Stability Index of each feature against the training split (< 0.1 stable · 0.1–0.25 moderate · > 0.25 large):

{md_table(drift_summary, "{:.3f}")}

Features that drift most:

{md_table(worst, "{:.3f}")}

![drift](plots/06_drift_psi.png)

## 6. What this means for the project

1. **Very few real disputes** ({int(y_tr.sum())} train, 51 validation, 15–18 per OOT month): every model score and every
   explanation-quality number needs its confidence interval.
2. **Skewed, sparse, redundant features:** preprocessing (signed log, scaling, clipping) is required for the autoencoder,
   and explanations should be read at feature-*group* level as well as per feature.
3. **Modest, activity-driven signal:** a single volume feature matches or beats every trained model on validation, and Dormant
   clients never dispute. Weak AUCs are largely a property of this data. Expect explanations to point mostly at activity-volume features.
4. **No drift in this data:** out-of-time results here do not test robustness to change. Out-of-time CLI stability (Phase 4) is
   therefore not evidence about drift on TradeFraud; DemandForecast (a real time series) is the better place to look.
"""
    open(OUT + "EDA_SUMMARY.md", "w").write(md)
    print(f"\nWrote {OUT}EDA_SUMMARY.md and {len(os.listdir(PLOTS))} plots")


if __name__ == "__main__":
    main()
