"""
Exploratory data analysis for DemandForecast.

Describes the data the models see: one row per product (UPC) per day for a single store,
Jul 2011 - Feb 2013, target QUANTITY (units sold). Nothing here fits a model.

Outputs: data/DemandForecast/EDA_Outputs/
  EDA_SUMMARY.md                 the readable report (tracked in git)
  split_overview.csv             per split: rows, products, brands, date range, demand level, zero share
  target_quantiles.csv           distribution of QUANTITY per split
  brand_summary.csv              demand level, zero share and volume share per brand
  upc_summary.csv                the same per product, with how many days each is recorded
  calendar_effects.csv           mean demand by day of week, month, holiday and promotion flags
  price_effect.csv               mean demand by price-ratio decile and by promotion status
  feature_summary.csv            the 51 model features: missing, zeros, quantiles, skew, group
  target_correlation.csv         Spearman correlation of every model feature with QUANTITY
  lag_correlation.csv            how well yesterday / last week / rolling means predict today
  correlated_pairs.csv           feature pairs with |Spearman| >= 0.95
  drift_psi.csv                  PSI of every feature (and of QUANTITY) vs the DEV split
  entity_overlap.csv             products / brands in each split that DEV never saw
  plots/*.png                    the charts used in the summary

Run:  python3 data/DemandForecast/scripts/run_eda.py
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.append("/Users/somyabhadada/nodex8")
from src.eda_utils import feature_summary, psi, correlated_pairs, md_table, style_axes

warnings.filterwarnings("ignore")
ROOT = "/Users/somyabhadada/nodex8/data/DemandForecast/"
SAMPLES = ROOT + "samples/"
OUT = ROOT + "EDA_Outputs/"
PLOTS = OUT + "plots/"
SPLITS = [("dev", "dev.csv"), ("oos", "oos.csv"), ("oot_2012-12", "oot_01_201212.csv"),
          ("oot_2013-01", "oot_02_201301.csv"), ("oot_2013-02", "oot_03_201302.csv")]
TEAL, AMBER, RED, GRAY, INK = "#0B6E69", "#F2A541", "#D1495B", "#6B7787", "#1B2430"
DOW = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}


def main():
    os.makedirs(PLOTS, exist_ok=True)
    print("Loading...")
    full = pd.read_csv(ROOT + "features.csv", parse_dates=["Date"])
    data = {n: pd.read_csv(SAMPLES + f, parse_dates=["Date"]) for n, f in SPLITS}
    hier = pd.read_csv(ROOT + "XAI_Outputs/feature_hierarchy.csv")
    features = list(hier.Feature)
    group_of = dict(zip(hier.Feature, hier.Group))
    for d in [full] + list(data.values()):
        for c in ("PROMOTION_FLG", "C_PROMOTION_FLG"):          # Y/N text -> 1/0, as in train_lstm.clean_categoricals
            d[c] = d[c].map({"Y": 1, "N": 0, "1": 1, "0": 0, 1: 1, 0: 0}).fillna(0)
        for c in features:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    dev = data["dev"]
    q = full.QUANTITY

    # ------------------------------------------------------------------ 1. splits
    rows = []
    for n, _ in SPLITS:
        d = data[n]
        rows.append({"split": n, "rows": len(d), "products": d.UPC.nunique(), "brands": d.BRAND.nunique(),
                     "min_date": str(d.Date.min())[:10], "max_date": str(d.Date.max())[:10], "mean_qty": d.QUANTITY.mean(),
                     "median_qty": d.QUANTITY.median(), "p99_qty": d.QUANTITY.quantile(0.99), "max_qty": d.QUANTITY.max(),
                     "zero_pct": 100 * (d.QUANTITY == 0).mean(), "promo_pct": 100 * d.PROMOTION_FLG_NUM.mean()})
    overview = pd.DataFrame(rows)
    overview.to_csv(OUT + "split_overview.csv", index=False)
    qt = pd.DataFrame({n: data[n].QUANTITY.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]) for n, _ in SPLITS}).T
    qt.columns = [f"p{int(c * 100)}" for c in qt.columns]
    qt.to_csv(OUT + "target_quantiles.csv")
    overdisp = float(q.var() / q.mean())

    # entities each split has that DEV never saw
    ent = []
    for n, _ in SPLITS[1:]:
        d = data[n]
        new_u, new_b = set(d.UPC) - set(dev.UPC), set(d.BRAND) - set(dev.BRAND)
        ent.append({"split": n, "products": d.UPC.nunique(), "unseen_products": len(new_u),
                    "rows_from_unseen_products_pct": 100 * d.UPC.isin(new_u).mean(),
                    "brands": d.BRAND.nunique(), "unseen_brands": len(new_b), "rows_from_unseen_brands_pct": 100 * d.BRAND.isin(new_b).mean()})
    ent = pd.DataFrame(ent)
    ent.to_csv(OUT + "entity_overlap.csv", index=False)

    # ------------------------------------------------------------------ 2. brands / products
    by_brand = full.groupby("BRAND").agg(rows=("QUANTITY", "size"), products=("UPC", "nunique"), mean_qty=("QUANTITY", "mean"),
                                         total_qty=("QUANTITY", "sum"), zero_pct=("QUANTITY", lambda s: 100 * (s == 0).mean())).reset_index()
    by_brand["volume_share_pct"] = 100 * by_brand.total_qty / by_brand.total_qty.sum()
    by_brand = by_brand.sort_values("total_qty", ascending=False)
    by_brand["split_it_sits_in"] = by_brand.BRAND.map(
        lambda b: ", ".join(n for n, _ in SPLITS if b in set(data[n].BRAND)))
    by_brand.to_csv(OUT + "brand_summary.csv", index=False)
    span = (full.Date.max() - full.Date.min()).days + 1
    by_upc = full.groupby("UPC").agg(brand=("BRAND", "first"), rows=("QUANTITY", "size"), first=("Date", "min"), last=("Date", "max"),
                                     mean_qty=("QUANTITY", "mean"), total_qty=("QUANTITY", "sum"),
                                     zero_pct=("QUANTITY", lambda s: 100 * (s == 0).mean())).reset_index()
    by_upc["days_in_life"] = (by_upc["last"] - by_upc["first"]).dt.days + 1
    by_upc["coverage_pct"] = 100 * by_upc.rows / by_upc.days_in_life
    by_upc = by_upc.sort_values("total_qty", ascending=False)
    by_upc["cum_volume_share_pct"] = 100 * by_upc.total_qty.cumsum() / by_upc.total_qty.sum()
    by_upc.to_csv(OUT + "upc_summary.csv", index=False)
    top10_share = float(by_upc.cum_volume_share_pct.iloc[9])
    n_for_half = int((by_upc.cum_volume_share_pct < 50).sum() + 1)

    # ------------------------------------------------------------------ 3. calendar, promotion, price
    cal = []
    for lab, col, mapping in (("day of week", "dayofweek_num", DOW), ("month", "month", None)):
        g = full.groupby(col).QUANTITY.agg(["mean", "size"]).reset_index()
        for _, r in g.iterrows():
            cal.append({"effect": lab, "level": mapping[int(r[col])] if mapping else int(r[col]), "mean_qty": r["mean"], "rows": int(r["size"])})
    for lab, col in (("holiday", "holiday"), ("holiday eve", "holiday_pre"), ("holiday after", "holiday_post"),
                     ("own promotion", "PROMOTION_FLG_NUM"), ("competitor promotion", "C_PROMOTION_FLG_NUM")):
        g = full.groupby(col).QUANTITY.agg(["mean", "size"]).reset_index()
        for _, r in g.iterrows():
            cal.append({"effect": lab, "level": "yes" if r[col] == 1 else "no", "mean_qty": r["mean"], "rows": int(r["size"])})
    cal = pd.DataFrame(cal)
    cal.to_csv(OUT + "calendar_effects.csv", index=False)
    promo = full.groupby("PROMOTION_FLG_NUM").QUANTITY.mean()
    promo_lift = float(promo.loc[1] / promo.loc[0])
    # promotion lift within a product, so brand mix cannot explain it
    within = full.groupby(["UPC", "PROMOTION_FLG_NUM"]).QUANTITY.mean().unstack().dropna()
    within_lift = float((within[1] / within[0].replace(0, np.nan)).median())
    pr = full[full.price_ratio > 0].copy()
    pr["price_ratio_decile"] = pd.qcut(pr.price_ratio.rank(method="first"), 10, labels=range(1, 11))
    price_eff = pr.groupby("price_ratio_decile").agg(mean_price_ratio=("price_ratio", "mean"), mean_qty=("QUANTITY", "mean"), rows=("QUANTITY", "size")).reset_index()
    price_eff.to_csv(OUT + "price_effect.csv", index=False)
    frac_pr = float((full.price_ratio > 0).mean())
    pr_only_promo = float(full.loc[full.price_ratio > 0, "PROMOTION_FLG_NUM"].mean())
    pr_zero_nopromo = float(1 - full.loc[full.price_ratio == 0, "PROMOTION_FLG_NUM"].mean())
    cp_rows = full[full.C_PROMOTION_FLG_NUM == 1]
    cp_brand = cp_rows.BRAND.value_counts()
    cp_top2_share = 100 * cp_brand.head(2).sum() / len(cp_rows)
    cp_raw = full.groupby("C_PROMOTION_FLG_NUM").QUANTITY.mean()
    wcp = full.groupby(["UPC", "C_PROMOTION_FLG_NUM"]).QUANTITY.mean().unstack().dropna()
    cp_within = float((wcp[1] / wcp[0].replace(0, np.nan)).median())
    shap = pd.read_csv(ROOT + "XAI_Outputs/shap_global.csv")
    shap_top = {m: list(d.sort_values("Mean_Abs_SHAP", ascending=False).Feature.head(3)) for m, d in shap.groupby("Model")}
    cov_min = None
    dow_mean = full.groupby("dayofweek_num").QUANTITY.mean()
    month_mean = full.groupby("month").QUANTITY.mean()
    daily = full.groupby("Date").QUANTITY.sum()

    # ------------------------------------------------------------------ 4. features
    print("Features...")
    fs = feature_summary(dev, features)
    fs.insert(1, "group", fs.feature.map(group_of))
    fs.to_csv(OUT + "feature_summary.csv", index=False)
    tc = []
    for f in features:
        x = dev[f].values
        ok = ~np.isnan(x)
        if ok.sum() > 10 and np.ptp(x[ok]) > 0:
            r, _ = spearmanr(x[ok], dev.QUANTITY.values[ok])
            tc.append({"feature": f, "group": group_of[f], "spearman_with_quantity": r, "abs": abs(r)})
    tc = pd.DataFrame(tc).sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
    tc.to_csv(OUT + "target_correlation.csv", index=False)
    lagcols = ["lag_1", "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_28"]
    lagc = pd.DataFrame([{"predictor": c, "spearman_with_quantity": spearmanr(dev[c], dev.QUANTITY, nan_policy="omit")[0],
                          "pearson_with_quantity": dev[[c, "QUANTITY"]].corr().iloc[0, 1], "missing_pct": 100 * dev[c].isna().mean()} for c in lagcols])
    lagc.to_csv(OUT + "lag_correlation.csv", index=False)
    C, pairs = correlated_pairs(dev, features, 0.95)
    pairs.to_csv(OUT + "correlated_pairs.csv", index=False)
    n_in_pair = len(pd.concat([pairs.feature_a, pairs.feature_b]).unique())
    gs = fs.groupby("group").agg(features=("feature", "count"), mean_zero_pct=("zero_pct", "mean"),
                                 mean_abs_skew=("skew", lambda s: s.abs().mean())).reset_index()

    # ------------------------------------------------------------------ 5. drift
    print("Drift...")
    cols = [n for n, _ in SPLITS[1:]]
    drift = pd.DataFrame({"feature": features + ["QUANTITY (target)"]})
    for n in cols:
        drift[n] = [psi(dev[f], data[n][f]) for f in features] + [psi(dev.QUANTITY, data[n].QUANTITY)]
    drift["max_psi"] = drift[cols].max(axis=1)
    drift["group"] = drift.feature.map(group_of)
    drift.sort_values("max_psi", ascending=False).to_csv(OUT + "drift_psi.csv", index=False)
    fd = drift[drift.feature != "QUANTITY (target)"]
    _s = fd[fd["oos"] > 0.25].group.value_counts()
    oos_groups = ", ".join(f"{g.split('. ', 1)[1].lower()} ({n})" for g, n in _s.items())
    drift_summary = pd.DataFrame({"split": cols, "features_psi_gt_0.1": [int((fd[c] > 0.1).sum()) for c in cols],
                                  "features_psi_gt_0.25": [int((fd[c] > 0.25).sum()) for c in cols],
                                  "median_psi": [fd[c].median() for c in cols],
                                  "target_psi": [float(drift.loc[drift.feature == "QUANTITY (target)", c].iloc[0]) for c in cols]})

    # ------------------------------------------------------------------ plots
    print("Plots...")
    plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": "#C9D1DB", "figure.dpi": 130})

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(q.clip(upper=60), bins=61, color=TEAL)
    ax[0].set_yscale("log")
    style_axes(ax[0], "Units sold per product-day (clipped at 60)", "units", "product-days (log scale)")
    ax[1].bar(overview.split, overview.zero_pct, color=AMBER)
    for i, v in enumerate(overview.zero_pct):
        ax[1].text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    ax[1].set_xticks(range(len(overview)), overview.split, rotation=25, ha="right")
    style_axes(ax[1], "Share of days with zero sales, by split", ylabel="% of rows")
    ax[2].bar(overview.split, overview.mean_qty, color=TEAL)
    for i, v in enumerate(overview.mean_qty):
        ax[2].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax[2].set_xticks(range(len(overview)), overview.split, rotation=25, ha="right")
    style_axes(ax[2], "Average units per product-day, by split", ylabel="units")
    fig.tight_layout(); fig.savefig(PLOTS + "01_target_distribution.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(daily.index, daily.values, color=TEAL, lw=0.8)
    ax.plot(daily.index, daily.rolling(28, center=True).mean(), color=INK, lw=2, label="28-day average")
    for n, _ in SPLITS:
        d0 = data[n].Date.min()
        if n.startswith("oot"):
            ax.axvline(d0, color=RED, ls="--", lw=1)
    ax.axvspan(data["oot_2012-12"].Date.min(), full.Date.max(), color=RED, alpha=0.07)
    ax.legend(frameon=False)
    style_axes(ax, "Total units sold per day, all products (shaded: the out-of-time months)", ylabel="units per day")
    fig.tight_layout(); fig.savefig(PLOTS + "02_daily_total.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    ax[0].bar([DOW[i] for i in dow_mean.index], dow_mean.values, color=TEAL)
    style_axes(ax[0], "Average units by day of week", ylabel="units per product-day")
    ax[1].bar(month_mean.index.astype(str), month_mean.values, color=TEAL)
    style_axes(ax[1], "Average units by calendar month", "month", "units per product-day")
    eff = cal[cal.effect.isin(["holiday", "own promotion", "competitor promotion"])]
    labs = [f"{e.replace('competitor promotion', 'comp.\npromo').replace('own promotion', 'own\npromo')}\n{l}" for e, l in zip(eff.effect, eff.level)]
    ax[2].bar(range(len(eff)), eff.mean_qty, color=[AMBER if l == "yes" else GRAY for l in eff.level])
    ax[2].set_xticks(range(len(eff)), labs, fontsize=7)
    style_axes(ax[2], "Average units: flag on (amber) vs off (grey)", ylabel="units per product-day")
    fig.tight_layout(); fig.savefig(PLOTS + "03_calendar_and_promotion.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(price_eff.price_ratio_decile.astype(int).astype(str), price_eff.mean_qty, color=TEAL)
    style_axes(ax[0], "Promotion days only: average units by price-cut decile", "price-ratio decile (1 = smallest cut)", "units per product-day")
    b = by_brand.sort_values("mean_qty", ascending=False)
    ax[1].bar(b.BRAND, b.mean_qty, color=TEAL)
    ax2 = ax[1].twinx()
    ax2.plot(b.BRAND, b.zero_pct, color=RED, marker="o")
    ax[1].set_xticks(range(len(b)), b.BRAND, rotation=40, ha="right", fontsize=8)
    style_axes(ax[1], "Brands: average units (bars) and % zero-sales days (line)", ylabel="units per product-day")
    ax2.set_ylabel("% zero days", color=RED); ax2.spines[["top"]].set_visible(False)
    fig.tight_layout(); fig.savefig(PLOTS + "04_price_and_brands.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    t = tc.head(15).iloc[::-1]
    ax[0].barh(t.feature, t.spearman_with_quantity, color=[TEAL if v > 0 else AMBER for v in t.spearman_with_quantity])
    style_axes(ax[0], "Features most correlated with units sold (Spearman, DEV)", "correlation")
    ax[0].grid(axis="x", color="#E3E8EE"); ax[0].grid(axis="y", visible=False)
    ax[1].bar(lagc.predictor, lagc.spearman_with_quantity, color=TEAL)
    ax[1].set_xticks(range(len(lagc)), lagc.predictor, rotation=30, ha="right")
    ax[1].set_ylim(0, 1)
    style_axes(ax[1], "How well past demand predicts today's demand", ylabel="Spearman correlation")
    fig.tight_layout(); fig.savefig(PLOTS + "05_correlation.png"); plt.close(fig)

    keep = list(tc.feature.head(30))
    keep = [k for k in keep if k in C.columns]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(C.loc[keep, keep].values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keep)), keep, rotation=90, fontsize=7); ax.set_yticks(range(len(keep)), keep, fontsize=7)
    ax.set_title("Spearman correlation among the 30 features most related to demand", loc="left", fontsize=12, fontweight="bold", color=INK)
    fig.colorbar(im, shrink=0.7); fig.tight_layout(); fig.savefig(PLOTS + "06_correlation_heatmap.png"); plt.close(fig)

    topd = drift.sort_values("max_psi", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(topd[cols].values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1.0)
    ax.set_xticks(range(len(cols)), cols, rotation=25, ha="right"); ax.set_yticks(range(len(topd)), topd.feature, fontsize=8)
    for i in range(len(topd)):
        for j, c in enumerate(cols):
            v = topd[c].iloc[i]
            ax.text(j, i, "" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if (not np.isnan(v) and v > 0.5) else INK)
    ax.set_title("Drift vs DEV (PSI; > 0.25 = large shift)", loc="left", fontsize=12, fontweight="bold", color=INK)
    fig.colorbar(im, shrink=0.8); fig.tight_layout(); fig.savefig(PLOTS + "07_drift_psi.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    s = by_upc.reset_index(drop=True)
    ax.plot(range(1, len(s) + 1), s.cum_volume_share_pct, color=TEAL, lw=2)
    ax.axhline(50, color=GRAY, ls="--", lw=1)
    style_axes(ax, "How concentrated is demand? (cumulative share of volume)", "products, largest first", "% of all units sold")
    fig.tight_layout(); fig.savefig(PLOTS + "08_product_concentration.png"); plt.close(fig)

    # ------------------------------------------------------------------ report
    ov = overview.rename(columns={"mean_qty": "mean units", "median_qty": "median", "p99_qty": "p99", "max_qty": "max", "zero_pct": "zero-sales %", "promo_pct": "promo %"})
    ov["rows"] = ov.rows.map("{:,}".format)
    miss = int((fs.missing_pct > 0).sum())
    brand_t = by_brand[["BRAND", "products", "rows", "mean_qty", "zero_pct", "volume_share_pct", "split_it_sits_in"]].rename(
        columns={"mean_qty": "mean units", "zero_pct": "zero-sales %", "volume_share_pct": "volume share %", "split_it_sits_in": "appears in"})
    cal_show = cal[cal.effect.isin(["own promotion", "competitor promotion", "holiday", "holiday eve", "holiday after"])]
    dow_t = pd.DataFrame({"day": [DOW[i] for i in dow_mean.index], "mean units": dow_mean.values})
    t15 = tc.head(10)
    grp = gs.rename(columns={"mean_zero_pct": "mean % zeros", "mean_abs_skew": "mean abs skew"})
    worst = drift.sort_values("max_psi", ascending=False).head(10)[["feature"] + cols]
    cov_med, cov_min = float(by_upc.coverage_pct.median()), float(by_upc.coverage_pct.min())
    lag1 = float(lagc.loc[lagc.predictor == "lag_1", "spearman_with_quantity"].iloc[0])
    rm7 = float(lagc.loc[lagc.predictor == "rolling_mean_7", "spearman_with_quantity"].iloc[0])
    zr = overview.set_index("split").zero_pct
    md = f"""# DemandForecast — exploratory data analysis

Generated by `data/DemandForecast/scripts/run_eda.py`. Descriptive only: nothing here fits a model.
One row = one product (UPC) on one day in a single store; the target is `QUANTITY` (units sold).
The {len(features)} model features are those used by the models and grouped by `feature_hierarchy.csv`.
Feature statistics use the DEV split; whole-history views use `features.csv`. Charts are in `EDA_Outputs/plots/`.

## 1. What the data looks like

- **{len(full):,} rows**, {full.UPC.nunique()} products, {full.BRAND.nunique()} brands, **one store**, {str(full.Date.min())[:10]} to {str(full.Date.max())[:10]}.
- **Missing values in the model features: {'none' if miss == 0 else f'{miss} features'}.**
- **No gaps in time:** every product has a row for every day between its first and last record (coverage min {cov_min:.0f}%, median {cov_med:.0f}%),
  so a zero-sales day is an explicit row of 0 — not a missing day. Products start and stop at different dates.
- **Demand is small, skewed and zero-heavy:** mean {q.mean():.1f} units, median {q.median():.0f}, 99th percentile {q.quantile(.99):.0f}, maximum {q.max():.0f}.
  **{100 * (q == 0).mean():.0f}% of product-days sell nothing.** The variance is {overdisp:.1f}× the mean (a Poisson count would have 1×), so demand
  is strongly over-dispersed — the reason a Tweedie loss was chosen for the tree models.

### Splits

{md_table(ov, "{:.2f}")}

![target](plots/01_target_distribution.png)

**The splits are not alike.** The zero-sales share is {zr['dev']:.0f}% in DEV, {zr['oos']:.0f}% in OOS and {zr['oot_2012-12']:.0f}% in December 2012.
OOS is a different set of brands from DEV (and holds only {int(overview.set_index('split').loc['oos', 'brands'])} brands), so scores on OOS reflect *which brands
were held out* as much as model quality — this very likely contributes to OOS R² being low and similar for every model (Section 4 shows how different OOS is).

### Brands and products

{md_table(brand_t, "{:.2f}")}

The top 10 products carry {top10_share:.0f}% of all units, and {n_for_half} products account for half of the volume
.

![concentration](plots/08_product_concentration.png)

![price and brands](plots/04_price_and_brands.png)

### Products and brands the model never saw in training

{md_table(ent, "{:.1f}")}

Out-of-time months contain products and brands that DEV never contained, so forecasting them is partly a cold-start problem.

## 2. Calendar, promotion and price

![total demand](plots/02_daily_total.png)

![calendar](plots/03_calendar_and_promotion.png)

- **Promotions lift demand:** average units are {promo_lift:.1f}× higher on a promotion day. Within the same product the median lift is
  {within_lift:.1f}× (so it is not just promoted products being bigger sellers).
- **Day of week:** the busiest day averages {dow_mean.max():.1f} units ({DOW[int(dow_mean.idxmax())]}) and the quietest {dow_mean.min():.1f} ({DOW[int(dow_mean.idxmin())]}).
- **Months:** monthly averages range from {month_mean.min():.1f} to {month_mean.max():.1f} units.
- **Price ratio is a promotion-depth measure, not a general price signal:** it is non-zero on {100 * frac_pr:.0f}% of rows, and on {100 * pr_only_promo:.0f}% of
  those it is a promotion day (it is zero on {100 * pr_zero_nopromo:.0f}% of non-promotion days). The price-decile chart therefore describes promotion days only.
- **Competitor promotions look like a huge effect, but it is a brand artefact.** Raw average demand is {cp_raw.loc[1]:.1f} units on days a competitor promotes
  vs {cp_raw.loc[0]:.1f} otherwise, but {cp_top2_share:.0f}% of those rows belong to just two small brands ({', '.join(cp_brand.head(2).index)}) that sell little anyway.
  Within the same product the median ratio is {cp_within:.2f}× (about no effect). Read the raw gap as confounded.

{md_table(cal_show, "{:.2f}")}

## 3. Which features relate to demand?

![correlation](plots/05_correlation.png)

Past demand is by far the strongest predictor: yesterday's units correlate {lag1:.2f} with today's and the 7-day rolling mean {rm7:.2f}.
Consistently, the baseline SHAP results (`XAI_Outputs/shap_global.csv`) put these at the top for the tree models — {'; '.join(f"{m}: {', '.join(v)}" for m, v in shap_top.items() if m != 'LinearRegression')} — while the linear model leans on
price and promotion ({', '.join(shap_top.get('LinearRegression', []))}). This is also why we verified (Phase 0) that the lag/rolling features use only *past* days.

{md_table(t15, "{:.3f}")}

Feature groups:

{md_table(grp, "{:.2f}")}

**Redundancy:** {len(pairs)} feature pairs have |Spearman| ≥ 0.95, involving {n_in_pair} of the {len(features)} features
(e.g. different rolling windows and price variants of one another), so explainers will tend to split credit between near-copies.

{md_table(pairs.head(8), "{:.3f}")}

![heatmap](plots/06_correlation_heatmap.png)

## 4. Drift against DEV

PSI of each feature and of the target itself (< 0.1 stable · 0.1–0.25 moderate · > 0.25 large):

{md_table(drift_summary, "{:.3f}")}

{md_table(worst, "{:.3f}")}

![drift](plots/07_drift_psi.png)

**Two different kinds of shift.**
- **OOS vs DEV is a large shift** ({int(drift_summary.loc[0, 'features_psi_gt_0.25'])} of {len(features)} features have PSI > 0.25, median PSI {drift_summary.loc[0, 'median_psi']:.2f}; target PSI {drift_summary.loc[0, 'target_psi']:.2f}),
  concentrated in {oos_groups} — because the held-out brands sell at different price levels and different demand levels. It is not drift over time — it is the brand split.
  TradeFraud's validation split, by contrast, shows no shift at all.
- **The out-of-time months shift in calendar and price features** (`month` and `quarter` are different seasons by construction — their PSI is large and mechanical, not a data problem —
  and {ent.rows_from_unseen_products_pct.min():.0f}–{ent.rows_from_unseen_products_pct.iloc[1:].max():.0f}% of out-of-time rows come from products DEV never saw). The *target* shifts only moderately (PSI {drift_summary.loc[1, 'target_psi']:.2f}–{drift_summary.loc[3, 'target_psi']:.2f}).

So unlike TradeFraud, the out-of-time months here are a genuine test of robustness to a changing population, and DemandForecast is the better dataset for
studying stability over time.

## 5. What this means for the project

1. **Few brands, one store:** OOS rests on {int(overview.set_index('split').loc['oos', 'brands'])} held-out brands and DEV on {int(overview.set_index('split').loc['dev', 'brands'])}; the splits differ in demand level
   and zero share. Treat model comparisons on OOS with care (already flagged in ARCHITECTURE.md Section 7).
2. **Zero-heavy, over-dispersed target:** justifies Tweedie loss, and means explanations of a "predicted ≈ 0" day are about small differences.
3. **Past demand and promotions dominate:** expect lag/rolling features and the promotion flags near the top of explanations; read raw competitor-promotion gaps with care (brand confound).
4. **Real drift:** out-of-time months differ from training, so explanation-agreement stability over time (Phase 4) is meaningful here.
"""
    open(OUT + "EDA_SUMMARY.md", "w").write(md)
    print(f"Wrote {OUT}EDA_SUMMARY.md and {len(os.listdir(PLOTS))} plots")


if __name__ == "__main__":
    main()
