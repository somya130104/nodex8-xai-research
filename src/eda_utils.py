"""
Helpers shared by the two exploratory-data-analysis scripts
(data/TradeFraud/scripts/run_eda.py, data/DemandForecast/scripts/run_eda.py).

Everything here is descriptive: it measures the data, it does not model it.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, skew


def feature_summary(df: pd.DataFrame, cols) -> pd.DataFrame:
    """One row per feature: type, missing, zeros, location/spread, tails, skew."""
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        v = s.dropna().values
        n = len(s)
        if len(v) == 0:
            rows.append({"feature": c, "missing_pct": 100.0, "n_unique": 0, "constant": True})
            continue
        q = np.percentile(v, [1, 25, 50, 75, 99])
        rows.append({
            "feature": c, "missing_pct": 100.0 * s.isna().mean(), "zero_pct": 100.0 * float((v == 0).mean()),
            "n_unique": int(len(np.unique(v))), "integer_valued": bool(np.all(np.mod(v, 1) == 0)),
            "constant": bool(np.ptp(v) == 0), "mean": v.mean(), "std": v.std(), "min": v.min(),
            "p01": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p99": q[4], "max": v.max(),
            "skew": float(skew(v)) if np.ptp(v) > 0 else 0.0,
            "neg_pct": 100.0 * float((v < 0).mean()),
            # how many times the 99th percentile the maximum is: a quick outlier gauge
            "max_over_p99": float(v.max() / q[4]) if q[4] > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def univariate_auc(df: pd.DataFrame, cols, y) -> pd.DataFrame:
    """
    AUC of each feature used alone as a score for a binary target (rank-based, so any
    monotone transform gives the same answer). Reported as the direction-free strength
    |AUC - 0.5| * 2 as well, because the sign of a feature's effect is not the question.
    """
    y = np.asarray(y).astype(int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    out = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").values
        ok = ~np.isnan(x)
        if ok.sum() < 10 or np.ptp(x[ok]) == 0 or y[ok].sum() == 0 or (1 - y[ok]).sum() == 0:
            continue
        r = rankdata(x[ok])
        a1 = int(y[ok].sum())
        auc = (r[y[ok] == 1].sum() - a1 * (a1 + 1) / 2) / (a1 * int((1 - y[ok]).sum()))
        out.append({"feature": c, "auc": auc, "strength": abs(auc - 0.5) * 2, "direction": "higher in disputes" if auc > 0.5 else "lower in disputes"})
    return pd.DataFrame(out).sort_values("strength", ascending=False).reset_index(drop=True)


def psi(expected, actual, bins: int = 10) -> float:
    """
    Population Stability Index of `actual` against `expected` using the expected sample's
    deciles. < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 large shift (industry convention).
    Features with too few distinct values to form bins return NaN.
    """
    e = pd.to_numeric(pd.Series(expected), errors="coerce").dropna().values
    a = pd.to_numeric(pd.Series(actual), errors="coerce").dropna().values
    if len(e) < 50 or len(a) < 50:
        return np.nan
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def correlated_pairs(df: pd.DataFrame, cols, threshold: float = 0.95, max_rows: int = 20000, seed: int = 0):
    """Spearman correlation among features. Returns (matrix, long table of pairs with |rho| >= threshold)."""
    d = df[cols].apply(pd.to_numeric, errors="coerce")
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=seed)
    d = d.loc[:, d.nunique() > 1]
    C = d.rank().corr()
    iu = np.triu_indices(len(C), 1)
    pairs = pd.DataFrame({"feature_a": C.columns[iu[0]], "feature_b": C.columns[iu[1]], "spearman": C.values[iu]})
    pairs = pairs[pairs.spearman.abs() >= threshold].copy()
    pairs["abs"] = pairs.spearman.abs()
    return C, pairs.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}", max_rows: int = None) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    if max_rows is not None:
        df = df.head(max_rows)
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    """One consistent, quiet plot style across every EDA chart."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E3E8EE", linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#1B2430")
    if xlabel:
        ax.set_xlabel(xlabel, color="#4B5563")
    if ylabel:
        ax.set_ylabel(ylabel, color="#4B5563")
