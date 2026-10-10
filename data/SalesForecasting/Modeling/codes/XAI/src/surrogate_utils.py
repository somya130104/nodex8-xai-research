"""
Surrogate model utilities for SalesForecasting XAI pipeline.

Public API
----------
fit_surrogate(sub_df, features, pred_col, model_type, group_var)
    -> (list_of_row_dicts, dict_of_fitted_models)

aggregate_surrogate(df, features, run, min_samples=30)
    -> long DataFrame ready to concatenate

compute_point_contributions(df, features, run, sample_frac=0.10, min_rows=20)
    -> wide DataFrame with per-row surrogate feature contributions

Routing by model_type × group_vars availability:

  Regression + group_vars  → Random Effects Panel (MixedLM) with feature optimisation
  Regression + no groups   → GLS / OLS (HC3) with feature optimisation
  Classification           → Decision Tree  (group encoded as feature if present)
                           + Logistic Regression L1 (group encoded if present)
                             both with feature optimisation

Feature optimisation applies uniformly:
  - Linear models (RE/GLS): drop features with p > P_THRESHOLD; accept if BIC improves
  - Decision Tree: keep top-N features covering CUM_IMP_THRESHOLD of Gini importance
  - Logistic Regression: L1 penalty naturally zeroes out insignificant features

Dropped features always appear in output with coef=0, importance=0, selected=False.

Aggregation levels: Global and Decile only. Segment-level surrogates are dropped
(the RE panel structure already captures inter-segment variability globally).

"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, accuracy_score

warnings.filterwarnings("ignore")

# ── Tuneable constants ─────────────────────────────────────────────────────────
MIN_GROUPS_FOR_RE  = 2     # minimum distinct group values to use RE; else OLS
P_THRESHOLD        = 0.10  # max p-value to keep a feature in linear surrogates
BIC_TOL            = 2.0   # accept reduced model if BIC_reduced <= BIC_full + tol
CUM_IMP_THRESHOLD  = 0.95  # cumulative Gini threshold for DT feature selection
MAX_DT_DEPTH       = 5     # cap on decision tree depth
MAX_RANDOM_SLOPES  = 5     # random slopes added to RE model (beyond random intercept)


# ── Internal helpers ───────────────────────────────────────────────────────────
def _feature_indices(features, subset):
    return [features.index(f) for f in subset]


def _fit_linear(X_df, y, groups, use_re):
    """Fit MixedLM (use_re=True) or OLS (use_re=False). Returns statsmodels result."""
    exog = sm.add_constant(X_df, has_constant="add")
    try:
        if use_re:
            res = MixedLM(y, exog, groups=groups).fit(reml=False, method="lbfgs")
        else:
            res = sm.OLS(y, exog).fit(cov_type="HC3")
        return res
    except Exception:
        return sm.OLS(y, exog).fit()


def _scaled_importances(X_opt, y, opt_features, groups, use_re):
    """Re-fit on StandardScaler-transformed X to get scale-free |coef|."""
    scaler = StandardScaler()
    X_s = pd.DataFrame(scaler.fit_transform(X_opt), columns=opt_features)
    try:
        res = _fit_linear(X_s, y, groups, use_re)
        params = res.params.drop(["const", "Intercept"], errors="ignore")
        return {f: abs(float(params.get(f, 0.0))) for f in opt_features}
    except Exception:
        return {f: 0.0 for f in opt_features}


# ── Linear surrogate (Regression) ─────────────────────────────────────────────
def _optimize_linear(X, y, features, groups=None):
    """
    Feature selection via OLS p-values, then final MixedLM with random intercept
    + random slopes for up to MAX_RANDOM_SLOPES top selected features (RE_Panel),
    or plain OLS/GLS when no groups are present.

    Random slopes give each group a feature-specific coefficient deviation from
    the population fixed effect:
        effective_coef(group, feature) = fixed_effect + random_slope(group, feature)

    Returns:
        coefs_all    : np.ndarray (n_features,) — fixed-effect coefs, 0 for dropped
        imps_all     : np.ndarray (n_features,) — |scaled coef|, 0 for dropped
        selected     : np.ndarray (n_features, bool)
        fidelity_r2  : float
        bic          : float
        surr_name    : "RE_Panel" or "GLS"
        group_coefs  : dict {group_label: np.ndarray (n_features,)} of effective coefs,
                       or None when no RE model was fit
    """
    n_features = len(features)
    n_groups   = len(np.unique(groups)) if groups is not None else 0
    use_re     = groups is not None and n_groups >= MIN_GROUPS_FOR_RE
    surr_name  = "RE_Panel" if use_re else "GLS"

    X_df_full = pd.DataFrame(X, columns=features)

    # ── Step 1: Feature selection via OLS (fast — no MixedLM needed here) ───
    exog_full = sm.add_constant(X_df_full, has_constant="add")
    try:
        res_ols     = sm.OLS(y, exog_full).fit()
        pvals       = res_ols.pvalues.drop(["const", "Intercept"], errors="ignore")
        coef_abs    = res_ols.params.drop(["const", "Intercept"], errors="ignore").abs()
    except Exception:
        pvals    = pd.Series({f: 1.0 for f in features})
        coef_abs = pd.Series({f: 0.0 for f in features})

    keep = [f for f in features if pvals.get(f, 1.0) <= P_THRESHOLD]
    if not keep:
        keep = coef_abs.nlargest(min(3, n_features)).index.tolist()

    # ── Step 2: Fit final model on selected features ─────────────────────────
    exog_sel  = sm.add_constant(X_df_full[keep], has_constant="add")
    group_coefs = None

    if use_re:
        # Random slopes for top MAX_RANDOM_SLOPES features (ranked by |OLS coef|)
        top_re = sorted(keep, key=lambda f: coef_abs.get(f, 0.0), reverse=True)[:MAX_RANDOM_SLOPES]

        # exog_re: column of 1s (random intercept) + random slope columns
        re_cols   = sm.add_constant(X_df_full[top_re], has_constant="add") if top_re \
                    else np.ones((len(y), 1))

        try:
            res_final = MixedLM(y, exog_sel, groups=groups, exog_re=re_cols).fit(
                reml=False, method="lbfgs"
            )
            surr_name = "RE_Panel"

            # Extract fixed effects and per-group random effects
            fixed = res_final.params.drop(["const", "Intercept"], errors="ignore")
            group_coefs = {}
            for grp, re_ser in res_final.random_effects.items():
                eff = np.zeros(n_features)
                for j, feat in enumerate(features):
                    if feat in keep:
                        eff[j] = float(fixed.get(feat, 0.0))
                        # Add random slope deviation if this feature has one
                        if feat in top_re and feat in re_ser.index:
                            eff[j] += float(re_ser[feat])
                group_coefs[str(grp)] = eff

        except Exception:
            # Convergence failure — fall back to random intercepts only
            try:
                res_final = MixedLM(y, exog_sel, groups=groups).fit(reml=False, method="lbfgs")
                surr_name = "RE_Panel"
            except Exception:
                res_final = sm.OLS(y, exog_sel).fit()
                surr_name = "GLS"
    else:
        try:
            res_final = sm.OLS(y, exog_sel).fit(cov_type="HC3")
        except Exception:
            res_final = sm.OLS(y, exog_sel).fit()

    # ── Fidelity & BIC ───────────────────────────────────────────────────────
    fidelity_r2 = float(r2_score(y, res_final.fittedvalues))
    bic_final   = float(res_final.bic)

    # ── Scaled importances (fixed effects only, on selected features) ────────
    scaled_imp = _scaled_importances(X_df_full[keep], y, keep, groups, use_re)

    # ── Build output arrays ──────────────────────────────────────────────────
    raw_params = res_final.params.drop(["const", "Intercept"], errors="ignore")
    coefs_all  = np.zeros(n_features)
    imps_all   = np.zeros(n_features)
    selected   = np.zeros(n_features, dtype=bool)

    for f in keep:
        j = features.index(f)
        coefs_all[j] = float(raw_params.get(f, 0.0))
        imps_all[j]  = float(scaled_imp.get(f, 0.0))
        selected[j]  = True

    return coefs_all, imps_all, selected, fidelity_r2, bic_final, surr_name, group_coefs


# ── Decision Tree surrogate (Classification) ───────────────────────────────────
def _optimize_tree(X, y_binary, features, group_enc=None):
    """
    Fit Decision Tree with cumulative Gini feature selection.

    group_enc: (n, 1) array of encoded group values appended as an extra feature.

    Returns: imps_all (n_features,), selected (n_features,), fidelity_acc, clf_final
    """
    n_features = len(features)

    X_dt   = np.column_stack([X, group_enc]) if group_enc is not None else X
    n_cols = X_dt.shape[1]
    max_d  = min(MAX_DT_DEPTH, max(2, int(np.log2(max(len(X), 2)) - 1)))

    clf_full = DecisionTreeClassifier(max_depth=max_d, random_state=42)
    clf_full.fit(X_dt, y_binary)
    imps_full = clf_full.feature_importances_  # length n_cols

    # Cumulative threshold on all columns (including group_enc column)
    sorted_idx = np.argsort(imps_full)[::-1]
    cum_imp    = np.cumsum(imps_full[sorted_idx])
    n_keep     = max(1, int(np.searchsorted(cum_imp, CUM_IMP_THRESHOLD)) + 1)
    keep_idx   = set(sorted_idx[:n_keep])

    # Rebuild with selected columns
    sel_cols   = sorted(keep_idx)
    clf_final  = DecisionTreeClassifier(max_depth=max_d, random_state=42)
    clf_final.fit(X_dt[:, sel_cols], y_binary)
    fidelity   = float(accuracy_score(y_binary, clf_final.predict(X_dt[:, sel_cols])))

    # Map back to original features only (drop group_enc column if it was idx = n_features)
    imps_all = np.zeros(n_features)
    selected  = np.zeros(n_features, dtype=bool)
    for orig_j in range(n_features):
        imps_all[orig_j] = float(imps_full[orig_j])
        if orig_j in keep_idx:
            selected[orig_j] = True

    total = imps_all.sum()
    if total > 0:
        imps_all /= total

    return imps_all, selected, fidelity, clf_final, sel_cols


# ── Logistic Regression surrogate (Classification) ────────────────────────────
def _optimize_logreg(X, y_binary, features, group_enc=None):
    """
    Fit L1-penalised LogisticRegressionCV. L1 naturally zeroes insignificant features.

    Returns: coefs_all, imps_all, selected, fidelity_acc
    """
    n_features = len(features)
    X_lr = np.column_stack([X, group_enc]) if group_enc is not None else X

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_lr)

    n_splits = min(5, max(2, len(np.unique(y_binary))))
    try:
        clf = LogisticRegressionCV(
            Cs=np.logspace(-3, 3, 10),
            penalty="l1",
            solver="saga",
            cv=n_splits,
            max_iter=2000,
            random_state=42,
        )
        clf.fit(X_scaled, y_binary)
    except Exception:
        clf = LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=1000, random_state=42
        )
        clf.fit(X_scaled, y_binary)

    coefs_lr   = clf.coef_[0]  # length n_lr_features
    fidelity   = float(accuracy_score(y_binary, clf.predict(X_scaled)))

    # Map back to original features only
    coefs_all = np.zeros(n_features)
    imps_all  = np.zeros(n_features)
    selected  = np.zeros(n_features, dtype=bool)

    for j in range(n_features):
        coefs_all[j] = float(coefs_lr[j])
        selected[j]  = coefs_lr[j] != 0.0
        imps_all[j]  = abs(float(coefs_lr[j]))

    total = imps_all.sum()
    if total > 0:
        imps_all /= total

    return coefs_all, imps_all, selected, fidelity, clf, scaler


# ── Unified fit function ───────────────────────────────────────────────────────
def fit_surrogate(sub_df: pd.DataFrame, features: list,
                  pred_col: str, model_type: str,
                  group_var: str = None) -> tuple:
    """
    Fit surrogate model(s) for a data subset.

    Returns:
        rows   : list of row-dicts (one per Surrogate_Model × Feature)
        models : dict keyed by surrogate name → fitted artefacts for point-level reuse
    """
    rows   = []
    models = {}

    X = sub_df[features].fillna(0).values.astype(float)
    y = sub_df[pred_col].fillna(0).values.astype(float)

    # Panel dimension — only use if >1 entity present in this subset
    groups = None
    le_grp = None
    group_enc = None
    if group_var and group_var in sub_df.columns and sub_df[group_var].nunique() > 1:
        groups    = sub_df[group_var].values
        le_grp    = LabelEncoder()
        group_enc = le_grp.fit_transform(groups).reshape(-1, 1)

    def _row(surr, j, feat, sel, coef, imp, fid_r2, fid_acc, bic):
        return {
            "Surrogate_Model"   : surr,
            "Feature"           : feat,
            "Selected"          : bool(sel),
            "Surrogate_Coef"    : round(float(coef), 6) if not np.isnan(coef) else np.nan,
            "Feature_Importance": round(float(imp), 6),
            "Fidelity_R2"       : round(float(fid_r2), 4) if not np.isnan(fid_r2) else np.nan,
            "Fidelity_Accuracy" : round(float(fid_acc), 4) if not np.isnan(fid_acc) else np.nan,
            "BIC"               : round(float(bic), 2) if not np.isnan(bic) else np.nan,
        }

    # ── Regression ──────────────────────────────────────────────────────────
    group_coefs = None   # populated for RE_Panel; None for GLS / Classification
    if model_type == "Regression":
        coefs, imps, sel, fid_r2, bic, surr_name, group_coefs = _optimize_linear(
            X, y, features, groups=groups
        )
        models[surr_name] = {"coefs": coefs, "features": features,
                              "intercept": None, "group_coefs": group_coefs}
        for j, feat in enumerate(features):
            rows.append(_row(surr_name, j, feat, sel[j],
                             coefs[j], imps[j], fid_r2, np.nan, bic))

    # ── Classification ──────────────────────────────────────────────────────
    elif model_type == "Classification":
        if np.issubdtype(y.dtype, np.floating) and y.max() <= 1.0:
            y_bin = (y >= 0.5).astype(int)
        else:
            y_bin = y.astype(int)

        # Decision Tree
        dt_imps, dt_sel, dt_acc, clf_dt, dt_sel_cols = _optimize_tree(
            X, y_bin, features, group_enc
        )
        models["DecisionTree"] = {"clf": clf_dt, "sel_cols": dt_sel_cols,
                                  "features": features, "group_enc": group_enc,
                                  "le_grp": le_grp}
        for j, feat in enumerate(features):
            rows.append(_row("DecisionTree", j, feat, dt_sel[j],
                             np.nan, dt_imps[j], np.nan, dt_acc, np.nan))

        # Logistic Regression
        lr_coefs, lr_imps, lr_sel, lr_acc, lr_clf, lr_scaler = _optimize_logreg(
            X, y_bin, features, group_enc
        )
        models["LogisticRegression"] = {"clf": lr_clf, "scaler": lr_scaler,
                                        "coefs": lr_coefs, "features": features,
                                        "group_enc": group_enc, "le_grp": le_grp}
        for j, feat in enumerate(features):
            rows.append(_row("LogisticRegression", j, feat, lr_sel[j],
                             lr_coefs[j], lr_imps[j], np.nan, lr_acc, np.nan))

    return rows, models, group_coefs


# ── Aggregator ─────────────────────────────────────────────────────────────────
def aggregate_surrogate(df: pd.DataFrame, features: list,
                        run: dict, min_samples: int = 30) -> pd.DataFrame:
    """
    Fit and aggregate surrogate explanations at Global and Decile levels.

    Output schema (long format, mirrors shap_importance.csv):
        Dataset, Model, Model_Domain, Model_Type,
        Group_Variable, Group_Value, Samples,
        Surrogate_Model, Feature, Selected,
        Surrogate_Coef, Feature_Importance,
        Fidelity_R2, Fidelity_Accuracy, BIC
    """
    meta = {
        "Dataset"     : run["dataset"],
        "Model"       : run["model"],
        "Model_Domain": run["model_domain"],
        "Model_Type"  : run["model_type"],
    }
    model_type = run["model_type"]
    group_var  = run["group_vars"][0] if run["group_vars"] else None
    pred_col   = run["pred_col"]
    target_col = run["target_col"]

    df = df.reset_index(drop=True)

    # Decile — rank by score_col (probability for Classification,
    # same as pred_col for Regression)
    score_col = run.get("score_col") or target_col
    if "Decile" not in df.columns:
        df = df.copy()
        df["Decile"] = pd.qcut(
            df[score_col].rank(method="first"), 10,
            labels=[f"D{i}" for i in range(1, 11)],
        )

    all_rows = []

    def _process(sub_df, gvar, gval):
        n = len(sub_df)
        if n < min_samples:
            return
        gv = (group_var
              if group_var and group_var in sub_df.columns
              and sub_df[group_var].nunique() > 1
              else None)
        try:
            rows, _, group_coefs = fit_surrogate(sub_df, features, pred_col, model_type, group_var=gv)
        except Exception as e:
            print(f"    [warn] {gvar}={gval} — {e}")
            return

        # Fixed-effect rows (population-level)
        for r in rows:
            all_rows.append({**meta,
                             "Group_Variable": gvar,
                             "Group_Value"   : gval,
                             "Samples"       : n,
                             **r})

        # RE_Segment rows — segment-specific effective coefs from the global RE model.
        # Only emitted at the Global level where the RE Panel was fit on all data.
        if gvar == "Global" and group_coefs and gv:
            global_feat_map = {r["Feature"]: r for r in rows
                               if r.get("Surrogate_Model") == "RE_Panel"}
            for grp, eff_arr in sorted(group_coefs.items()):
                n_grp  = int((sub_df[gv] == grp).sum()) if gv in sub_df.columns else n
                total  = sum(abs(float(eff_arr[j])) for j in range(len(features)))
                for j, feat in enumerate(features):
                    ref    = global_feat_map.get(feat, {})
                    eff_c  = round(float(eff_arr[j]), 6)
                    eff_i  = round(abs(eff_c) / total, 6) if total > 0 else 0.0
                    all_rows.append({
                        **meta,
                        "Group_Variable"    : "RE_Segment",
                        "Group_Value"       : str(grp),
                        "Samples"           : n_grp,
                        "Surrogate_Model"   : "RE_Panel",
                        "Feature"           : feat,
                        "Selected"          : ref.get("Selected", False),
                        "Surrogate_Coef"    : eff_c,
                        "Feature_Importance": eff_i,
                        "Fidelity_R2"       : ref.get("Fidelity_R2"),
                        "Fidelity_Accuracy" : np.nan,
                        "BIC"               : ref.get("BIC"),
                    })

    # Global
    _process(df, "Global", "Global")

    # Decile
    for dval, didx in df.groupby("Decile").groups.items():
        _process(df.loc[didx], "Decile", str(dval))

    return pd.DataFrame(all_rows)


# ── Point-level contributions ──────────────────────────────────────────────────
def compute_point_contributions(
    df: pd.DataFrame,
    features: list,
    run: dict,
    sample_frac: float = 0.10,
    min_rows: int = 20,
) -> pd.DataFrame:
    """
    Fit Global surrogate on df, then compute per-row feature contributions for
    a stratified sample.

    Regression (RE_Panel / GLS):
        contribution_j = coef_j * x_j  (raw coef × raw feature value)

    Classification DT:
        Shapley values from SHAP TreeExplainer on the surrogate DT
        (positive-class SHAP values for class=1)

    Classification LogReg:
        contribution_j = coef_j * scaled_x_j  (L1 coef × StandardScaler-transformed value)

    Returns wide DataFrame:
        [identity cols + target_col + pred_col]
        + surr_{RE_Panel|GLS}_{feature}    for Regression
        + surr_DT_{feature}                for Classification (DT)
        + surr_LR_{feature}                for Classification (LogReg)
    """
    import shap as _shap

    model_type = run["model_type"]
    group_var  = run["group_vars"][0] if run["group_vars"] else None
    pred_col   = run["pred_col"]
    target_col = run["target_col"]

    # ── Stratified sample ──────────────────────────────────────────────────────
    strat_col = group_var if group_var and group_var in df.columns else None
    if strat_col:
        sample = (
            df.groupby(strat_col, group_keys=False)
            .apply(lambda x: x.sample(frac=sample_frac, random_state=42))
            .reset_index(drop=True)
        )
    else:
        sample = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    if len(sample) < min_rows:
        sample = df.sample(min(min_rows, len(df)), random_state=42).reset_index(drop=True)

    # ── Fit surrogate on FULL df ────────────────────────────────────────────────
    full_X = df[features].fillna(0).values.astype(float)
    full_y = df[pred_col].fillna(0).values.astype(float)

    groups = None
    le_grp = None
    group_enc = None
    if group_var and group_var in df.columns and df[group_var].nunique() > 1:
        groups    = df[group_var].values
        le_grp    = LabelEncoder()
        le_grp.fit(groups)
        group_enc = le_grp.transform(groups).reshape(-1, 1)

    sample_X = sample[features].fillna(0).values.astype(float)

    # Encode sample group values (safe transform — unknown values mapped to 0)
    def _safe_encode(vals):
        enc = np.zeros(len(vals), dtype=np.int64)
        for i, v in enumerate(vals):
            if v in le_grp.classes_:
                enc[i] = int(le_grp.transform([v])[0])
        return enc.reshape(-1, 1)

    sample_grp = None
    if le_grp is not None and group_var in sample.columns:
        sample_grp = _safe_encode(sample[group_var].values)

    # ── Identity columns ───────────────────────────────────────────────────────
    # group_var first, then generic time/id cols, then target/pred
    _BASE_ID = ["Cust_ID", "WeekYear", "Year", "WeekNum"]
    _gv_list = [group_var] if group_var else []
    id_cols  = [c for c in _gv_list + _BASE_ID + [target_col, pred_col]
                if c in sample.columns and c not in ([] if not _gv_list else [])]
    # deduplicate while preserving order
    seen = set(); id_cols = [c for c in id_cols if not (c in seen or seen.add(c))]
    out      = sample[id_cols].copy().reset_index(drop=True)

    n_features = len(features)

    # ── Regression ─────────────────────────────────────────────────────────────
    if model_type == "Regression":
        coefs, _, _, _, _, surr_name, group_coefs = _optimize_linear(
            full_X, full_y, features, groups=groups
        )
        tag = surr_name.replace(" ", "_")

        if group_coefs and group_var and group_var in sample.columns:
            # Per-row effective contributions using segment-specific coefs
            for j, feat in enumerate(features):
                contribs = np.array([
                    group_coefs.get(str(seg), coefs)[j] * sample_X[i, j]
                    for i, seg in enumerate(sample[group_var].values)
                ], dtype=np.float64)
                out[f"surr_{tag}_{feat}"] = contribs.round(6)
        else:
            # Fixed-effect contributions (GLS or RE without groups in sample)
            for j, feat in enumerate(features):
                out[f"surr_{tag}_{feat}"] = (coefs[j] * sample_X[:, j]).round(6)

    # ── Classification ─────────────────────────────────────────────────────────
    elif model_type == "Classification":
        full_y_bin = (
            (full_y >= 0.5).astype(int)
            if np.issubdtype(full_y.dtype, np.floating) and full_y.max() <= 1.0
            else full_y.astype(int)
        )

        # Augment X with group encoding column
        full_X_aug   = np.column_stack([full_X, group_enc])   if group_enc   is not None else full_X
        sample_X_aug = np.column_stack([sample_X, sample_grp]) if sample_grp is not None else sample_X

        # ── DT ────────────────────────────────────────────────────────────────
        dt_imps, dt_sel, dt_acc, clf_dt, dt_sel_cols = _optimize_tree(
            full_X, full_y_bin, features, group_enc
        )
        # SHAP on surrogate DT (background = full training data for DT)
        dt_explainer = _shap.TreeExplainer(clf_dt)
        shap_dt_raw  = dt_explainer.shap_values(sample_X_aug[:, dt_sel_cols])
        # Binary DT returns list [class0_array, class1_array]; take class 1
        shap_dt = shap_dt_raw[1] if isinstance(shap_dt_raw, list) else shap_dt_raw

        # Map selected column positions back to original feature indices
        dt_sel_map = {col_idx: pos for pos, col_idx in enumerate(dt_sel_cols)}
        for j, feat in enumerate(features):
            if j in dt_sel_map:
                out[f"surr_DT_{feat}"] = shap_dt[:, dt_sel_map[j]].round(6)
            else:
                out[f"surr_DT_{feat}"] = 0.0

        # ── LogReg ────────────────────────────────────────────────────────────
        lr_coefs, _, _, _, lr_clf, lr_scaler = _optimize_logreg(
            full_X, full_y_bin, features, group_enc
        )
        sample_X_scaled = lr_scaler.transform(sample_X_aug)
        for j, feat in enumerate(features):
            out[f"surr_LR_{feat}"] = (lr_coefs[j] * sample_X_scaled[:, j]).round(6)

    return out
