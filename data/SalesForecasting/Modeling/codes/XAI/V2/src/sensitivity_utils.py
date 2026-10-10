import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from lime.lime_tabular import LimeTabularExplainer
from tqdm.auto import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 1. SHAP SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_sensitivity(shap_df, feature_cols, model_name, dataset_name,
                              group_col=None):
    """
    Compute SHAP-based sensitivity as std(SHAP value) per feature.
    High std means the feature's contribution is highly variable across samples.

    Parameters
    ----------
    shap_df : DataFrame  — local SHAP values (already loaded)
    feature_cols : list  — list of feature column names
    group_col : str or None  — if set, compute within each group (e.g. 'Decile' or 'Segment')
    """
    results = []

    def _summarise(df_slice, group_val=None):
        feat_df = df_slice[feature_cols]
        std_shap = feat_df.std()
        mean_abs_shap = feat_df.abs().mean()
        row = {
            'Model': model_name, 'Dataset': dataset_name,
            'Feature': std_shap.index,
            'SHAP_Std (Sensitivity)': std_shap.values,
            'SHAP_MeanAbs': mean_abs_shap.values,
        }
        if group_col:
            row[group_col] = group_val
        return pd.DataFrame(row)

    if group_col is None:
        results.append(_summarise(shap_df))
    else:
        for g, grp in shap_df.groupby(group_col):
            if len(grp) < 5:
                continue
            results.append(_summarise(grp, g))

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 2. SURROGATE REGRESSION SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_surrogate_sensitivity(X_df, y_pred, feature_cols, model_name,
                                   dataset_name, group_col=None, group_data=None):
    """
    Fit a LinearRegression surrogate on X → y_pred (model output).
    Coefficients = sensitivity of each feature on the model's output.
    Reports Fidelity R² of the surrogate.
    
    Parameters
    ----------
    X_df : DataFrame — features
    y_pred : Series/array — model predictions
    group_col : str or None — 'Decile' or 'Segment' for grouped
    group_data : Series/array or None — group labels aligned with X_df
    """
    results = []

    def _fit(X_slice, y_slice, group_val=None):
        if len(X_slice) < 10:
            return None
        X_clean = X_slice[feature_cols].fillna(0)
        y_clean = pd.Series(y_slice).fillna(0).values
        model = LinearRegression()
        model.fit(X_clean, y_clean)
        fidelity_r2 = r2_score(y_clean, model.predict(X_clean))
        row = {
            'Model': model_name, 'Dataset': dataset_name,
            'Feature': feature_cols,
            'Surrogate_Coefficient': model.coef_,
            'Surrogate_Abs_Coefficient': np.abs(model.coef_),
            'Fidelity_R2': fidelity_r2,
            'Intercept': model.intercept_,
        }
        if group_col:
            row[group_col] = group_val
        return pd.DataFrame(row)

    if group_col is None:
        r = _fit(X_df, y_pred)
        if r is not None:
            results.append(r)
    else:
        for g in pd.Series(group_data).unique():
            mask = group_data == g
            r = _fit(X_df[mask], y_pred[mask], g)
            if r is not None:
                results.append(r)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 3. LIME SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_lime_sensitivity(X_df, predict_fn, feature_cols, model_name,
                              dataset_name, group_col=None, group_data=None,
                              sample_frac=0.05, random_state=42):
    """
    Run LIME on a sample of observations, aggregate per-feature importance
    (absolute weight) as the sensitivity measure.
    NaN values are filled with 0. Features with zero variance per slice are
    silently ignored (LIME can't discretize them).
    """
    results = []
    # Fill NaN for LIME
    X_clean_df = X_df[feature_cols].fillna(0)
    X_arr = X_clean_df.values

    def _run_lime(X_slice_arr, group_val=None):
        n = len(X_slice_arr)
        if n < 5:
            return pd.DataFrame()
        n_sample = max(5, int(n * sample_frac))
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, size=min(n_sample, n), replace=False)
        sampled = X_slice_arr[idx]

        # Build explainer on this slice (avoids cross-slice variance issues)
        local_explainer = LimeTabularExplainer(
            training_data=X_slice_arr,
            feature_names=feature_cols,
            mode='regression',
            random_state=random_state,
            verbose=False,
        )

        feature_weights = {f: [] for f in feature_cols}
        for row in tqdm(sampled, desc=f"LIME [{model_name}|{group_val or dataset_name}]", leave=False):
            try:
                exp = local_explainer.explain_instance(
                    row, predict_fn, num_features=len(feature_cols)
                )
                for feat, weight in exp.as_list():
                    for col_name in feature_cols:
                        if col_name in feat:
                            feature_weights[col_name].append(abs(weight))
            except Exception:
                continue  # Skip zero-variance instances

        rows = []
        for f, weights in feature_weights.items():
            rows.append({
                'Feature': f,
                'LIME_Mean_Abs_Weight (Sensitivity)': np.mean(weights) if weights else np.nan,
                'LIME_Std_Weight': np.std(weights) if weights else np.nan,
                'LIME_N_Samples': n_sample,
                'Model': model_name,
                'Dataset': dataset_name,
            })
            if group_col:
                rows[-1][group_col] = group_val
        return pd.DataFrame(rows)

    if group_col is None:
        results.append(_run_lime(X_arr))
    else:
        for g in pd.Series(group_data).unique():
            mask = (group_data == g).values
            r = _run_lime(X_arr[mask], g)
            results.append(r)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
