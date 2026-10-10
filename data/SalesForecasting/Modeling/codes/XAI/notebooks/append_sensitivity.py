import json, sys

sys.path.insert(0, '..')

with open('03_shap_explanations.ipynb', 'r') as f:
    nb = json.load(f)

new_cells = [
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "---\n",
      "## Sensitivity Metrics\n",
      "\n",
      "Three frameworks are applied at **Global**, **Decile**, and **Segment** levels for XGBoost, LightGBM, ARDL Base and ARDL Adjusted:\n",
      "- **SHAP Sensitivity** — `std(SHAP values)` per feature (variability of contribution)\n",
      "- **Surrogate Regression** — Linear surrogate fitted on `X → ŷ`; coefficients = sensitivity, R² = fidelity\n",
      "- **LIME Sensitivity** — Sampled local linear explanations; mean abs weight = sensitivity"
     ]
    },
    # ── Setup ─────────────────────────────────────────────────────────────────
    {
     "cell_type": "code",
     "execution_count": None,
     "metadata": {},
     "outputs": [],
     "source": [
      "from src.sensitivity_utils import (\n",
      "    compute_shap_sensitivity, compute_surrogate_sensitivity, compute_lime_sensitivity\n",
      ")\n",
      "import os, pickle\n",
      "os.makedirs('../Outputs/sensitivity', exist_ok=True)\n",
      "\n",
      "# Helper: load per-Cust_ID XGB/LGBM predict fn that averages across Cust_IDs\n",
      "def make_tree_predict_fn(df_ref, model_dir, prefix, feature_cols):\n",
      "    \"\"\"Return a callable predict(X_arr) that dispatches to per-Cust_ID models.\"\"\"\n",
      "    import numpy as np, os as _os\n",
      "    cust_id_models = {}\n",
      "    for cust_id, _ in df_ref.groupby('Cust_ID'):\n",
      "        p = _os.path.join(model_dir, f'{prefix}_{cust_id}.pkl')\n",
      "        if _os.path.exists(p):\n",
      "            with open(p,'rb') as f: cust_id_models[cust_id] = pickle.load(f)\n",
      "    # Build a global surrogate predict using the first available model (for LIME background)\n",
      "    first_model = list(cust_id_models.values())[0]\n",
      "    def predict_fn(X_arr):\n",
      "        import pandas as pd\n",
      "        return first_model.predict(pd.DataFrame(X_arr, columns=feature_cols))\n",
      "    return predict_fn\n",
      "\n",
      "print('Sensitivity utils loaded.')\n"
     ]
    },
    # ─── SHAP SENSITIVITY ─────────────────────────────────────────────────────
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": ["### 7.1  SHAP Sensitivity"]
    },
    {
     "cell_type": "code",
     "execution_count": None,
     "metadata": {},
     "outputs": [],
     "source": [
      "shap_sens_dfs = []\n",
      "\n",
      "shap_configs = [\n",
      "    ('XGBoost',       'Validation', shap_xgb_val),\n",
      "    ('LightGBM',      'Validation', shap_lgbm_val),\n",
      "    ('ARDL Base',     'Validation', shap_ardl_base_val),\n",
      "    ('ARDL Adjusted', 'Validation', shap_ardl_adj_val),\n",
      "]\n",
      "\n",
      "# Global\n",
      "for m, d, sdf in shap_configs:\n",
      "    if sdf.empty: continue\n",
      "    shap_sens_dfs.append(compute_shap_sensitivity(sdf, feature_cols, m, d))\n",
      "\n",
      "shap_sens_global = pd.concat(shap_sens_dfs, ignore_index=True)\n",
      "shap_sens_global['Level'] = 'Global'\n",
      "shap_sens_global.to_csv('../Outputs/sensitivity/shap_sensitivity_global.csv', index=False)\n",
      "print('SHAP Sensitivity — Global done.')\n"
     ]
    },
    {
     "cell_type": "code",
     "execution_count": None,
     "metadata": {},
     "outputs": [],
     "source": [
      "# Decile\n",
      "shap_sens_decile_dfs = []\n",
      "for m, d, sdf in shap_configs:\n",
      "    if sdf.empty or 'Decile' not in sdf.columns: continue\n",
      "    shap_sens_decile_dfs.append(compute_shap_sensitivity(sdf, feature_cols, m, d, group_col='Decile'))\n",
      "\n",
      "shap_sens_decile = pd.concat(shap_sens_decile_dfs, ignore_index=True)\n",
      "shap_sens_decile['Level'] = 'Decile'\n",
      "shap_sens_decile.to_csv('../Outputs/sensitivity/shap_sensitivity_decile.csv', index=False)\n",
      "print('SHAP Sensitivity — Decile done.')\n"
     ]
    },
    {
     "cell_type": "code",
     "execution_count": None,
     "metadata": {},
     "outputs": [],
     "source": [
      "# Segment\n",
      "shap_val_with_brand = shap_xgb_val.copy()\n",
      "shap_val_with_brand['Segment'] = xgb_val['Segment'].values\n",
      "shap_lgbm_val_b = shap_lgbm_val.copy(); shap_lgbm_val_b['Segment'] = xgb_val['Segment'].values\n",
      "shap_ardl_base_val_b = shap_ardl_base_val.copy(); shap_ardl_base_val_b['Segment'] = ardl_val['Segment'].values\n",
      "shap_ardl_adj_val_b  = shap_ardl_adj_val.copy();  shap_ardl_adj_val_b['Segment'] = ardl_val['Segment'].values\n",
      "\n",
      "shap_segment_configs = [\n",
      "    ('XGBoost', 'Validation', shap_val_with_brand),\n",
      "    ('LightGBM', 'Validation', shap_lgbm_val_b),\n",
      "    ('ARDL Base', 'Validation', shap_ardl_base_val_b),\n",
      "    ('ARDL Adjusted', 'Validation', shap_ardl_adj_val_b),\n",
      "]\n",
      "shap_sens_segment_dfs = []\n",
      "for m, d, sdf in shap_segment_configs:\n",
      "    if sdf.empty: continue\n",
      "    shap_sens_segment_dfs.append(compute_shap_sensitivity(sdf, feature_cols, m, d, group_col='Segment'))\n",
      "\n",
      "shap_sens_brand = pd.concat(shap_sens_segment_dfs, ignore_index=True)\n",
      "shap_sens_brand['Level'] = 'Segment'\n",
      "shap_sens_brand.to_csv('../Outputs/sensitivity/shap_sensitivity_brand.csv', index=False)\n",
      "print('SHAP Sensitivity — Segment done.')\n"
     ]
    },
    # ─── SURROGATE REGRESSION ─────────────────────────────────────────────────
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": ["### 7.2  Surrogate Regression Sensitivity"]
    },
    {
     "cell_type": "code",
     "execution_count": None,
     "metadata": {},
     "outputs": [],
     "source": [
      "surr_configs = [\n",
      "    ('XGBoost',       xgb_val, 'xgb_predicted_quantity'),\n",
      "    ('LightGBM',      xgb_val, 'lgbm_predicted_quantity'),\n",
      "    ('ARDL Base',     ardl_val, 'ardl_base_predicted'),\n",
      "    ('ARDL Adjusted', ardl_val, 'ardl_adjusted_predicted'),\n",
      "]\n",
      "\n",
      "# Global\n",
      "surr_global_dfs = []\n",
      "for m, df_src, pred_col in surr_configs:\n",
      "    X = df_src[feature_cols]\n",
      "    y_hat = df_src[pred_col].fillna(0)\n",
      "    surr_global_dfs.append(compute_surrogate_sensitivity(X, y_hat, feature_cols, m, 'Validation'))\n",
      "\n",
      "surr_global = pd.concat(surr_global_dfs, ignore_index=True)\n",
      "surr_global['Level'] = 'Global'\n",
      "surr_global.to_csv('../Outputs/sensitivity/surrogate_sensitivity_global.csv', index=False)\n",
      "print('Surrogate Sensitivity — Global done.')\n"
     ]
    },
    {
     "cell_type": "code",
     "execution_count": None,
     "metadata": {},
     "outputs": [],
     "source": [
      "# Decile\n",
      "surr_decile_dfs = []\n",
      "for m, df_src, pred_col in surr_configs:\n",
      "    X = df_src[feature_cols]\n",
      "    y_hat = df_src[pred_col].fillna(0)\n",
      "    decile_data = df_src['Decile']\n",
      "    surr_decile_dfs.append(\n",
      "        compute_surrogate_sensitivity(X, y_hat, feature_cols, m, 'Validation',\n",
      "                                      group_col='Decile', group_data=decile_data)\n",
      "    )\n",
      "\n",
      "surr_decile = pd.concat(surr_decile_dfs, ignore_index=True)\n",
      "surr_decile['Level'] = 'Decile'\n",
      "surr_decile.to_csv('../Outputs/sensitivity/surrogate_sensitivity_decile.csv', index=False)\n",
      "print('Surrogate Sensitivity — Decile done.')\n"
     ]
    },
    {
     "cell_type": "code",
     "execution_count": None,
     "metadata": {},
     "outputs": [],
     "source": [
      "# Segment\n",
      "surr_segment_dfs = []\n",
      "for m, df_src, pred_col in surr_configs:\n",
      "    X = df_src[feature_cols]\n",
      "    y_hat = df_src[pred_col].fillna(0)\n",
      "    segment_data = df_src['Segment']\n",
      "    surr_segment_dfs.append(\n",
      "        compute_surrogate_sensitivity(X, y_hat, feature_cols, m, 'Validation',\n",
      "                                      group_col='Segment', group_data=segment_data)\n",
      "    )\n",
      "\n",
      "surr_brand = pd.concat(surr_segment_dfs, ignore_index=True)\n",
      "surr_brand['Level'] = 'Segment'\n",
      "surr_brand.to_csv('../Outputs/sensitivity/surrogate_sensitivity_brand.csv', index=False)\n",
      "print('Surrogate Sensitivity — Segment done.')\n"
     ]
    },
    # ─── LIME SENSITIVITY ─────────────────────────────────────────────────────
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": ["### 7.3  LIME Sensitivity"]
    },
    {
     "cell_type": "code",
     "execution_count": None,
     "metadata": {},
     "outputs": [],
     "source": [
      "lime_configs = [\n",
      "    ('XGBoost',       xgb_val, make_tree_predict_fn(xgb_val, model_dir, 'xgb', feature_cols)),\n",
      "    ('LightGBM',      xgb_val, make_tree_predict_fn(xgb_val, model_dir, 'lgbm', feature_cols)),\n",
      "]\n",
      "\n",
      "# ARDL predict wrapper (use ardl_adjusted as single function)\n",
      "def make_ardl_predict_fn(df_ref, segment_models, cust_id_adjust_models, proportions, feature_cols):\n",
      "    \"\"\"Returns a predict(X_arr) that uses the already-computed predictions via linear model.\"\"\"\n",
      "    first_brand = list(segment_models.keys())[0]\n",
      "    bm = segment_models[first_brand]\n",
      "    def predict_fn(X_arr):\n",
      "        import pandas as pd, numpy as np\n",
      "        X = pd.DataFrame(X_arr, columns=feature_cols)\n",
      "        return bm.predict(X[[c for c in bm.model.exog_names if c in X.columns]].assign(const=1))\n",
      "    return predict_fn\n",
      "\n",
      "# Global — LIME\n",
      "lime_global_dfs = []\n",
      "for m, df_src, predict_fn in lime_configs:\n",
      "    lime_global_dfs.append(\n",
      "        compute_lime_sensitivity(df_src, predict_fn, feature_cols, m, 'Validation', sample_frac=0.05)\n",
      "    )\n",
      "\n",
      "lime_global = pd.concat(lime_global_dfs, ignore_index=True)\n",
      "lime_global['Level'] = 'Global'\n",
      "lime_global.to_csv('../Outputs/sensitivity/lime_sensitivity_global.csv', index=False)\n",
      "print('LIME Sensitivity — Global done.')\n"
     ]
    },
    {
     "cell_type": "code",
     "execution_count": None,
     "metadata": {},
     "outputs": [],
     "source": [
      "# Decile — LIME\n",
      "lime_decile_dfs = []\n",
      "for m, df_src, predict_fn in lime_configs:\n",
      "    lime_decile_dfs.append(\n",
      "        compute_lime_sensitivity(df_src, predict_fn, feature_cols, m, 'Validation',\n",
      "                                  group_col='Decile', group_data=df_src['Decile'],\n",
      "                                  sample_frac=0.10)\n",
      "    )\n",
      "\n",
      "lime_decile = pd.concat(lime_decile_dfs, ignore_index=True)\n",
      "lime_decile['Level'] = 'Decile'\n",
      "lime_decile.to_csv('../Outputs/sensitivity/lime_sensitivity_decile.csv', index=False)\n",
      "print('LIME Sensitivity — Decile done.')\n"
     ]
    },
    {
     "cell_type": "code",
     "execution_count": None,
     "metadata": {},
     "outputs": [],
     "source": [
      "# Segment — LIME\n",
      "lime_segment_dfs = []\n",
      "for m, df_src, predict_fn in lime_configs:\n",
      "    lime_segment_dfs.append(\n",
      "        compute_lime_sensitivity(df_src, predict_fn, feature_cols, m, 'Validation',\n",
      "                                  group_col='Segment', group_data=df_src['Segment'],\n",
      "                                  sample_frac=0.10)\n",
      "    )\n",
      "\n",
      "lime_brand = pd.concat(lime_segment_dfs, ignore_index=True)\n",
      "lime_brand['Level'] = 'Segment'\n",
      "lime_brand.to_csv('../Outputs/sensitivity/lime_sensitivity_brand.csv', index=False)\n",
      "print('LIME Sensitivity — Segment done.')\n",
      "print('\\n✅ All Sensitivity Metrics generated and saved.')\n"
     ]
    },
]

nb['cells'].extend(new_cells)

with open('03_shap_explanations.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Cells appended.')
