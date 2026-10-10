import json

with open('03_shap_explanations.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if 'cell_type' in cell and cell['cell_type'] == 'code':
        source = cell['source']
        if "def regenerate_and_save_all_deciles():" in source[0] or "def regenerate_and_save_all_deciles():" in source[3]:
            # Overwrite the entire cell source with the comprehensive version
            cell['source'] = [
                "import os\n",
                "os.makedirs('../Outputs/shap_local_decile_bins', exist_ok=True)\n",
                "\n",
                "def regenerate_and_save_all_deciles():\n",
                "    for d in range(1, 11):\n",
                "        # XGBoost\n",
                "        d_val_xgb = xgb_val[xgb_val['Decile'] == d]\n",
                "        s_xgb = calculate_tree_shap(d_val_xgb, model_dir, 'xgb', feature_cols)\n",
                "        if not s_xgb.empty:\n",
                "            s_xgb.to_csv(f'../Outputs/shap_local_decile_bins/xgb_val_decile_{d}.csv', index=False)\n",
                "        \n",
                "        # LightGBM\n",
                "        s_lgbm = calculate_tree_shap(d_val_xgb, model_dir, 'lgbm', feature_cols)\n",
                "        if not s_lgbm.empty:\n",
                "            s_lgbm.to_csv(f'../Outputs/shap_local_decile_bins/lgbm_val_decile_{d}.csv', index=False)\n",
                "        \n",
                "        # ARDL Base and ARDL Adjusted\n",
                "        a_val = ardl_val[ardl_val['Decile'] == d]\n",
                "        if len(a_val) == 0: continue\n",
                "        x_exp = ardl_dev[ardl_dev['Decile'] == d][feature_cols].mean()\n",
                "        s_base, s_adj = calculate_ardl_linear_shap(\n",
                "            a_val, segment_models, cust_id_adjust_models, proportions, feature_cols, x_exp\n",
                "        )\n",
                "        if not s_base.empty:\n",
                "            s_base.to_csv(f'../Outputs/shap_local_decile_bins/ardl_base_val_decile_{d}.csv', index=False)\n",
                "        if not s_adj.empty:\n",
                "            s_adj.to_csv(f'../Outputs/shap_local_decile_bins/ardl_adj_val_decile_{d}.csv', index=False)\n",
                "        \n",
                "print('Saving explicit Decile-Based Local SHAP files for ALL models...')\n",
                "regenerate_and_save_all_deciles()\n",
                "print('✅ Explicit Decile-based Local SHAP calculated and saved for all 4 models.')\n"
            ]

with open('03_shap_explanations.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
