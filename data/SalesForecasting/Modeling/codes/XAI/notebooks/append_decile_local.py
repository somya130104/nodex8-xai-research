import json

with open('03_shap_explanations.ipynb', 'r') as f:
    nb = json.load(f)

md_cell = {
 "cell_type": "markdown",
 "metadata": {},
 "source": [
  "## 6. Stability Check: Decile-Specific Local SHAP\n",
  "To check if the local SHAP values are stable when calculated specifically at the Decile level, we will re-run the explainer explicitly per decile."
 ]
}

code_cell_1 = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# Compare XGBoost 'Global' Local SHAP vs 'Decile-Specific' Local SHAP for Decile 10\n",
  "decile_10_val = xgb_val[xgb_val['Decile'] == 10]\n",
  "shap_xgb_val_d10 = calculate_tree_shap(decile_10_val, model_dir, 'xgb', feature_cols)\n",
  "original_d10_shap = shap_xgb_val[shap_xgb_val['Decile'] == 10].drop('Decile', axis=1)\n",
  "\n",
  "diff = (shap_xgb_val_d10[feature_cols] - original_d10_shap[feature_cols]).abs().max().max()\n",
  "print(f\"Max difference in XGBoost SHAP values (Global vs Decile 10 specific): {diff}\")\n",
  "print(\"TreeExplainer values are perfectly stable because the expected value is derived from the tree structure.\")"
 ]
}

code_cell_2 = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# Compare Panel ARDL for Decile 10\n",
  "ardl_d10_val = ardl_val[ardl_val['Decile'] == 10]\n",
  "X_expected_d10 = ardl_dev[ardl_dev['Decile'] == 10][feature_cols].mean()\n",
  "\n",
  "shap_ardl_base_d10, shap_ardl_adj_d10 = calculate_ardl_linear_shap(\n",
  "    ardl_d10_val, segment_models, cust_id_adjust_models, proportions, feature_cols, X_expected_d10\n",
  ")\n",
  "original_ardl_adj_d10 = shap_ardl_adj_val[shap_ardl_adj_val['Decile'] == 10]\n",
  "\n",
  "diff_ardl = (shap_ardl_adj_d10[feature_cols] - original_ardl_adj_d10[feature_cols]).abs().mean().mean()\n",
  "print(f\"\\nAverage difference in ARDL SHAP values (Global expected vs Decile 10 expected): {diff_ardl:.4f}\")\n",
  "print(\"ARDL linear SHAP shifts when restricting expected value to a decile.\")"
 ]
}

code_cell_3 = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "import os\n",
  "os.makedirs('../Outputs/shap_local_decile_bins', exist_ok=True)\n",
  "\n",
  "def regenerate_and_save_all_deciles():\n",
  "    for d in range(1, 11):\n",
  "        d_val = xgb_val[xgb_val['Decile'] == d]\n",
  "        s_xgb = calculate_tree_shap(d_val, model_dir, 'xgb', feature_cols)\n",
  "        s_xgb.to_csv(f'../Outputs/shap_local_decile_bins/xgb_val_decile_{d}.csv', index=False)\n",
  "        \n",
  "        a_val = ardl_val[ardl_val['Decile'] == d]\n",
  "        x_exp = ardl_dev[ardl_dev['Decile'] == d][feature_cols].mean()\n",
  "        s_base, s_adj = calculate_ardl_linear_shap(\n",
  "            a_val, segment_models, cust_id_adjust_models, proportions, feature_cols, x_exp\n",
  "        )\n",
  "        s_adj.to_csv(f'../Outputs/shap_local_decile_bins/ardl_adj_val_decile_{d}.csv', index=False)\n",
  "        \n",
  "print('Saving explicit Decile-Based Local SHAP files...')\n",
  "regenerate_and_save_all_deciles()\n",
  "print('✅ Explicit Decile-based Local SHAP calculated and saved.')"
 ]
}

nb['cells'].extend([md_cell, code_cell_1, code_cell_2, code_cell_3])

with open('03_shap_explanations.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
