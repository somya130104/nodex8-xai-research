"""
TradeFraud: Cross-Architecture Consensus Logic Index (CLI) Audit
================================================================
Evaluates concordance and explainability agreement across:
  - 4 Baseline Models (XGBoost, Random Forest, LightGBM, Logistic Regression)
  - 3 Advanced Deep Models (Autoencoder, TabNet Classifier, Tabular ResNet)
Across 5 Attribution Modalities (TreeSHAP, LIME, Sparsemax Attention, Integrated Gradients, Reconstruction Saliency).
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

# Import consensus engine
sys.path.append("/Users/somyabhadada/nodex8/src")
from consensus_engine import kendalls_w, top_k_jaccard

BASE_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud"
XAI_DIR  = os.path.join(BASE_DIR, "XAI_Outputs")
CLI_DIR  = os.path.join(BASE_DIR, "Consensus_Outputs")

def main():
    os.makedirs(CLI_DIR, exist_ok=True)
    print("Loading Baseline and Advanced attributions...")
    
    # 1. Load Baseline SHAP & LIME
    shap_df = pd.read_csv(os.path.join(XAI_DIR, "shap_global.csv"))
    lime_df = pd.read_csv(os.path.join(XAI_DIR, "lime_global.csv"))
    adv_df  = pd.read_csv(os.path.join(XAI_DIR, "advanced_xai.csv"))
    
    # Standardize format: DataFrame with [Architecture_Explainer, Feature, Attribution]
    unified_records = []
    
    # Baseline SHAP
    for _, r in shap_df.iterrows():
        unified_records.append({
            "Model_Key": f"{r['Model']}_SHAP",
            "Category": "Baseline_Tree_Linear",
            "Family": r['Model'],
            "Explainer": "TreeSHAP/LinearSHAP",
            "Feature": r['Feature'],
            "Score": abs(float(r['Mean_Abs_SHAP']))
        })
        
    # Baseline LIME
    for _, r in lime_df.iterrows():
        unified_records.append({
            "Model_Key": f"{r['Model']}_LIME",
            "Category": "Baseline_Tree_Linear",
            "Family": r['Model'],
            "Explainer": "LIME",
            "Feature": r['Feature'],
            "Score": abs(float(r['Mean_Abs_LIME']))
        })
        
    # Advanced
    for _, r in adv_df.iterrows():
        unified_records.append({
            "Model_Key": f"{r['Model']}_{r['Explainer']}",
            "Category": "Advanced_Deep_Learning",
            "Family": r['Model'],
            "Explainer": r['Explainer'],
            "Feature": r['Feature'],
            "Score": abs(float(r['Attribution']))
        })
        
    df_unified = pd.DataFrame(unified_records)
    
    # Common features across all models
    feature_counts = df_unified.groupby("Feature")["Model_Key"].nunique()
    total_model_keys = df_unified["Model_Key"].nunique()
    common_features = sorted(feature_counts[feature_counts == total_model_keys].index.tolist())
    
    print(f"Total Model-Explainer configurations: {total_model_keys}")
    print(f"Common aligned features across all methods: {len(common_features)}")
    
    # Pivot matrix: rows = Model_Key, cols = Feature
    pivot_df = df_unified[df_unified["Feature"].isin(common_features)].pivot_table(
        index="Model_Key", columns="Feature", values="Score", aggfunc="mean"
    ).fillna(0)
    
    model_keys = list(pivot_df.index)
    n_models = len(model_keys)
    n_features = len(common_features)
    
    # Rankings matrix (1 = top rank)
    ranks_matrix = np.zeros((n_models, n_features))
    model_top_k = {}
    for i, mk in enumerate(model_keys):
        vals = pivot_df.loc[mk].values
        ranks_matrix[i] = rankdata(-vals, method="average")
        model_top_k[mk] = [common_features[idx] for idx in np.argsort(-vals)]
        
    # ── 1. Kendall's W across ALL models ─────────────────────────────────
    w_global = kendalls_w(ranks_matrix)
    print(f"\nGlobal Cross-Architecture Kendall's W: {w_global:.4f}")
    
    # ── 2. Pairwise Spearman Correlation Matrix ──────────────────────────
    spearman_matrix = pd.DataFrame(index=model_keys, columns=model_keys, dtype=float)
    for i, m1 in enumerate(model_keys):
        for j, m2 in enumerate(model_keys):
            r, _ = spearmanr(pivot_df.loc[m1], pivot_df.loc[m2])
            spearman_matrix.loc[m1, m2] = float(r)
            
    spearman_matrix.to_csv(os.path.join(CLI_DIR, "pairwise_spearman_matrix.csv"))
    
    # ── 3. Pairwise Top-10 Jaccard Overlap Matrix ────────────────────────
    jaccard_matrix = pd.DataFrame(index=model_keys, columns=model_keys, dtype=float)
    for m1 in model_keys:
        for m2 in model_keys:
            jaccard_matrix.loc[m1, m2] = top_k_jaccard(model_top_k[m1], model_top_k[m2], k=10)
            
    jaccard_matrix.to_csv(os.path.join(CLI_DIR, "pairwise_jaccard_top10_matrix.csv"))
    
    # ── 4. Baseline vs Advanced Concordance Analysis ────────────────────
    baseline_keys = [k for k in model_keys if "Model2" in k or "Model3" in k or "Model5" in k or "Model6" in k]
    advanced_keys = [k for k in model_keys if k not in baseline_keys]
    
    cross_pairs_spearman = []
    cross_pairs_jaccard = []
    
    for b in baseline_keys:
        for a in advanced_keys:
            cross_pairs_spearman.append(spearman_matrix.loc[b, a])
            cross_pairs_jaccard.append(jaccard_matrix.loc[b, a])
            
    mean_cross_spearman = float(np.mean(cross_pairs_spearman))
    mean_cross_jaccard = float(np.mean(cross_pairs_jaccard))
    norm_cross_spearman = max(0.0, (1.0 + mean_cross_spearman) / 2.0)
    
    # Baseline internal concordance
    base_submatrix = spearman_matrix.loc[baseline_keys, baseline_keys].values
    tri_b = np.triu_indices(len(baseline_keys), k=1)
    mean_base_spearman = float(np.mean(base_submatrix[tri_b]))
    
    # Advanced internal concordance
    adv_submatrix = spearman_matrix.loc[advanced_keys, advanced_keys].values
    tri_a = np.triu_indices(len(advanced_keys), k=1)
    mean_adv_spearman = float(np.mean(adv_submatrix[tri_a]))
    
    # ── 5. Cross-Architecture Consensus Logic Index (CLI) ───────────────
    # Weights:
    # 30% Overall Kendall's W
    # 35% Baseline-to-Advanced Spearman rank monotonicity
    # 35% Baseline-to-Advanced Top-10 Jaccard overlap
    cli_cross_arch = (
        0.30 * w_global +
        0.35 * norm_cross_spearman +
        0.35 * mean_cross_jaccard
    )
    
    # Top consensus features across ALL models (Borda Count)
    borda_scores = np.sum(ranks_matrix, axis=0) # lower sum of ranks = higher overall rank
    consensus_ranking = pd.DataFrame({
        "Feature": common_features,
        "Mean_Rank": borda_scores / n_models
    }).sort_values("Mean_Rank", ascending=True)
    consensus_ranking["Consensus_Rank"] = range(1, len(consensus_ranking) + 1)
    consensus_ranking.to_csv(os.path.join(CLI_DIR, "consensus_feature_ranking.csv"), index=False)
    
    # Save Summary JSON
    summary = {
        "dataset": "TradeFraud",
        "total_model_explainer_configs": n_models,
        "total_aligned_features": n_features,
        "kendalls_w_global": round(w_global, 4),
        "mean_baseline_internal_spearman": round(mean_base_spearman, 4),
        "mean_advanced_internal_spearman": round(mean_adv_spearman, 4),
        "mean_baseline_vs_advanced_spearman": round(mean_cross_spearman, 4),
        "mean_baseline_vs_advanced_top10_jaccard": round(mean_cross_jaccard, 4),
        "composite_cross_architecture_cli": round(cli_cross_arch, 4),
        "top_10_consensus_features": consensus_ranking.head(10)["Feature"].tolist()
    }
    
    with open(os.path.join(CLI_DIR, "cross_architecture_cli_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\n=======================================================")
    print(" TradeFraud Cross-Architecture Consensus Audit Summary")
    print("=======================================================")
    print(f" Global Kendall's W (Concordance):       {w_global:.4f}")
    print(f" Baseline Internal Spearman Rho:        {mean_base_spearman:.4f}")
    print(f" Advanced Internal Spearman Rho:        {mean_adv_spearman:.4f}")
    print(f" Tree-vs-Deep Mean Spearman Rho:        {mean_cross_spearman:.4f}")
    print(f" Tree-vs-Deep Top-10 Jaccard Overlap:   {mean_cross_jaccard:.4f}")
    print(f" COMPOSITE CROSS-ARCH CLI SCORE:        {cli_cross_arch:.4f}")
    print("=======================================================")
    print("\nTop 10 Universal Consensus Risk Drivers:")
    for idx, r in consensus_ranking.head(10).iterrows():
        print(f"  {r['Consensus_Rank']:2d}. {r['Feature']:30s} (Mean Rank: {r['Mean_Rank']:.2f})")
        
    print(f"\n✅ All outputs saved in {CLI_DIR}")

if __name__ == "__main__":
    main()
