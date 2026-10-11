"""
NodeX8 Sovereign AI Governance - Consensus Logic Index (CLI) Engine
Computes inter-model and inter-explainer concordance across attribution methodologies.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata


def kendalls_w(rankings_matrix):
    """
    Computes Kendall's W (Coefficient of Concordance) for m raters and n items.
    
    Parameters
    ----------
    rankings_matrix : np.ndarray
        Shape (m, n) where m is the number of models/raters,
        and n is the number of features/items. Each row contains ranks for features.
        
    Returns
    -------
    float : Kendall's W in [0, 1]
    """
    m, n = rankings_matrix.shape
    if m <= 1 or n <= 1:
        return 1.0
    
    # Sum of ranks for each feature across all m raters
    rank_sums = np.sum(rankings_matrix, axis=0)
    mean_rank_sum = np.mean(rank_sums)
    
    # Sum of squared deviations
    s = np.sum((rank_sums - mean_rank_sum) ** 2)
    
    # Maximum possible sum of squared deviations
    denominator = (m ** 2) * (n ** 3 - n) / 12.0
    
    w = s / denominator if denominator > 0 else 0.0
    return float(np.clip(w, 0.0, 1.0))


def top_k_jaccard(features_a, features_b, k=10):
    """
    Computes Jaccard similarity between the top-k features of two rankings.
    """
    set_a = set(features_a[:k])
    set_b = set(features_b[:k])
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return float(intersection / union) if union > 0 else 1.0


def compute_consensus_report(shap_df, lime_df, top_k_list=[5, 10, 20]):
    """
    Computes comprehensive Consensus Logic Index (CLI) across SHAP and LIME attributions.
    
    Parameters
    ----------
    shap_df : pd.DataFrame
        Columns: ['Model', 'Feature', 'Mean_Abs_SHAP']
    lime_df : pd.DataFrame
        Columns: ['Model', 'Feature', 'Mean_Abs_LIME']
    top_k_list : list of int
        List of k values for Top-k Jaccard similarity.
        
    Returns
    -------
    dict containing:
        - 'kendalls_w_shap': float
        - 'kendalls_w_lime': float
        - 'pairwise_spearman_shap': pd.DataFrame
        - 'pairwise_spearman_lime': pd.DataFrame
        - 'pairwise_jaccard_shap': dict of pd.DataFrame
        - 'pairwise_jaccard_lime': dict of pd.DataFrame
        - 'cross_explainer_spearman': pd.DataFrame (SHAP vs LIME per model)
        - 'cross_explainer_jaccard': dict of pd.DataFrame
        - 'composite_cli': float
        - 'detailed_metrics': dict
    """
    models = sorted(list(set(shap_df['Model'].unique()).intersection(set(lime_df['Model'].unique()))))
    all_features = sorted(list(set(shap_df['Feature'].unique()).intersection(set(lime_df['Feature'].unique()))))
    n_features = len(all_features)
    n_models = len(models)
    
    # 1. Pivot matrices (rows: Model, cols: Feature)
    shap_pivot = shap_df.pivot(index='Model', columns='Feature', values='Mean_Abs_SHAP').reindex(index=models, columns=all_features).fillna(0)
    lime_pivot = lime_df.pivot(index='Model', columns='Feature', values='Mean_Abs_LIME').reindex(index=models, columns=all_features).fillna(0)
    
    # Convert attributions to ranks (higher importance -> rank 1, so ascending rank on negative values)
    shap_ranks = np.zeros((n_models, n_features))
    lime_ranks = np.zeros((n_models, n_features))
    
    model_top_k_shap = {}
    model_top_k_lime = {}
    
    for i, m in enumerate(models):
        shap_vals = shap_pivot.loc[m].values
        lime_vals = lime_pivot.loc[m].values
        
        # Rank: 1 is top feature
        shap_ranks[i] = rankdata(-shap_vals, method='average')
        lime_ranks[i] = rankdata(-lime_vals, method='average')
        
        # Sorted feature names
        model_top_k_shap[m] = [all_features[idx] for idx in np.argsort(-shap_vals)]
        model_top_k_lime[m] = [all_features[idx] for idx in np.argsort(-lime_vals)]

    # 2. Kendall's W across models
    w_shap = kendalls_w(shap_ranks)
    w_lime = kendalls_w(lime_ranks)
    
    # 3. Pairwise Spearman rank correlation across models
    spearman_shap = pd.DataFrame(index=models, columns=models, dtype=float)
    spearman_lime = pd.DataFrame(index=models, columns=models, dtype=float)
    
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            r_s, _ = spearmanr(shap_pivot.loc[m1], shap_pivot.loc[m2])
            r_l, _ = spearmanr(lime_pivot.loc[m1], lime_pivot.loc[m2])
            spearman_shap.loc[m1, m2] = float(r_s)
            spearman_lime.loc[m1, m2] = float(r_l)
            
    # 4. Pairwise Top-K Jaccard similarity across models
    jaccard_shap = {}
    jaccard_lime = {}
    for k in top_k_list:
        df_j_shap = pd.DataFrame(index=models, columns=models, dtype=float)
        df_j_lime = pd.DataFrame(index=models, columns=models, dtype=float)
        for m1 in models:
            for m2 in models:
                df_j_shap.loc[m1, m2] = top_k_jaccard(model_top_k_shap[m1], model_top_k_shap[m2], k=k)
                df_j_lime.loc[m1, m2] = top_k_jaccard(model_top_k_lime[m1], model_top_k_lime[m2], k=k)
        jaccard_shap[f"top_{k}"] = df_j_shap
        jaccard_lime[f"top_{k}"] = df_j_lime
        
    # 5. Cross-explainer concordance (SHAP vs LIME for each model)
    cross_spearman = {}
    cross_jaccard = {f"top_{k}": {} for k in top_k_list}
    
    for m in models:
        r_cross, _ = spearmanr(shap_pivot.loc[m], lime_pivot.loc[m])
        cross_spearman[m] = float(r_cross)
        for k in top_k_list:
            j_cross = top_k_jaccard(model_top_k_shap[m], model_top_k_lime[m], k=k)
            cross_jaccard[f"top_{k}"][m] = float(j_cross)
            
    cross_spearman_df = pd.DataFrame.from_dict(cross_spearman, orient='index', columns=['Spearman_rho_SHAP_vs_LIME'])
    cross_jaccard_df = pd.DataFrame(cross_jaccard)
    
    # 6. Composite Consensus Logic Index (CLI) Formulation
    # Mean off-diagonal pairwise Spearman correlation across models
    upper_tri_mask = np.triu_indices(n_models, k=1)
    mean_pairwise_spearman_shap = float(np.mean(spearman_shap.values[upper_tri_mask])) if n_models > 1 else 1.0
    mean_pairwise_spearman_lime = float(np.mean(spearman_lime.values[upper_tri_mask])) if n_models > 1 else 1.0
    
    # Mean cross-explainer Spearman correlation
    mean_cross_spearman = float(np.mean(list(cross_spearman.values())))
    
    # Normalized correlations to [0, 1] range: (1 + rho) / 2
    norm_spearman_shap = max(0.0, (1.0 + mean_pairwise_spearman_shap) / 2.0)
    norm_spearman_lime = max(0.0, (1.0 + mean_pairwise_spearman_lime) / 2.0)
    norm_cross_spearman = max(0.0, (1.0 + mean_cross_spearman) / 2.0)
    
    # Mean Top-10 Jaccard across models and cross-explainers
    mean_jaccard_top10_shap = float(np.mean(jaccard_shap["top_10"].values[upper_tri_mask])) if n_models > 1 else 1.0
    mean_jaccard_top10_lime = float(np.mean(jaccard_lime["top_10"].values[upper_tri_mask])) if n_models > 1 else 1.0
    mean_cross_jaccard_top10 = float(np.mean(list(cross_jaccard["top_10"].values())))
    
    # Composite CLI: weighted combination of inter-model agreement (SHAP & LIME) and inter-explainer fidelity
    # 25% W_SHAP, 25% W_LIME, 25% Cross-Explainer Monotonicity, 25% Cross-Explainer Top-10 Overlap
    composite_cli = (
        0.25 * w_shap +
        0.25 * w_lime +
        0.25 * norm_cross_spearman +
        0.25 * mean_cross_jaccard_top10
    )
    
    return {
        "models": models,
        "n_features": n_features,
        "kendalls_w_shap": round(w_shap, 4),
        "kendalls_w_lime": round(w_lime, 4),
        "mean_pairwise_spearman_shap": round(mean_pairwise_spearman_shap, 4),
        "mean_pairwise_spearman_lime": round(mean_pairwise_spearman_lime, 4),
        "mean_cross_spearman_shap_lime": round(mean_cross_spearman, 4),
        "mean_cross_jaccard_top10": round(mean_cross_jaccard_top10, 4),
        "composite_cli": round(composite_cli, 4),
        "spearman_shap_matrix": spearman_shap,
        "spearman_lime_matrix": spearman_lime,
        "jaccard_shap_dict": jaccard_shap,
        "jaccard_lime_dict": jaccard_lime,
        "cross_spearman_df": cross_spearman_df,
        "cross_jaccard_df": cross_jaccard_df,
        "model_top_k_shap": model_top_k_shap,
        "model_top_k_lime": model_top_k_lime
    }
