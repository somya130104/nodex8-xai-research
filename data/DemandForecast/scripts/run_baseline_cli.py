"""
Compute Baseline Consensus Logic Index (CLI) for DemandForecast
"""

import os
import sys
import json
import pandas as pd

# Add repository root to path
sys.path.append("/Users/somyabhadada/nodex8")
from src.consensus_engine import compute_consensus_report

XAI_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/XAI_Outputs/"
OUT_DIR = os.path.join(XAI_DIR, "consensus")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading SHAP and LIME global importance for DemandForecast...")
    
    shap_path = os.path.join(XAI_DIR, "shap_global.csv")
    lime_path = os.path.join(XAI_DIR, "lime_global.csv")
    
    if not os.path.exists(shap_path) or not os.path.exists(lime_path):
        print("Error: SHAP or LIME global files missing!")
        return
        
    shap_df = pd.read_csv(shap_path)
    lime_df = pd.read_csv(lime_path)
    
    print(f"Loaded {len(shap_df)} SHAP entries, {len(lime_df)} LIME entries.")
    
    report = compute_consensus_report(shap_df, lime_df, top_k_list=[5, 10, 20])
    
    print("\n" + "="*60)
    print("  NODEX8 BASELINE CONSENSUS LOGIC INDEX (DEMAND FORECAST)")
    print("="*60)
    print(f"Models Evaluated                   : {', '.join(report['models'])}")
    print(f"Common Features                    : {report['n_features']}")
    print(f"Kendall's W (SHAP Inter-Model)     : {report['kendalls_w_shap']}")
    print(f"Kendall's W (LIME Inter-Model)     : {report['kendalls_w_lime']}")
    print(f"Mean Pairwise Spearman (SHAP)      : {report['mean_pairwise_spearman_shap']}")
    print(f"Mean Pairwise Spearman (LIME)      : {report['mean_pairwise_spearman_lime']}")
    print(f"Mean Cross-Explainer Spearman (rho): {report['mean_cross_spearman_shap_lime']}")
    print(f"Mean Cross-Explainer Top-10 Jaccard: {report['mean_cross_jaccard_top10']}")
    print("-"*60)
    print(f"COMPOSITE BASELINE CLI SCORE       : {report['composite_cli']} / 1.0000")
    print("="*60)
    
    # Save matrices and report
    report['spearman_shap_matrix'].to_csv(os.path.join(OUT_DIR, "spearman_shap_matrix.csv"))
    report['spearman_lime_matrix'].to_csv(os.path.join(OUT_DIR, "spearman_lime_matrix.csv"))
    report['jaccard_shap_dict']['top_10'].to_csv(os.path.join(OUT_DIR, "jaccard_top10_shap_matrix.csv"))
    report['jaccard_lime_dict']['top_10'].to_csv(os.path.join(OUT_DIR, "jaccard_top10_lime_matrix.csv"))
    report['cross_spearman_df'].to_csv(os.path.join(OUT_DIR, "cross_explainer_spearman.csv"))
    report['cross_jaccard_df'].to_csv(os.path.join(OUT_DIR, "cross_explainer_jaccard.csv"))
    
    summary = {
        "dataset": "DemandForecast",
        "models": report['models'],
        "n_features": report['n_features'],
        "kendalls_w_shap": report['kendalls_w_shap'],
        "kendalls_w_lime": report['kendalls_w_lime'],
        "mean_pairwise_spearman_shap": report['mean_pairwise_spearman_shap'],
        "mean_pairwise_spearman_lime": report['mean_pairwise_spearman_lime'],
        "mean_cross_spearman_shap_lime": report['mean_cross_spearman_shap_lime'],
        "mean_cross_jaccard_top10": report['mean_cross_jaccard_top10'],
        "composite_cli": report['composite_cli']
    }
    with open(os.path.join(OUT_DIR, "baseline_cli_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\n✅ All baseline consensus outputs written to {OUT_DIR}")

if __name__ == "__main__":
    main()
