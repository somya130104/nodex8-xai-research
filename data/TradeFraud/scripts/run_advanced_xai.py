"""
Extract Explainability Attributions for Advanced Deep Architectures in TradeFraud:
  - Model 4: Autoencoder Reconstruction Error Saliency
  - Model 7: TabNet Attentive Feature Masks
  - Model 8: Tabular ResNet Integrated Gradients
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier

# Config
DATA_DIR   = "/Users/somyabhadada/nodex8/data/TradeFraud/Modeling_Data_Enriched/"
MODEL_DIR  = "/Users/somyabhadada/nodex8/data/TradeFraud/Model_Outputs/"
OUTPUT_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/XAI_Outputs/"

EXCLUDE_COLS = ["Client", "Client_ID", "Date", "Dispute", "Segment", "Activeness_Segment"]

# ResNet definition
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x + self.block(x))

class TabularResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_blocks=3, dropout=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)

def compute_integrated_gradients(model, x, baseline, steps=25):
    """
    Computes Integrated Gradients for a batch of inputs x against baseline.
    Path: gamma(alpha) = baseline + alpha * (x - baseline)
    """
    model.eval()
    x_diff = x - baseline
    total_grads = torch.zeros_like(x)
    
    for step in range(steps + 1):
        alpha = float(step) / steps
        interpolated = baseline + alpha * x_diff
        interpolated.requires_grad_(True)
        
        logits = model(interpolated)
        probs = torch.sigmoid(logits)
        
        grads = torch.autograd.grad(probs.sum(), interpolated)[0]
        if step == 0 or step == steps:
            total_grads += 0.5 * grads
        else:
            total_grads += grads
            
    avg_grads = total_grads / steps
    ig = x_diff * avg_grads
    return ig.detach()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading test/evaluation sample data...")
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val_oos.csv"))
    feat_cols = [c for c in val_df.columns if c not in EXCLUDE_COLS]
    
    # Sample 200 instances for attribution
    np.random.seed(42)
    sample_indices = np.random.choice(len(val_df), min(200, len(val_df)), replace=False)
    val_sample = val_df.iloc[sample_indices]
    X_val_sample_raw = val_sample[feat_cols].fillna(0).values.astype(np.float32)
    
    results = []
    
    # ── 1. Model 4 Autoencoder ──────────────────────────────────────────
    print("\n--- Extracting Model 4 Autoencoder Attributions ---")
    ae_fi_path = os.path.join(MODEL_DIR, "Model4_Autoencoder/feature_importance.csv")
    if os.path.exists(ae_fi_path):
        ae_fi = pd.read_csv(ae_fi_path)
        # Using Gap (Dispute-Normal reconstruction error) as the importance
        for _, row in ae_fi.iterrows():
            f = row['Feature']
            if f in feat_cols:
                results.append({
                    "Model": "Model4_Autoencoder",
                    "Explainer": "Recon_Error_Saliency",
                    "Feature": f,
                    "Attribution": max(0.0, float(row['Gap (Dispute-Normal)']))
                })
        print(f"  Extracted {len(ae_fi)} Autoencoder feature attributions.")
        
    # ── 2. Model 7 TabNet ───────────────────────────────────────────────
    print("\n--- Extracting Model 7 TabNet Attributions ---")
    tabnet_fi_path = os.path.join(MODEL_DIR, "Model7_TabNet/feature_importance.csv")
    if os.path.exists(tabnet_fi_path):
        tn_fi = pd.read_csv(tabnet_fi_path)
        for _, row in tn_fi.iterrows():
            results.append({
                "Model": "Model7_TabNet",
                "Explainer": "Sparsemax_Attention_Global",
                "Feature": row["Feature"],
                "Attribution": float(row["Importance"])
            })
            
    tabnet_model_path = os.path.join(MODEL_DIR, "Model7_TabNet/tabnet_model.zip.zip")
    if not os.path.exists(tabnet_model_path):
        tabnet_model_path = os.path.join(MODEL_DIR, "Model7_TabNet/tabnet_model.zip")
    
    if os.path.exists(tabnet_model_path):
        clf = TabNetClassifier()
        clf.load_model(tabnet_model_path)
        
        # Instance-level explain masks
        explain_matrix, masks = clf.explain(X_val_sample_raw)
        mean_mask_attr = np.mean(explain_matrix, axis=0)
        for f, imp in zip(feat_cols, mean_mask_attr):
            results.append({
                "Model": "Model7_TabNet",
                "Explainer": "Instance_Mask_Attribution",
                "Feature": f,
                "Attribution": float(imp)
            })
        print("  Extracted TabNet Sparsemax global and mask attributions.")
        
    # ── 3. Model 8 Tabular ResNet ───────────────────────────────────────
    print("\n--- Extracting Model 8 Tabular ResNet Integrated Gradients ---")
    resnet_path = os.path.join(MODEL_DIR, "Model8_ResNet/best_resnet.pt")
    scaler_path = os.path.join(MODEL_DIR, "Model8_ResNet/scaler.pkl")
    
    if os.path.exists(resnet_path) and os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
            
        model = TabularResNet(input_dim=len(feat_cols), hidden_dim=128, n_blocks=3, dropout=0.2)
        model.load_state_dict(torch.load(resnet_path))
        model.eval()
        
        X_sample_norm = scaler.transform(X_val_sample_raw)
        x_tensor = torch.tensor(X_sample_norm, dtype=torch.float32)
        baseline = torch.zeros_like(x_tensor)  # Mean in standardized space is 0
        
        ig_attributions = compute_integrated_gradients(model, x_tensor, baseline, steps=25)
        mean_abs_ig = torch.mean(torch.abs(ig_attributions), dim=0).numpy()
        
        for f, imp in zip(feat_cols, mean_abs_ig):
            results.append({
                "Model": "Model8_ResNet",
                "Explainer": "Integrated_Gradients",
                "Feature": f,
                "Attribution": float(imp)
            })
        print("  Extracted Tabular ResNet Integrated Gradients attributions.")
        
    df_out = pd.DataFrame(results)
    df_out.to_csv(os.path.join(OUTPUT_DIR, "advanced_xai.csv"), index=False)
    print(f"\n✅ Advanced XAI extraction complete! Saved {len(df_out)} rows to {OUTPUT_DIR}/advanced_xai.csv")

if __name__ == "__main__":
    main()
