"""
Extract Explainability Attributions for Advanced Deep Architectures in DemandForecast:
  - TabNet Regressor: Sparsemax Attention Importances & Mask Attributions
  - DemandLSTM Regressor: Integrated Gradients & Temporal Saliency Attributions
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetRegressor

DATA_DIR   = "/Users/somyabhadada/nodex8/data/DemandForecast/samples/"
MODEL_DIR  = "/Users/somyabhadada/nodex8/data/DemandForecast/Model_Outputs/"
OUTPUT_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/XAI_Outputs/"

EXCLUDE_COLS = ["RowNum", "Date", "UPC", "COMP_STR_NO", "QUANTITY", "BRAND", "RET_LIR_ID"]
CATEGORICAL_COLS = ["PROMOTION_FLG", "C_PROMOTION_FLG"]

class DemandLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        y = self.fc(out[:, -1, :])
        return torch.relu(y).squeeze(-1)

def compute_integrated_gradients_lstm(model, x, baseline, steps=25):
    model.eval()
    x_diff = x - baseline
    total_grads = torch.zeros_like(x)
    
    for step in range(steps + 1):
        alpha = float(step) / steps
        interpolated = baseline + alpha * x_diff
        interpolated.requires_grad_(True)
        
        preds = model(interpolated)
        grads = torch.autograd.grad(preds.sum(), interpolated)[0]
        if step == 0 or step == steps:
            total_grads += 0.5 * grads
        else:
            total_grads += grads
            
    avg_grads = total_grads / steps
    ig = x_diff * avg_grads
    return ig.detach()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading test sample data for DemandForecast...")
    oos_df = pd.read_csv(os.path.join(DATA_DIR, "oos.csv"))
    for col in CATEGORICAL_COLS:
        if col in oos_df.columns:
            oos_df[col] = oos_df[col].map({'Y': 1, 'N': 0, '1': 1, '0': 0, 1: 1, 0: 0}).fillna(0)
            
    feat_cols = [c for c in oos_df.columns if c not in EXCLUDE_COLS]
    
    np.random.seed(42)
    sample_indices = np.random.choice(len(oos_df), min(200, len(oos_df)), replace=False)
    sample_df = oos_df.iloc[sample_indices]
    X_sample_raw = sample_df[feat_cols].fillna(0).values.astype(np.float32)
    
    results = []
    
    # ── 1. TabNet Regressor ─────────────────────────────────────────────
    print("\n--- Extracting TabNet Regressor Attributions ---")
    tn_fi_path = os.path.join(MODEL_DIR, "TabNet/feature_importance.csv")
    if os.path.exists(tn_fi_path):
        tn_fi = pd.read_csv(tn_fi_path)
        for _, row in tn_fi.iterrows():
            results.append({
                "Model": "TabNet",
                "Explainer": "Sparsemax_Attention_Global",
                "Feature": row["Feature"],
                "Attribution": float(row["Importance"])
            })
            
    tn_model_path = os.path.join(MODEL_DIR, "TabNet/tabnet_model.zip.zip")
    if not os.path.exists(tn_model_path):
        tn_model_path = os.path.join(MODEL_DIR, "TabNet/tabnet_model.zip")
        
    if os.path.exists(tn_model_path):
        reg = TabNetRegressor()
        reg.load_model(tn_model_path)
        explain_matrix, masks = reg.explain(X_sample_raw)
        mean_mask_attr = np.mean(explain_matrix, axis=0)
        for f, imp in zip(feat_cols, mean_mask_attr):
            results.append({
                "Model": "TabNet",
                "Explainer": "Instance_Mask_Attribution",
                "Feature": f,
                "Attribution": float(imp)
            })
        print("  Extracted TabNet Sparsemax global and mask attributions.")
        
    # ── 2. DemandLSTM Regressor ─────────────────────────────────────────
    print("\n--- Extracting DemandLSTM Integrated Gradients ---")
    lstm_model_path = os.path.join(MODEL_DIR, "LSTM/best_lstm.pt")
    scaler_path = os.path.join(MODEL_DIR, "LSTM/scaler.pkl")
    
    if os.path.exists(lstm_model_path) and os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
            
        model = DemandLSTM(input_dim=len(feat_cols), hidden_dim=64, num_layers=2, dropout=0.2)
        model.load_state_dict(torch.load(lstm_model_path))
        model.eval()
        
        X_sample_norm = scaler.transform(X_sample_raw)
        x_tensor = torch.tensor(X_sample_norm, dtype=torch.float32).unsqueeze(1)
        baseline = torch.zeros_like(x_tensor)
        
        ig_attributions = compute_integrated_gradients_lstm(model, x_tensor, baseline, steps=25)
        mean_abs_ig = torch.mean(torch.abs(ig_attributions), dim=[0, 1]).numpy()
        
        for f, imp in zip(feat_cols, mean_abs_ig):
            results.append({
                "Model": "DemandLSTM",
                "Explainer": "Integrated_Gradients",
                "Feature": f,
                "Attribution": float(imp)
            })
        print("  Extracted DemandLSTM Integrated Gradients attributions.")
        
    df_out = pd.DataFrame(results)
    df_out.to_csv(os.path.join(OUTPUT_DIR, "advanced_xai.csv"), index=False)
    print(f"\n✅ Advanced XAI extraction complete! Saved {len(df_out)} rows to {OUTPUT_DIR}/advanced_xai.csv")

if __name__ == "__main__":
    main()
