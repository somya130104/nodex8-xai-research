"""
Model 8 — Deep Tabular ResNet (Residual MLP) for TradeFraud
============================================================
Deep non-linear residual network architecture with skip connections and LayerNorm
optimized for extreme tabular class imbalance (0.23% dispute rate).

Outputs:
  - Model checkpoint (.pt)
  - Scaler (.pkl)
  - Performance scorecard across all 6 validation / OOT datasets
  - Gradient-based feature saliency attributions
"""
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score, confusion_matrix)

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

DATA_DIR   = "/Users/somyabhadada/nodex8/data/TradeFraud/Modeling_Data_Enriched/"
OUTPUT_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/Model_Outputs/Model8_ResNet/"

EXCLUDE_COLS = ["Client", "Client_ID", "Date", "Dispute", "Segment", "Activeness_Segment"]
EVAL_SETS = [
    ("train.csv",        "Train (all)"),
    ("val_oos.csv",      "Validation OOS"),
    ("internal_oot.csv", "Internal OOT (Aug–Sep)"),
    ("oot_202510.csv",   "OOT Oct 2025"),
    ("oot_202511.csv",   "OOT Nov 2025"),
    ("oot_202512.csv",   "OOT Dec 2025"),
]

# ── Residual Block ────────────────────────────────────────────────────
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

def load_data(fname, feat_cols=None):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    y = (df["Dispute"] == "D").astype(np.float32).values
    if feat_cols is None:
        feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    return X, y, feat_cols

def evaluate(y_true, y_score, label):
    n_pos = int(y_true.sum())
    try:
        auc = roc_auc_score(y_true, y_score) if 0 < n_pos < len(y_true) else np.nan
    except Exception:
        auc = np.nan
    gini = 2 * auc - 1 if not np.isnan(auc) else np.nan
    
    try:
        auc_pr = average_precision_score(y_true, y_score) if 0 < n_pos < len(y_true) else np.nan
    except Exception:
        auc_pr = np.nan
        
    # KS
    try:
        order = np.argsort(-y_score)
        cum_pos = np.cumsum(y_true[order]) / max(n_pos, 1)
        cum_neg = np.cumsum(1 - y_true[order]) / max(len(y_true) - n_pos, 1)
        ks = float(np.max(np.abs(cum_pos - cum_neg)))
    except Exception:
        ks = np.nan
        
    n_top = max(1, int(len(y_true) * 0.10))
    top_idx = np.argsort(-y_score)[:n_top]
    top_decile_recall = float(y_true[top_idx].sum() / max(n_pos, 1))
    
    thresholds = np.linspace(0.01, 0.99, 99)
    best_th, best_f1, best_prec, best_rec = 0.5, 0.0, 0.0, 0.0
    for th in thresholds:
        pred = (y_score >= th).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_th = th
            best_prec = precision_score(y_true, pred, zero_division=0)
            best_rec = recall_score(y_true, pred, zero_division=0)
            
    pred_best = (y_score >= best_th).astype(int)
    cm = confusion_matrix(y_true, pred_best)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    return {
        "Dataset": label,
        "Rows": len(y_true),
        "Disputes": n_pos,
        "AUC_ROC": round(auc, 4) if not np.isnan(auc) else None,
        "AUC_PR": round(auc_pr, 4) if not np.isnan(auc_pr) else None,
        "Gini": round(gini, 4) if not np.isnan(gini) else None,
        "KS": round(ks, 4) if not np.isnan(ks) else None,
        "Top_Decile_Recall": round(top_decile_recall, 4),
        "Best_Threshold": round(best_th, 4),
        "Precision": round(best_prec, 4),
        "Recall": round(best_rec, 4),
        "F1": round(best_f1, 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
    }

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading data for Tabular ResNet...")
    X_train_raw, y_train, feat_cols = load_data("train.csv")
    X_val_raw, y_val, _ = load_data("val_oos.csv", feat_cols)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUTPUT_DIR, "features.json"), "w") as f:
        json.dump(feat_cols, f, indent=4)
        
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset   = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    # Class weight
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    model = TabularResNet(input_dim=len(feat_cols), hidden_dim=128, n_blocks=3, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_val_auc = 0.0
    patience_cnt = 0
    patience = 8
    
    print("\nTraining Tabular ResNet...")
    for epoch in range(1, 35):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(by)
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                probs = torch.sigmoid(model(bx)).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(by.numpy())
                
        val_auc = roc_auc_score(val_targets, val_preds)
        scheduler.step(val_auc)
        
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_resnet.pt"))
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
    # Load best model
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_resnet.pt")))
    model.eval()
    
    print("\nEvaluating across sets...")
    scorecards = []
    for fname, label in EVAL_SETS:
        X_eval_raw, y_eval, _ = load_data(fname, feat_cols)
        X_eval = scaler.transform(X_eval_raw)
        eval_loader = DataLoader(TensorDataset(torch.tensor(X_eval)), batch_size=1024, shuffle=False)
        preds = []
        with torch.no_grad():
            for (bx,) in eval_loader:
                bx = bx.to(device)
                preds.extend(torch.sigmoid(model(bx)).cpu().numpy())
        preds = np.array(preds)
        metrics = evaluate(y_eval, preds, label)
        scorecards.append(metrics)
        print(f"  {label:25s}: AUC={metrics['AUC_ROC']}, Gini={metrics['Gini']}, Top10_Recall={metrics['Top_Decile_Recall']}")
        
    df_score = pd.DataFrame(scorecards)
    df_score.to_csv(os.path.join(OUTPUT_DIR, "performance_scorecard.csv"), index=False)
    
    # Compute Input Saliency Gradients (Deep Feature Importance)
    print("\nComputing Gradient Saliency Importance...")
    sample_x = torch.tensor(X_train[:2000], requires_grad=True).to(device)
    model.zero_grad()
    out = model(sample_x).sum()
    out.backward()
    saliency = torch.mean(torch.abs(sample_x.grad), dim=0).cpu().numpy()
    
    df_fi = pd.DataFrame({
        "Feature": feat_cols,
        "Importance": saliency
    }).sort_values("Importance", ascending=False)
    df_fi.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    print("\nTop 10 features by Gradient Saliency:")
    print(df_fi.head(10).to_string(index=False))
    
    print(f"\n✅ Model 8 Tabular ResNet complete! Outputs saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    run()
