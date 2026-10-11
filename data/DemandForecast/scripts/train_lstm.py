"""
Deep Temporal LSTM Regressor for DemandForecast
===============================================
Recurrent neural architecture utilizing sequential price, promo, and lag features
for zero-inflated continuous retail sales and demand forecasting.

Outputs:
  - Model checkpoint (.pt)
  - Scaler (.pkl)
  - Performance scorecard across DEV, OOS, and OOT datasets
  - Gradient saliency feature attributions
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

DATA_DIR   = "/Users/somyabhadada/nodex8/data/DemandForecast/samples/"
OUTPUT_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/Model_Outputs/LSTM/"

EXCLUDE_COLS = ["RowNum", "Date", "UPC", "COMP_STR_NO", "QUANTITY", "BRAND", "RET_LIR_ID"]
CATEGORICAL_COLS = ["PROMOTION_FLG", "C_PROMOTION_FLG"]

EVAL_SETS = [
    ("dev.csv", "DEV"),
    ("oos.csv", "OOS"),
    ("oot_01_201212.csv", "OOT_201212"),
    ("oot_02_201301.csv", "OOT_201301"),
    ("oot_03_201302.csv", "OOT_201302")
]

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
        # x shape: (batch_size, 1, input_dim)
        out, _ = self.lstm(x)
        # out shape: (batch_size, 1, hidden_dim)
        y = self.fc(out[:, -1, :])
        return torch.relu(y).squeeze(-1)

def load_dataset(fname):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].map({'Y': 1, 'N': 0, '1': 1, '0': 0, 1: 1, 0: 0}).fillna(0)
            
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    y = df["QUANTITY"].fillna(0).values.astype(np.float32)
    return X, y, feat_cols

def evaluate(model, X_eval, y_true, device, label):
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X_eval).unsqueeze(1)), batch_size=1024, shuffle=False)
    preds = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            preds.extend(model(bx).cpu().numpy())
    pred = np.array(preds)
    pred = np.maximum(0, pred)
    
    mae = mean_absolute_error(y_true, pred)
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    r2 = r2_score(y_true, pred)
    
    mask = y_true > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - pred[mask]) / y_true[mask]))
    else:
        mape = np.nan
        
    hits = np.abs(y_true - pred) <= (0.2 * np.abs(y_true))
    acc_20 = hits.mean()
    
    return {
        "Dataset": label,
        "Rows": len(y_true),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4) if not np.isnan(mape) else None,
        "R2": round(r2, 4),
        "Accuracy_20pct": round(acc_20, 4)
    }

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading data for DemandLSTM...")
    X_dev_raw, y_dev, feat_cols = load_dataset("dev.csv")
    
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_dev_raw, y_dev, test_size=0.15, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUTPUT_DIR, "features.json"), "w") as f:
        json.dump(feat_cols, f, indent=4)
        
    train_dataset = TensorDataset(torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train))
    val_dataset   = TensorDataset(torch.tensor(X_val).unsqueeze(1), torch.tensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    model = DemandLSTM(input_dim=len(feat_cols), hidden_dim=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_rmse = float('inf')
    patience = 6
    patience_cnt = 0
    
    print("\nTraining DemandLSTM...")
    for epoch in range(1, 30):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(by)
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                val_loss += criterion(pred, by).item() * len(by)
        val_loss /= len(val_dataset)
        val_rmse = np.sqrt(val_loss)
        scheduler.step(val_rmse)
        
        print(f"Epoch {epoch:2d} | Train MSE: {train_loss:.4f} | Val RMSE: {val_rmse:.4f}")
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_lstm.pt"))
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
    # Load best model
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_lstm.pt")))
    
    print("\nEvaluating across sets...")
    scorecards = []
    for fname, lbl in EVAL_SETS:
        if not os.path.exists(os.path.join(DATA_DIR, fname)):
            continue
        X_raw, y_eval, _ = load_dataset(fname)
        X_eval = scaler.transform(X_raw)
        metrics = evaluate(model, X_eval, y_eval, device, lbl)
        scorecards.append(metrics)
        print(f"  {lbl:12s}: MAE={metrics['MAE']}, RMSE={metrics['RMSE']}, R2={metrics['R2']}, Acc_20%={metrics['Accuracy_20pct']}")
        
    df_score = pd.DataFrame(scorecards)
    df_score.to_csv(os.path.join(OUTPUT_DIR, "performance_scorecard.csv"), index=False)
    
    # Feature Saliency
    print("\nComputing Gradient Saliency Importance...")
    sample_x = torch.tensor(X_train[:1000], dtype=torch.float32).unsqueeze(1).to(device)
    sample_x.requires_grad_(True)
    model.zero_grad()
    out = model(sample_x).sum()
    out.backward()
    saliency = torch.mean(torch.abs(sample_x.grad), dim=[0, 1]).detach().cpu().numpy()
    
    df_fi = pd.DataFrame({
        "Feature": feat_cols,
        "Importance": saliency
    }).sort_values("Importance", ascending=False)
    df_fi.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    print("\nTop 10 features by Gradient Saliency:")
    print(df_fi.head(10).to_string(index=False))
    
    print(f"\n✅ DemandLSTM complete! Outputs saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    run()
