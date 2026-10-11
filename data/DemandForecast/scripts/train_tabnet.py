"""
TabNet Regressor for DemandForecast
===================================
Attentive tabular deep learning architecture with sequential attention mechanism (Sparsemax)
for zero-inflated continuous retail demand forecasting.
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

DATA_DIR   = "/Users/somyabhadada/nodex8/data/DemandForecast/samples/"
OUTPUT_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/Model_Outputs/TabNet/"

EXCLUDE_COLS = ["RowNum", "Date", "UPC", "COMP_STR_NO", "QUANTITY", "BRAND", "RET_LIR_ID"]
CATEGORICAL_COLS = ["PROMOTION_FLG", "C_PROMOTION_FLG"]

def load_dataset(fname):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].map({'Y': 1, 'N': 0, '1': 1, '0': 0, 1: 1, 0: 0}).fillna(0)
            
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feat_cols].fillna(0).values
    y = df["QUANTITY"].fillna(0).values.reshape(-1, 1)
    return X, y, feat_cols

def evaluate(model, X, y, label):
    pred = model.predict(X).flatten()
    pred = np.maximum(0, pred)  # Quantity cannot be negative
    y_true = y.flatten()
    
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
    print("Loading data for TabNet Regressor...")
    X_dev, y_dev, feat_cols = load_dataset("dev.csv")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.15, random_state=42
    )
    
    print(f"Features: {len(feat_cols)}, Train rows: {len(y_train)}, Val rows: {len(y_val)}")
    
    print("\nInitialising TabNetRegressor...")
    reg = TabNetRegressor(
        n_d=32,
        n_a=32,
        n_steps=5,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        momentum=0.02,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='sparsemax',
        verbose=10
    )
    
    print("Training TabNet Regressor...")
    reg.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=['val'],
        eval_metric=['rmse', 'mae'],
        max_epochs=40,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    
    # Save model
    reg.save_model(os.path.join(OUTPUT_DIR, "tabnet_model.zip"))
    with open(os.path.join(OUTPUT_DIR, "features.json"), "w") as f:
        json.dump(feat_cols, f, indent=4)
        
    EVAL_SETS = [
        ("dev.csv", "DEV"),
        ("oos.csv", "OOS"),
        ("oot_01_201212.csv", "OOT_201212"),
        ("oot_02_201301.csv", "OOT_201301"),
        ("oot_03_201302.csv", "OOT_201302")
    ]
    
    print("\nEvaluating across sets...")
    scorecards = []
    for fname, lbl in EVAL_SETS:
        if not os.path.exists(os.path.join(DATA_DIR, fname)):
            continue
        X_eval, y_eval, _ = load_dataset(fname)
        metrics = evaluate(reg, X_eval, y_eval, lbl)
        scorecards.append(metrics)
        print(f"  {lbl:12s}: MAE={metrics['MAE']}, RMSE={metrics['RMSE']}, R2={metrics['R2']}, Acc_20%={metrics['Accuracy_20pct']}")
        
    df_score = pd.DataFrame(scorecards)
    df_score.to_csv(os.path.join(OUTPUT_DIR, "performance_scorecard.csv"), index=False)
    
    # Feature Importance
    importances = reg.feature_importances_
    df_fi = pd.DataFrame({
        "Feature": feat_cols,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    df_fi.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    print("\nTop 10 features by TabNet attention:")
    print(df_fi.head(10).to_string(index=False))
    
    print(f"\n✅ DemandForecast TabNet complete! Outputs in {OUTPUT_DIR}")

if __name__ == "__main__":
    run()
