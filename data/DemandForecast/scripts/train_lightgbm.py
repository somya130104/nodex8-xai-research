"""
LightGBM model for DemandForecast
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import pickle
warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/samples/"
OUTPUT_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/Model_Outputs/LightGBM/"

EXCLUDE_COLS = ["RowNum", "Date", "UPC", "COMP_STR_NO", "QUANTITY", "BRAND", "RET_LIR_ID"]
CATEGORICAL_COLS = ["PROMOTION_FLG", "C_PROMOTION_FLG"]

def load_dataset(fname):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    
    # Process categorical
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feat_cols]
    y = df["QUANTITY"].fillna(0)
    return X, y, feat_cols

def evaluate(model, X, y, label):
    pred = model.predict(X)
    pred = np.maximum(0, pred)  # Quantity cannot be negative
    
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    
    mask = y > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y[mask] - pred[mask]) / y[mask]))
    else:
        mape = np.nan
        
    hits = np.abs(y - pred) <= (0.2 * np.abs(y))
    accuracy = hits.mean()
    
    return {
        "Dataset": label,
        "Rows": len(y),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4) if not np.isnan(mape) else None,
        "R2": round(r2, 4),
        "Accuracy_20pct": round(accuracy, 4)
    }

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading DEV data...")
    X_dev, y_dev, feat_cols = load_dataset("dev.csv")
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.15, random_state=42
    )
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    print("\nTraining LightGBM Regressor...")
    params = {
        'objective': 'tweedie',
        'tweedie_variance_power': 1.5,
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[valid_data],
        valid_names=['valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(50)]
    )
    print(f"\nBest iteration: {model.best_iteration}")
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, "model_lightgbm.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    # Evaluate across splits
    print("\nEvaluating across all datasets...")
    eval_sets = [
        ("dev.csv", "DEV"),
        ("oos.csv", "OOS"),
        ("oot_01_201212.csv", "OOT Dec 2012"),
        ("oot_02_201301.csv", "OOT Jan 2013"),
        ("oot_03_201302.csv", "OOT Feb 2013")
    ]
    
    results = []
    for fname, label in eval_sets:
        try:
            X, y, _ = load_dataset(fname)
            res = evaluate(model, X, y, label)
            results.append(res)
            print(f"  {label:15s} | RMSE={res['RMSE']:.4f} | MAE={res['MAE']:.4f} | "
                  f"MAPE={res['MAPE']:.4f} | R2={res['R2']:.4f}")
        except FileNotFoundError:
            print(f"  Skipping {fname} - not found")
            
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, "performance_scorecard.csv"), index=False)
    
    # Feature importance
    print("\nComputing feature importance...")
    fi = pd.DataFrame({
        "Feature": feat_cols,
        "Importance_Gain": model.feature_importance(importance_type='gain')
    }).sort_values("Importance_Gain", ascending=False).reset_index(drop=True)
    fi["Rank"] = fi.index + 1
    fi.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    
    print("\nTop 15 features by Gain:")
    print(fi.head(15).to_string(index=False))
    print(f"\n✅ LightGBM complete. Outputs saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    run()
