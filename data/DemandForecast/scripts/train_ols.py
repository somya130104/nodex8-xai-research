"""
Linear Regression (OLS) model for DemandForecast
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import pickle
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/samples/"
OUTPUT_DIR = "/Users/somyabhadada/nodex8/data/DemandForecast/Model_Outputs/LinearRegression/"

EXCLUDE_COLS = ["RowNum", "Date", "UPC", "COMP_STR_NO", "QUANTITY", "BRAND", "RET_LIR_ID"]
CATEGORICAL_COLS = ["PROMOTION_FLG", "C_PROMOTION_FLG"]

def load_dataset(fname):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    
    # Process categorical (one-hot encode for linear model)
    df = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns], drop_first=True)
    
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    # Fill NaN values which OLS cannot handle
    X = df[feat_cols].fillna(0)
    y = df["QUANTITY"].fillna(0)
    return X, y, feat_cols

def evaluate(model, scaler, X, y, label, expected_features):
    # Ensure X has all expected features (dummy columns might be missing)
    for col in expected_features:
        if col not in X.columns:
            X[col] = 0
    X = X[expected_features]
    
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    pred = np.maximum(0, pred)  # Quantity cannot be negative
    
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    
    # MAPE only where y > 0
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
    X_train, y_train, feat_cols = load_dataset("dev.csv")
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("\nTraining Linear Regression (OLS)...")
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    model_path = os.path.join(OUTPUT_DIR, "model_ols.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({'model': model, 'scaler': scaler, 'features': feat_cols}, f)
        
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
            res = evaluate(model, scaler, X, y, label, feat_cols)
            results.append(res)
            print(f"  {label:15s} | RMSE={res['RMSE']:.4f} | MAE={res['MAE']:.4f} | "
                  f"MAPE={res['MAPE']:.4f} | R2={res['R2']:.4f}")
        except FileNotFoundError:
            print(f"  Skipping {fname} - not found")
            
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, "performance_scorecard.csv"), index=False)
    
    # Feature importance
    print("\nComputing feature importance (absolute coefficients)...")
    fi = pd.DataFrame({
        "Feature": feat_cols,
        "Coefficient": model.coef_,
        "Abs_Coefficient": np.abs(model.coef_)
    }).sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)
    fi["Rank"] = fi.index + 1
    fi.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    
    print("\nTop 15 features by Absolute Coefficient:")
    print(fi[["Rank", "Feature", "Coefficient", "Abs_Coefficient"]].head(15).to_string(index=False))
    print(f"\n✅ Linear Regression complete. Outputs saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    run()
