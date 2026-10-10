import pickle
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, '..')

with open('../Client_Input/Common/feature_engineering_artifacts.pkl', 'rb') as f:
    fe_artifacts = pickle.load(f)
feature_cols = fe_artifacts['feature_cols']

# Load test data
with open('../Outputs/manifesto_config.pkl', 'rb') as f:
    manifesto = pickle.load(f)
run = manifesto['dataset_configs'][0]
df = pd.read_csv(run['path']).head(2)

# Load a model
cust_id = df.iloc[0]['Cust_ID']
prefix = 'xgb'
p = os.path.join('../Client_Input/xgboost/models', f"{prefix}_{cust_id}.pkl")
with open(p, 'rb') as f: model = pickle.load(f)

print(f"Testing Model Variance for Cust_ID {cust_id}...")
import xgboost as xgb

# Perturb a row slightly
row = df[feature_cols].iloc[0].copy()
row_perturbed = row.copy()
row_perturbed['MIN_PRICE'] *= 0.5 

X_test = pd.DataFrame([row, row_perturbed])
preds = model.predict(xgb.DMatrix(X_test))

print(f"Predictions: {preds}")
if preds[0] == preds[1]:
    print("WARNING: Model returned identical predictions for significantly different inputs!")
else:
    print("SUCCESS: Model returned different predictions.")

# Check for NaNs
print(f"NaNs in inputs: {X_test.isnull().sum().sum()}")
