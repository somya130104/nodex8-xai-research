import pickle
import pandas as pd
import numpy as np
import lime.lime_tabular
import os
import sys
sys.path.insert(0, '..')

# Load artifacts
with open('../Outputs/manifesto_config.pkl', 'rb') as f:
    manifesto = pickle.load(f)
dataset_configs = manifesto['dataset_configs']

with open('../Client_Input/Common/feature_engineering_artifacts.pkl', 'rb') as f:
    fe_artifacts = pickle.load(f)
feature_cols = fe_artifacts['feature_cols']

# Load test data
run = dataset_configs[0] # XGBoost Dev
df = pd.read_csv(run['path']).head(10)

X_train_bg = pd.read_csv('../Client_Input/Common/dev_xgboost.csv')[feature_cols].values
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_bg,
    feature_names=feature_cols,
    mode='regression',
    discretize_continuous=True
)

# Load a model
cust_id = df.iloc[0]['Cust_ID']
prefix = 'xgb'
p = os.path.join('../Client_Input/xgboost/models', f"{prefix}_{cust_id}.pkl")
with open(p, 'rb') as f: model = pickle.load(f)

def predict_fn(X_array):
    import xgboost as xgb
    X_df = pd.DataFrame(X_array, columns=feature_cols)
    return model.predict(xgb.DMatrix(X_df))

print("-" * 30)
print("Testing LIME on Instance 0")
exp = lime_explainer.explain_instance(df[feature_cols].values[0], predict_fn, num_features=len(feature_cols))

print("\nRaw LIME as_list():")
for t_feat, weight in exp.as_list():
    print(f"'{t_feat}': {weight}")
    
print("-" * 30)
print("Testing Case Sensitivity and Mapping:")
for t_feat, weight in exp.as_list():
    matched = False
    for orig_f in feature_cols:
        if orig_f in t_feat:
            print(f"MATCH: '{orig_f}' found in '{t_feat}'")
            matched = True
            break
    if not matched:
        print(f"FAIL: No match found for '{t_feat}'")
