import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

DATA_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/Modeling_Data_Enriched/"
MODEL_PATH = "/Users/somyabhadada/nodex8/data/TradeFraud/Model_Outputs/Model2_XGBoost/model2_xgboost.pkl"
LOCAL_OUT_IMG = "/Users/somyabhadada/nodex8/data/TradeFraud/Model_Outputs/Model2_XGBoost/hosmer_lemeshow_chart.png"
CSV_OUT = "/Users/somyabhadada/nodex8/data/TradeFraud/Model_Outputs/Model2_XGBoost/hosmer_lemeshow_data.csv"

EXCLUDE_COLS = ["Client", "Client_ID", "Date", "Dispute", "Segment", "Activeness_Segment"]

eval_sets = [
    ("train.csv", "Train (oversampled)"),
    ("val_oos.csv", "Validation OOS"),
    ("internal_oot.csv", "Internal OOT (Aug-Sep)"),
    ("oot_202510.csv", "OOT Oct 2025"),
    ("oot_202511.csv", "OOT Nov 2025"),
    ("oot_202512.csv", "OOT Dec 2025"),
]

def hosmer_lemeshow(y_true, y_prob, bins=10):
    df = pd.DataFrame({'y': y_true, 'p': y_prob})
    
    # Sort by probability
    df = df.sort_values(by='p').reset_index(drop=True)
    
    # Use qcut with duplicates='drop' in case there are many tied probabilities
    try:
        df['decile'] = pd.qcut(df['p'], q=bins, duplicates='drop')
    except Exception as e:
        df['decile'] = pd.cut(df['p'], bins=bins)
        
    hl_df = df.groupby('decile', observed=False).agg(
        Total=('y', 'count'),
        Obs_Event=('y', 'sum'),
        Exp_Event=('p', 'sum')
    ).reset_index()
    
    # Expected non-events
    hl_df['Obs_NonEvent'] = hl_df['Total'] - hl_df['Obs_Event']
    hl_df['Exp_NonEvent'] = hl_df['Total'] - hl_df['Exp_Event']
    
    # To avoid divide by zero, add a tiny epsilon
    eps = 1e-10
    
    # HL Statistic
    hl_stat = np.sum(
        ((hl_df['Obs_Event'] - hl_df['Exp_Event'])**2 / (hl_df['Exp_Event'] + eps)) +
        ((hl_df['Obs_NonEvent'] - hl_df['Exp_NonEvent'])**2 / (hl_df['Exp_NonEvent'] + eps))
    )
    
    # degrees of freedom = bins - 2 if bins >= 3 else 1
    dof = max(1, len(hl_df) - 2)
    p_value = 1 - chi2.cdf(hl_stat, dof)
    
    # Convert interval to string
    hl_df['decile'] = hl_df['decile'].astype(str)
    hl_df['Decile_Rank'] = range(1, len(hl_df) + 1)
    
    return hl_stat, p_value, hl_df

print("Loading XGBoost model...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

all_hl_data = []
results_summary = []

for i, (fname, label) in enumerate(eval_sets):
    print(f"Processing {fname}...")
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df["target"] = (df["Dispute"] == "D").astype(int)
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != "target"]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    X = df[feat_cols].fillna(0)
    y = df["target"]
    
    proba = model.predict_proba(X)[:, 1]
    
    hl_stat, p_value, hl_df = hosmer_lemeshow(y, proba)
    
    # Save data
    hl_df['Dataset'] = label
    # Rearrange cols
    hl_df = hl_df[['Dataset', 'Decile_Rank', 'decile', 'Total', 'Obs_Event', 'Exp_Event', 'Obs_NonEvent', 'Exp_NonEvent']]
    all_hl_data.append(hl_df)
    results_summary.append({
        'Dataset': label,
        'HL_Statistic': hl_stat,
        'P-Value': p_value,
        'Bins_Used': len(hl_df)
    })
    
    # Plot Expected vs Observed
    ax = axes[i]
    x = hl_df['Decile_Rank']
    width = 0.35
    
    ax.bar(x - width/2, hl_df['Exp_Event'], width, label='Expected (Predicted)', color='#3498db')
    ax.bar(x + width/2, hl_df['Obs_Event'], width, label='Observed (Actual)', color='#e74c3c')
    
    ax.set_title(f"{label}\nHL Stat: {hl_stat:.2f} (p={p_value:.4f})")
    ax.set_xlabel("Decile (Low to High Risk probabilities)")
    ax.set_ylabel("Number of Disputes")
    ax.set_xticks(x)
    if i == 0:
        ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
os.makedirs(os.path.dirname(LOCAL_OUT_IMG), exist_ok=True)
plt.savefig(LOCAL_OUT_IMG, dpi=300, bbox_inches="tight")
plt.close()

final_df = pd.concat(all_hl_data, ignore_index=True)
final_df.to_csv(CSV_OUT, index=False)

summary_df = pd.DataFrame(results_summary)
print("\nHosmer-Lemeshow Results Summary:")
print(summary_df)

print("\nHosmer-Lemeshow OOS Calibration Data Preview:")
print(final_df[final_df['Dataset'] == 'Validation OOS'][['Decile_Rank', 'decile', 'Obs_Event', 'Exp_Event']])
print(f"Saved data to:   {CSV_OUT}")
