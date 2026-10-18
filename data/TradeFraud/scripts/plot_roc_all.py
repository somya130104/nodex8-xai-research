import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

DATA_DIR = "/Users/somyabhadada/nodex8/data/TradeFraud/Modeling_Data_Enriched/"
MODEL_PATH = "/Users/somyabhadada/nodex8/data/TradeFraud/Model_Outputs/Model2_XGBoost/model2_xgboost.pkl"
LOCAL_OUT_IMG = "/Users/somyabhadada/nodex8/data/TradeFraud/Model_Outputs/Model2_XGBoost/roc_chart_model2.png"

EXCLUDE_COLS = ["Client", "Client_ID", "Date", "Dispute", "Segment", "Activeness_Segment"]

eval_sets = [
    ("train.csv", "Train (oversampled)"),
    ("val_oos.csv", "Validation OOS"),
    ("internal_oot.csv", "Internal OOT (Aug-Sep)"),
    ("oot_202510.csv", "OOT Oct 2025"),
    ("oot_202511.csv", "OOT Nov 2025"),
    ("oot_202512.csv", "OOT Dec 2025"),
]

print("Loading model...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

plt.figure(figsize=(10, 8))

for fname, label in eval_sets:
    print(f"Processing {fname}...")
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df["target"] = (df["Dispute"] == "D").astype(int)
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != "target"]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    X = df[feat_cols].fillna(0)
    y = df["target"]
    
    proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    auc = roc_auc_score(y, proba)
    
    plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Model 2 (XGBoost + Enriched Features)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

os.makedirs(os.path.dirname(LOCAL_OUT_IMG), exist_ok=True)
plt.savefig(LOCAL_OUT_IMG, dpi=300, bbox_inches="tight")
plt.close()
print("Saved ROC chart to:", LOCAL_OUT_IMG)
