import pandas as pd
import re

# ─────────────────────────────────────────────────────────────────
# Rebuild variable_dictionary.csv with full dual-axis Rolling Window
# hierarchy: Nature-axis + Time-Horizon-axis
# ─────────────────────────────────────────────────────────────────

# Time horizon grouping for the 13 windows
def get_time_horizon(w):
    if w in [3, 7]:           return "Very Short-term (3–7 days)"
    elif w in [15, 21, 30]:   return "Short-term (15–30 days)"
    elif w in [45, 60, 75, 90]: return "Medium-term (45–90 days)"
    elif w in [120, 150, 180]:  return "Long-term (120–180 days)"
    elif w == 365:              return "Annual Baseline (365 days)"
    return "Unknown"

# Nature subgroup for each metric template
def get_nature_subgroup(template):
    if template in ["Trade_days_l_{w}d", "Trade_numbers_l_{w}d", "Avg_Trade_l_{w}d"]:
        return "Activity — Trading Frequency"
    elif template in ["Trade_bAmt_l_{w}d", "Trade_sAmt_l_{w}d",
                      "Trade_totalAmt_l_{w}d", "Avg_Trade_Amt_l_{w}d"]:
        return "Trade Value — Amount Metrics"
    elif template in ["BscriptAvg_l_{w}d", "SscriptAvg_l_{w}d"]:
        return "Script Category — Average Composition"
    elif template in ["minBscript_l_{w}d", "minSscript_l_{w}d"]:
        return "Script Category — Minimum Concentration"
    elif template in ["maxBscript_l_{w}d", "maxSscript_l_{w}d"]:
        return "Script Category — Maximum Concentration"
    return "Unknown"

windows = [3, 7, 15, 21, 30, 45, 60, 75, 90, 120, 150, 180, 365]
rows = []

# ── 1. Identity ────────────────────────────────────────────────
rows += [
    ("Client",    "1. Identity", "Client-Level Identifiers", "—", "Raw",     "object",  "N",
     "Original raw client username / ID from trading system",
     "Core trading system", "N/A"),
    ("Client_ID", "1. Identity", "Client-Level Identifiers", "—", "Engineered","object","N",
     "Unique alphanumeric client identifier (FP_CL_XXXXX)",
     "Sequential encoding of Client username", "Client"),
]

# ── 2. Client Profile — Tenure ────────────────────────────────
rows += [
    ("cash_membershipLength",    "2. Client Profile", "Membership & Tenure", "—", "Raw",    "int64",  "N",
     "Total days since account opening",
     "System date − account open date", "N/A"),
    ("cash_preDurationLength",   "2. Client Profile", "Membership & Tenure", "—", "Raw",    "int64",  "N",
     "Days active before the dormancy period",
     "Trading system history logs", "N/A"),
    ("cash_dormancyPeriod",      "2. Client Profile", "Membership & Tenure", "—", "Raw",    "int64",  "N",
     "Days with no trading activity (dormancy window)",
     "membershipLength − liveDays − preDurationLength", "cash_membershipLength, cash_liveDays, cash_preDurationLength"),
    ("cash_liveDays",            "2. Client Profile", "Membership & Tenure", "—", "Raw",    "int64",  "N",
     "Total calendar days with at least one trade executed",
     "Count of unique trade dates in full client history", "N/A"),
    ("cash_liveMonths",          "2. Client Profile", "Membership & Tenure", "—", "Derived","float64","N",
     "Active trading months",
     "cash_liveDays / 30", "cash_liveDays"),
    ("cash_daysTraded",          "2. Client Profile", "Trading Behaviour", "—", "Raw",    "int64",  "N",
     "Distinct days on which the client executed at least one trade",
     "Count of unique trading dates in full history", "N/A"),
]

# ── 3. Lifetime Aggregates ───────────────────────────────────
rows += [
    ("cash_Td_Bamt",               "3. Lifetime Aggregates", "Trade Amounts (Lifetime)", "—", "Raw",    "float64","N",
     "Total Buy Amount — all scripts, full client tenure",
     "Sum of all buy-side transaction values", "N/A"),
    ("cash_Td_Samt",               "3. Lifetime Aggregates", "Trade Amounts (Lifetime)", "—", "Raw",    "float64","N",
     "Total Sell Amount — all scripts, full client tenure",
     "Sum of all sell-side transaction values", "N/A"),
    ("cash_Td_totalAmt",           "3. Lifetime Aggregates", "Trade Amounts (Lifetime)", "—", "Derived","float64","N",
     "Total Gross Trade Amount (Buy + Sell) — full client tenure",
     "cash_Td_Bamt + cash_Td_Samt", "cash_Td_Bamt, cash_Td_Samt"),
    ("cash_AvgTradePerTradingDay", "3. Lifetime Aggregates", "Trade Frequency (Lifetime)","—","Derived","float64","N",
     "Average trades per active trading day over full client tenure",
     "Total trades / cash_daysTraded", "cash_daysTraded"),
    ("cash_AvgTradePerTradingMonth","3. Lifetime Aggregates","Trade Frequency (Lifetime)","—","Derived","float64","N",
     "Average trades per active trading month over full client tenure",
     "Total trades / cash_liveMonths", "cash_liveMonths"),
]

# ── 4. Rolling Window Features ──────────────────────────────
metric_meta = {
    "Trade_days_l_{w}d":     ("int64",   "Number of distinct days traded in the last {w} days",               "Count unique trade dates in window"),
    "Trade_numbers_l_{w}d":  ("int64",   "Total trade transactions in the last {w} days",                     "Sum of all trade records in window"),
    "Avg_Trade_l_{w}d":      ("float64", "Average trades per active day in the last {w} days",                "Trade_numbers / Trade_days in window"),
    "Trade_bAmt_l_{w}d":     ("float64", "Total Buy Amount — all scripts — in the last {w} days",             "Sum of buy-side values in window"),
    "Trade_sAmt_l_{w}d":     ("float64", "Total Sell Amount — all scripts — in the last {w} days",            "Sum of sell-side values in window"),
    "Trade_totalAmt_l_{w}d": ("float64", "Total Gross Trade Amount (Buy+Sell) in the last {w} days",          "Trade_bAmt + Trade_sAmt"),
    "Avg_Trade_Amt_l_{w}d":  ("float64", "Average value per transaction in the last {w} days",                "Trade_totalAmt / Trade_numbers"),
    "BscriptAvg_l_{w}d":     ("float64", "Average B-category (mid/small-cap) scripts bought per trade-day in last {w} days", "Mean daily buy-side B-script count"),
    "SscriptAvg_l_{w}d":     ("float64", "Average B-category scripts sold per trade-day in last {w} days",    "Mean daily sell-side B-script count"),
    "minBscript_l_{w}d":     ("int64",   "Minimum B-scripts bought on any single day in last {w} days",       "Min of daily buy B-script count"),
    "minSscript_l_{w}d":     ("int64",   "Minimum B-scripts sold on any single day in last {w} days",         "Min of daily sell B-script count"),
    "maxBscript_l_{w}d":     ("int64",   "Maximum B-scripts bought on any single day in last {w} days",       "Max of daily buy B-script count"),
    "maxSscript_l_{w}d":     ("int64",   "Maximum B-scripts sold on any single day in last {w} days",         "Max of daily sell B-script count"),
}

for w in windows:
    wstr = f"{w:03d}" if w < 100 else str(w)
    horizon = get_time_horizon(w)
    for template, (dtype, desc_tmpl, logic) in metric_meta.items():
        col_name = template.replace("{w}", wstr)
        desc = desc_tmpl.replace("{w}", str(w))
        nature = get_nature_subgroup(template)
        group_label = f"4. Rolling Window — {nature}"
        rows.append((col_name, group_label, nature, horizon, "Derived", dtype, "N",
                     desc, logic, f"Raw transaction records, {w}d lookback"))

# ── 5. Target ────────────────────────────────────────────────
rows += [
    ("Dispute", "5. Target Variable", "Model Target", "—", "Raw", "object", "Y",
     "Whether the cash trade transaction was disputed (D = Dispute, N = No Dispute)",
     "Renamed from Cash_Dispute-TargetVariable. Binary rare-event target.",
     "Cash_Dispute-TargetVariable"),
]

# ── 6. Metadata ──────────────────────────────────────────────
rows += [
    ("Date",              "6. Metadata", "Temporal",            "—", "Engineered","date",  "N",
     "Randomised transaction date (Apr 2024 – Dec 2025), balanced monthly",
     "Synthetically generated with uniform monthly distribution", "Generated"),
    ("Segment",           "6. Metadata", "Client Segmentation", "—", "Engineered","object","N",
     "Demographic/behavioural client segment (e.g. Sages, Anchors, Hustlers)",
     "Randomly assigned from Segment_Master.xlsx", "Segment_Master.xlsx"),
    ("Activeness_Segment","6. Metadata", "Client Segmentation", "—", "Derived",  "object","N",
     "6-tier trading activity level (Dormant / Occasional / Regular / Active / Frequent / Power Trader)",
     "pd.qcut on cash_daysTraded.rank(): 6 equal-frequency bins", "cash_daysTraded"),
]

# ── Build DataFrame ─────────────────────────────────────────
cols = [
    "Variable_Name", "Group", "Sub_Group", "Time_Horizon",
    "Variable_Type", "Data_Type", "Is_Target",
    "Description", "Calculation_Logic", "Source_Variables"
]
df_dict = pd.DataFrame(rows, columns=cols)

out_path = "/Users/somyabhadada/nodex8/data/TradeFraud/Data/variable_dictionary.csv"
df_dict.to_csv(out_path, index=False)

print(f"Saved {len(df_dict)} variables to {out_path}\n")
print("Group distribution:")
print(df_dict['Group'].value_counts().sort_index().to_string())
print("\nTime Horizon distribution (Rolling Window only):")
rw = df_dict[df_dict['Group'].str.startswith("4.")]
print(rw['Time_Horizon'].value_counts().to_string())
print("\nNature Sub-Group distribution (Rolling Window only):")
print(rw['Sub_Group'].value_counts().to_string())
