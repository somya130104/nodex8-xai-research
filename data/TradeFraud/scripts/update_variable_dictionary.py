"""
Update variable_dictionary.csv — append Group 7: Engineered Features (91 vars)
"""
import pandas as pd

WINDOWS = [3, 7, 15, 21, 30, 45, 60, 75, 90, 120, 150, 180, 365]

def wstr(w):
    return f"{w:03d}" if w < 100 else str(w)

def get_time_horizon(w):
    if w in [3, 7]:              return "Very Short-term (3–7 days)"
    elif w in [15, 21, 30]:      return "Short-term (15–30 days)"
    elif w in [45, 60, 75, 90]:  return "Medium-term (45–90 days)"
    elif w in [120, 150, 180]:   return "Long-term (120–180 days)"
    elif w == 365:               return "Annual Baseline (365 days)"

rows = []

# ── Group A: Buy/Sell Imbalance ──────────────────────────────────────
for w in WINDOWS:
    ws = wstr(w)
    horizon = get_time_horizon(w)
    rows += [
        (f"BS_ratio_l_{ws}d",
         "7. Engineered — Buy/Sell Imbalance", "7A — Buy/Sell Ratio", horizon,
         "Derived", "float64", "N",
         f"Buy-to-Sell amount ratio in last {w} days. >1 = net buyer, <1 = net seller. Extreme values flag asymmetric activity.",
         f"Trade_bAmt_l_{ws}d / (Trade_sAmt_l_{ws}d + 1)",
         f"Trade_bAmt_l_{ws}d, Trade_sAmt_l_{ws}d"),

        (f"net_trade_amt_l_{ws}d",
         "7. Engineered — Buy/Sell Imbalance", "7A — Net Trade Amount", horizon,
         "Derived", "float64", "N",
         f"Net signed position (Buy − Sell) in last {w} days. Positive = net buyer, negative = net seller.",
         f"Trade_bAmt_l_{ws}d − Trade_sAmt_l_{ws}d",
         f"Trade_bAmt_l_{ws}d, Trade_sAmt_l_{ws}d"),
    ]

# ── Group B: Behavioural Shift ───────────────────────────────────────
for w_short, w_long in [(3, 30), (7, 90), (30, 180), (90, 365)]:
    ss, sl = wstr(w_short), wstr(w_long)
    rows += [
        (f"activity_shift_{w_short}v{w_long}d",
         "7. Engineered — Behavioural Shift", "7B — Activity Acceleration", "—",
         "Derived", "float64", "N",
         f"Ratio of recent ({w_short}d) to longer-term ({w_long}d) trade count. Value >>1 signals sudden burst in activity.",
         f"Trade_numbers_l_{ss}d / (Trade_numbers_l_{sl}d + 1)",
         f"Trade_numbers_l_{ss}d, Trade_numbers_l_{sl}d"),

        (f"amount_shift_{w_short}v{w_long}d",
         "7. Engineered — Behavioural Shift", "7B — Amount Acceleration", "—",
         "Derived", "float64", "N",
         f"Ratio of recent ({w_short}d) to longer-term ({w_long}d) trade value. Value >>1 flags abnormal value surge.",
         f"Trade_totalAmt_l_{ss}d / (Trade_totalAmt_l_{sl}d + 1)",
         f"Trade_totalAmt_l_{ss}d, Trade_totalAmt_l_{sl}d"),
    ]

# ── Group C: Script Concentration Volatility ─────────────────────────
for w in WINDOWS:
    ws = wstr(w)
    horizon = get_time_horizon(w)
    rows += [
        (f"bscript_range_l_{ws}d",
         "7. Engineered — Script Concentration", "7C — Intra-Window Range", horizon,
         "Derived", "float64", "N",
         f"Range (max − min) of daily B-script buy count in last {w} days. High range = erratic/spiky B-script activity.",
         f"maxBscript_l_{ws}d − minBscript_l_{ws}d",
         f"maxBscript_l_{ws}d, minBscript_l_{ws}d"),

        (f"bscript_bs_ratio_l_{ws}d",
         "7. Engineered — Script Concentration", "7C — Buy/Sell Script Ratio", horizon,
         "Derived", "float64", "N",
         f"Ratio of avg B-scripts bought vs sold per day in last {w} days. Asymmetry flags one-sided B-script exposure.",
         f"BscriptAvg_l_{ws}d / (SscriptAvg_l_{ws}d + 1)",
         f"BscriptAvg_l_{ws}d, SscriptAvg_l_{ws}d"),
    ]

# ── Group D: Dormancy & Recency Engagement ───────────────────────────
rows += [
    ("dormancy_ratio",
     "7. Engineered — Dormancy & Recency", "7D — Lifecycle Profile", "—",
     "Derived", "float64", "N",
     "Proportion of total account life spent dormant. High value = rarely active account, sudden reactivation risk.",
     "cash_dormancyPeriod / (cash_membershipLength + 1)",
     "cash_dormancyPeriod, cash_membershipLength"),

    ("trade_intensity",
     "7. Engineered — Dormancy & Recency", "7D — Lifecycle Profile", "—",
     "Derived", "float64", "N",
     "Fraction of live days on which the client actually traded. Measures engagement density.",
     "cash_daysTraded / (cash_liveDays + 1)",
     "cash_daysTraded, cash_liveDays"),

    ("recent_activity_ratio_30d",
     "7. Engineered — Dormancy & Recency", "7D — Recency Ratio", "Short-term (15–30 days)",
     "Derived", "float64", "N",
     "Proportion of all-time trading days concentrated in the last 30 days. High = sudden spike after long inactivity.",
     "Trade_days_l_030d / (cash_daysTraded + 1)",
     "Trade_days_l_030d, cash_daysTraded"),

    ("recent_amount_ratio_30d",
     "7. Engineered — Dormancy & Recency", "7D — Recency Ratio", "Short-term (15–30 days)",
     "Derived", "float64", "N",
     "Proportion of lifetime total trade value executed in the last 30 days. >0.5 = half of all-time value in one month.",
     "Trade_totalAmt_l_030d / (cash_Td_totalAmt + 1)",
     "Trade_totalAmt_l_030d, cash_Td_totalAmt"),

    ("recent_amount_ratio_90d",
     "7. Engineered — Dormancy & Recency", "7D — Recency Ratio", "Medium-term (45–90 days)",
     "Derived", "float64", "N",
     "Proportion of lifetime total trade value executed in the last 90 days. Captures slower value concentration buildup.",
     "Trade_totalAmt_l_090d / (cash_Td_totalAmt + 1)",
     "Trade_totalAmt_l_090d, cash_Td_totalAmt"),
]

# ── Group E: Per-Trade Size Consistency ──────────────────────────────
for w in WINDOWS:
    ws = wstr(w)
    horizon = get_time_horizon(w)
    rows += [
        (f"avg_buy_size_l_{ws}d",
         "7. Engineered — Trade Size", "7E — Average Buy Size per Trade", horizon,
         "Derived", "float64", "N",
         f"Average buy-side transaction value in last {w} days. Spike vs history flags unusually large single trades.",
         f"Trade_bAmt_l_{ws}d / (Trade_numbers_l_{ws}d + 1)",
         f"Trade_bAmt_l_{ws}d, Trade_numbers_l_{ws}d"),

        (f"avg_sell_size_l_{ws}d",
         "7. Engineered — Trade Size", "7E — Average Sell Size per Trade", horizon,
         "Derived", "float64", "N",
         f"Average sell-side transaction value in last {w} days. Captures abnormal large sell events.",
         f"Trade_sAmt_l_{ws}d / (Trade_numbers_l_{ws}d + 1)",
         f"Trade_sAmt_l_{ws}d, Trade_numbers_l_{ws}d"),
    ]

# ── Append to existing dictionary ────────────────────────────────────
base_path = "/Users/somyabhadada/nodex8/data/TradeFraud/Data/variable_dictionary.csv"
df_base = pd.read_csv(base_path)

cols = ["Variable_Name", "Group", "Sub_Group", "Time_Horizon",
        "Variable_Type", "Data_Type", "Is_Target",
        "Description", "Calculation_Logic", "Source_Variables"]

df_new = pd.DataFrame(rows, columns=cols)
df_full = pd.concat([df_base, df_new], ignore_index=True)
df_full.to_csv(base_path, index=False)

print(f"Original vars: {len(df_base)}")
print(f"New engineered vars: {len(df_new)}")
print(f"Total in dictionary: {len(df_full)}")
print("\nGroup 7 breakdown:")
print(df_new["Group"].value_counts().to_string())
print("\nSub-Group breakdown:")
print(df_new["Sub_Group"].value_counts().to_string())
