import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

def preprocess_raw_data():
    raw_data_path = '/Users/angshubh/FraudPrediction/cash_all_var.csv'
    output_data_path = '/Users/somyabhadada/nodex8/data/TradeFraud/Data/Data.csv'
    segment_master_path = '/Users/somyabhadada/nodex8/data/TradeFraud/Data/Segment_Master.xlsx'

    if not os.path.exists(raw_data_path):
        print(f"Error: {raw_data_path} not found.")
        return

    print("Loading raw data...")
    df = pd.read_csv(raw_data_path)

    # 1. Rename target variable
    if 'Cash_Dispute-TargetVariable' in df.columns:
        df.rename(columns={'Cash_Dispute-TargetVariable': 'Dispute'}, inplace=True)
        print("Renamed target to 'Dispute'")

    # 2. Map Client to Alphanumeric Client_ID
    print("Mapping Client to Client_ID...")
    unique_clients = df['Client'].unique()
    client_mapping = {client: f"FP_CL_{i+1:05d}" for i, client in enumerate(unique_clients)}
    df['Client_ID'] = df['Client'].map(client_mapping)

    # Move Client_ID next to Client
    cols = list(df.columns)
    client_idx = cols.index('Client')
    cols.insert(client_idx + 1, cols.pop(cols.index('Client_ID')))
    df = df[cols]

    # 3. Generate balanced monthly datestamps (01 Apr 2024 - 31 Dec 2025)
    print("Generating balanced monthly dates...")
    num_rows = len(df)
    
    # Generate month starts
    months = pd.date_range(start='2024-04-01', end='2025-12-01', freq='MS')
    num_months = len(months)
    
    rows_per_month = num_rows // num_months
    remainder = num_rows % num_months
    
    dates = []
    for i, month_start in enumerate(months):
        n = rows_per_month + (1 if i < remainder else 0)
        # Get number of days in month
        month_end_day = (month_start + pd.offsets.MonthEnd(0)).day
        for _ in range(n):
            random_day = random.randint(1, month_end_day)
            dates.append(month_start.replace(day=random_day))
            
    # Shuffle dates to avoid temporal bias in original data order
    random.shuffle(dates)
    df['Date'] = dates
    print(f"Generated dates for {num_months} months.")

    # 4. Assign random Segments
    print("Assigning random segments...")
    if os.path.exists(segment_master_path):
        df_segments = pd.read_excel(segment_master_path)
        # Look for 'Nickname' or 'Segment' column
        if 'Nickname' in df_segments.columns:
            segment_list = df_segments['Nickname'].dropna().unique().tolist()
        elif 'Segment' in df_segments.columns:
            segment_list = df_segments['Segment'].dropna().unique().tolist()
        else:
            segment_list = ['Segment_A', 'Segment_B', 'Segment_C']
        
        df['Segment'] = [random.choice(segment_list) for _ in range(num_rows)]
    else:
        print("Warning: Segment_Master.xlsx not found. Assigning dummy segments.")
        df['Segment'] = 'General'

    # 5. Derive Activeness_Segment (6 Levels)
    print("Deriving 6-level Activeness_Segment...")
    if 'cash_daysTraded' in df.columns:
        # First, ensure it's numeric
        df['cash_daysTraded'] = pd.to_numeric(df['cash_daysTraded'], errors='coerce').fillna(0)
        
        # Define 6 levels based on quantiles
        # We use rank method='first' to ensure we get unique split even if many values are the same
        labels = ['Dormant', 'Occasional', 'Regular', 'Active', 'Frequent', 'Power Trader']
        try:
            df['Activeness_Segment'] = pd.qcut(df['cash_daysTraded'].rank(method='first'), q=6, labels=labels)
        except Exception as e:
            print(f"Error in qcut: {e}. Falling back to uniform split.")
            df['Activeness_Segment'] = 'Unknown'
            
        # Log thresholds
        for i in range(1, 6):
            q_val = df['cash_daysTraded'].quantile(i/6)
            print(f"Quantile {i/6*100:.1f}% threshold: {q_val}")
    else:
        print("Warning: 'cash_daysTraded' not found. Assigning 'Unknown' activeness.")
        df['Activeness_Segment'] = 'Unknown'

    # 6. Save output
    print(f"Saving preprocessed data to {output_data_path}...")
    df.to_csv(output_data_path, index=False)
    print("Done!")

if __name__ == "__main__":
    preprocess_raw_data()
