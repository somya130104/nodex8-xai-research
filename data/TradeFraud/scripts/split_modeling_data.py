import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def oversample_dataset(df, target_col='Dispute', minor_class='D', factor=10):
    """Oversamples the minority class by a fixed factor."""
    minor_df = df[df[target_col] == minor_class]
    if len(minor_df) == 0:
        return df
    
    # We already have 1 copy in the original df, so we add (factor - 1) copies
    additional_copies = pd.concat([minor_df] * (factor - 1), ignore_index=True)
    combined_df = pd.concat([df, additional_copies], ignore_index=True)
    
    # Shuffle to mix the duplicates
    return combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

def split_modeling_data():
    input_path = '/Users/somyabhadada/nodex8/data/TradeFraud/Data/Data.csv'
    output_dir = '/Users/somyabhadada/nodex8/data/TradeFraud/Modeling_Data/'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print("Loading data...")
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Define time-based splits
    cutoff_date = pd.to_datetime('2025-09-30')
    
    # 1. External OOT Sets (Oct, Nov, Dec 2025)
    print("Extracting monthly OOT sets...")
    oot_oct = df[(df['Date'] >= '2025-10-01') & (df['Date'] <= '2025-10-31')]
    oot_nov = df[(df['Date'] >= '2025-11-01') & (df['Date'] <= '2025-11-30')]
    oot_dec = df[(df['Date'] >= '2025-12-01') & (df['Date'] <= '2025-12-31')]
    
    # 2. Dev pool (<= 30 Sep 2025)
    dev_pool = df[df['Date'] <= cutoff_date]
    
    # Internal OOT (Aug - Sep 2025)
    internal_oot = dev_pool[dev_pool['Date'] >= '2025-08-01']
    
    # Pool for Train/OOS (Apr 2024 - Jul 2025)
    train_val_pool = dev_pool[dev_pool['Date'] < '2025-08-01']
    
    # Stratified Split (80/20)
    print("Performing stratified split for Train and OOS...")
    train_df, val_oos_df = train_test_split(
        train_val_pool, 
        test_size=0.2, 
        stratify=train_val_pool['Dispute'], 
        random_state=42
    )
    
    # 3. Oversampling (10x for all)
    print("Applying 10x oversampling to all datasets...")
    datasets = {
        'train.csv': train_df,
        'val_oos.csv': val_oos_df,
        'internal_oot.csv': internal_oot,
        'oot_202510.csv': oot_oct,
        'oot_202511.csv': oot_nov,
        'oot_202512.csv': oot_dec
    }
    
    summary = []
    for name, data in datasets.items():
        if len(data) == 0:
            print(f"Warning: Dataset {name} is empty. Skipping.")
            continue
            
        # Perform oversampling
        oversampled_data = oversample_dataset(data, factor=10)
        
        save_path = os.path.join(output_dir, name)
        oversampled_data.to_csv(save_path, index=False)
        
        # Stats
        orig_count = len(data)
        new_count = len(oversampled_data)
        orig_d = (data['Dispute'] == 'D').sum()
        new_d = (oversampled_data['Dispute'] == 'D').sum()
        
        summary.append({
            'File': name,
            'Original Rows': orig_count,
            'New Rows': new_count,
            'Original Disputes': orig_d,
            'New Disputes': new_d,
            'Min Date': data['Date'].min().strftime('%Y-%m-%d'),
            'Max Date': data['Date'].max().strftime('%Y-%m-%d')
        })
        
    print("\nPartitioning Summary:")
    summary_df = pd.DataFrame(summary)
    print(summary_df)
    
    summary_df.to_csv(os.path.join(output_dir, 'split_summary.csv'), index=False)
    print(f"\nAll datasets saved to {output_dir}")

if __name__ == "__main__":
    split_modeling_data()
