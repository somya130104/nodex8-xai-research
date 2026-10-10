import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

def generate_consent_log():
    # 1. Load Customer Master
    master_path = 'customer_master.csv'
    if not os.path.exists(master_path):
        print(f"Error: {master_path} not found.")
        return

    print("Loading customer master...")
    df_master = pd.read_csv(master_path)
    
    # Configuration
    purposes = ['Marketing', 'Communication', 'Service', 'Analytics', 'ThirdPartySharing', 'Transaction']
    channels = ['Web', 'App', 'Email', 'Phone', 'SMS', 'Whatsapp', 'Unknown']
    audit_date = datetime(2026, 3, 31)
    withdrawal_rate = 0.15
    
    np.random.seed(42)
    random.seed(42)
    
    consent_data = []
    
    print("Generating consent records...")
    for idx, row in df_master.iterrows():
        # Probability that this customer is present in the consent log (e.g., 95%)
        if random.random() > 0.95:
            continue
            
        onboarding_date = pd.to_datetime(row['Customer_onboarding_Date'])
        
        # Determine which purposes the customer has interacted with
        num_purposes = random.randint(0, len(purposes))
        selected_purposes = random.sample(purposes, num_purposes)
        
        for purpose in selected_purposes:
            # Granted Date: Random date between onboarding and audit date
            days_range = (audit_date - onboarding_date).days
            if days_range <= 1:
                granted_days = 0
            else:
                granted_days = random.randint(0, days_range - 1)
            
            granted_date = onboarding_date + timedelta(days=granted_days)
            granted_channel = random.choice(channels)
            expiry_date = granted_date + timedelta(days=365)
            
            # Withdrawal logic
            withdrawn_date = None
            withdrawn_channel = None
            
            if random.random() < withdrawal_rate:
                wd_range = (audit_date - granted_date).days
                if wd_range > 1:
                    wd_days = random.randint(1, wd_range)
                    withdrawn_date = (granted_date + timedelta(days=wd_days)).date()
                    withdrawn_channel = random.choice(channels)
            
            consent_data.append({
                'Client_ID': row['Client_ID'],
                'Purpose': purpose,
                'GrantedDate': granted_date.date(),
                'GrantedChannel': granted_channel,
                'ExpiryDate': expiry_date.date(),
                'WithdrawnDate': withdrawn_date,
                'WithdrawnChannel': withdrawn_channel,
                'LastAuditDate': audit_date.strftime('%d%b%Y')
            })
            
    df_consent = pd.DataFrame(consent_data)
    
    # 5. Output
    output_path = 'consent_log.csv'
    df_consent.to_csv(output_path, index=False)
    print(f"Success! Created {output_path} with {len(df_consent)} records.")

if __name__ == "__main__":
    generate_consent_log()
