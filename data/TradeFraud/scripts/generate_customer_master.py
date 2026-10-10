import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

def generate_customer_master():
    # 1. Load Data
    data_path = 'Part_A/Data/Data.csv'
    segment_master_path = 'Part_A/Data/Segment_Master.xlsx'
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    if not os.path.exists(segment_master_path):
        print(f"Error: {segment_master_path} not found.")
        return

    print("Loading data...")
    df_data = pd.read_csv(data_path)
    df_segments = pd.read_excel(segment_master_path)
    
    # 2. Extract unique customers and theories first transaction date
    print("Extracting unique clients...")
    df_data['Date'] = pd.to_datetime(df_data['Date'])
    
    unique_custs = df_data.groupby('Client_ID').agg({
        'Segment': 'first',
        'Activeness_Segment': 'first',
        'Date': 'min'
    }).reset_index()
    unique_custs.rename(columns={'Date': 'First_Transaction_Date'}, inplace=True)
    
    # 3. Merge with segment metadata
    print("Merging with segment master...")
    df_master = pd.merge(unique_custs, df_segments, left_on='Segment', right_on='Nickname', how='left')
    
    # 4. Generate random but logical data
    print("Generating PII data...")
    
    first_names = ["John", "Jane", "Robert", "Emily", "Michael", "Sarah", "David", "Jessica", "William", "Ashley", 
                   "Joseph", "Amanda", "James", "Melissa", "Charles", "Stephanie", "Christopher", "Nicole", "Daniel", "Elizabeth"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", 
                  "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]
    genders = ["Male", "Female", "Other"]
    streets = ["Main St", "Oak Ave", "Maple Dr", "Park Blvd", "Pine Ln", "Washington St", "Lake Ave", "Broadway", "Highland Rd", "Central Hwy"]
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]
    states = ["NY", "CA", "IL", "TX", "AZ", "PA", "TX", "CA", "TX", "CA"]

    np.random.seed(42)
    random.seed(42)

    data = []
    for idx, row in df_master.iterrows():
        f_name = random.choice(first_names)
        l_name = random.choice(last_names)
        full_name = f"{f_name} {l_name}"
        email = f"{f_name.lower()}.{l_name.lower()}{random.randint(10,99)}@example.com"
        national_id = f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"
        phone = f"({random.randint(200,999)}) {random.randint(100,999)}-{random.randint(1000,9999)}"
        address = f"{random.randint(1,9999)} {random.choice(streets)}, {random.choice(cities)}, {random.choice(states)}"
        zip_code = f"{random.randint(10000, 99999)}"
        
        # Date of Birth (Age 18-80)
        days_ago = random.randint(18*365, 80*365)
        dob = (datetime.now() - timedelta(days=days_ago)).date()
        
        gender = random.choice(genders)
        
        # Onboarding Date (30-1000 days before first transaction)
        days_before = random.randint(30, 1000)
        onboarding_date = (row['First_Transaction_Date'] - timedelta(days=days_before)).date()
        
        data.append({
            'Client_ID': row['Client_ID'],
            'Name': full_name,
            'Email': email,
            'National_ID': national_id,
            'PhoneNo': phone,
            'Address': address,
            'ZipCode': zip_code,
            'Date_Of_Birth': dob,
            'Gender': gender,
            'Customer_onboarding_Date': onboarding_date,
            'First_Transaction_Date': row['First_Transaction_Date'].date()
        })
        
    df_pii = pd.DataFrame(data)
    
    # Final merge
    df_final = pd.merge(df_master, df_pii, on='Client_ID')
    
    # Drop Nickname as it's redundant with Segment
    if 'Nickname' in df_final.columns:
        df_final.drop(columns=['Nickname'], inplace=True)
        
    # Reorder columns to put PII and Onboarding first
    cols = ['Client_ID', 'Name', 'Email', 'National_ID', 'PhoneNo', 'Address', 'ZipCode', 
            'Date_Of_Birth', 'Gender', 'Customer_onboarding_Date', 'First_Transaction_Date',
            'Segment', 'Activeness_Segment', 'Status', 'Segment Name', 'Demographic Profile', 
            'Socio-Economic Definition', 'Behavioral Traits', 'Composition Details (RAI Testing Context)']
    
    # Keep only columns that exist
    df_final = df_final[[c for c in cols if c in df_final.columns]]

    # 5. Output
    output_path = 'customer_master.csv'
    df_final.to_csv(output_path, index=False)
    print(f"Success! Created {output_path} with {len(df_final)} records.")

if __name__ == "__main__":
    generate_customer_master()
