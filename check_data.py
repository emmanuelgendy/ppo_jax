import pandas as pd
import numpy as np

def inspect_dataset(file_path):
    print(f"--- Inspecting {file_path} ---")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    # 1. Check length
    print(f"Total rows: {len(df):,}")
    
    # 2. Check for NaNs (The JAX Killer)
    nans = df.isna().sum()
    if nans.sum() > 0:
        print("\n🚨 WARNING: Missing Data (NaNs) Found! This will break JAX.")
        print(nans[nans > 0])
    else:
        print("✅ No missing data (NaNs) detected.")

    # 3. Check Temperature Extremes
    if 'T_ext' in df.columns: # Replace 'T_ext' with whatever your outdoor temp column is named
        min_temp = df['T_ext'].min()
        max_temp = df['T_ext'].max()
        print(f"\n🌡️ Outdoor Temp Range: {min_temp}°C to {max_temp}°C")
        if min_temp < 0:
            print("❄️ Warning: Sub-zero temperatures detected. A 2500W Heat Pump might not be enough!")
    
    # 4. Check Pricing
    if 'price' in df.columns: # Replace with your price column name
        print(f"💰 Price Range: {df['price'].min():.4f} to {df['price'].max():.4f}")

if __name__ == "__main__":
    inspect_dataset("heeten_training_master.csv")