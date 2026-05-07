import pandas as pd

# 1. Load the Building Data (Load and PV)
building_df = pd.read_csv("/home/emmanuel-gendy/Documents/EnergySim/src/energysim/core/data/heeten_building_df14.csv")

# 2. Load the Price Data
price_df = pd.read_csv("/home/emmanuel-gendy/Documents/EnergySim/src/energysim/core/data/ee_day_ahead_prices_2018_till_2020.csv")

# 3. Merge them on the 'unixtime' column
# This ensures every load value is paired with the correct price
merged_df = pd.merge(building_df, price_df[['unixtime', 'germany']], on='unixtime', how='inner')

# 4. Rename columns to match what ExoKey (and SimulationDataset) expects
# Based on your error log, it specifically wants 'timestamp' and 'price'
merged_df = merged_df.rename(columns={
    'unixtime': 'timestamp',
    'germany': 'price',
    'load': 'load',  # Ensure this matches your ExoKey.LOAD
    'pv': 'pv'
})

# 5. Handle missing weather columns
# Your SimulationDataset defaults these to 0.0 if missing, 
# but it's cleaner to have them in the CSV.
merged_df['ambient_temp'] = 15.0  # Optional: placeholder for now
merged_df['solar_irradiance'] = merged_df['pv'] * 1000.0 # Heuristic scaling

# 6. Save the unified training set
merged_df.to_csv("heeten_training_master.csv", index=False)
print("✅ Created heeten_training_master.csv with", len(merged_df), "steps.")