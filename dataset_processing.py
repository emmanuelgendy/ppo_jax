import pandas as pd

building_df = pd.read_csv("/home/emmanuel-gendy/Documents/EnergySim/src/energysim/core/data/heeten_building_df14.csv")
price_df = pd.read_csv("/home/emmanuel-gendy/Documents/EnergySim/src/energysim/core/data/ee_day_ahead_prices_2018_till_2020.csv")
merged_df = pd.merge(building_df, price_df[['unixtime', 'germany']], on='unixtime', how='inner')

# RENAME AND SCALE
merged_df = merged_df.rename(columns={
    'unixtime': 'timestamp',
    'germany': 'raw_price',
    'load': 'load',
    'pv': 'pv'
})

# SCALE PRICE: EUR/MWh -> EUR/kWh (Divide by 1000)
merged_df['price'] = merged_df['raw_price'] / 1000.0

merged_df.to_csv("heeten_training_master.csv", index=False)
print(f"✅ Price Corrected: Avg Price is now €{merged_df['price'].mean():.4f}/kWh")