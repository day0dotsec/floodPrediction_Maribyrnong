import pandas as pd

# Load your dataset
df = pd.read_csv("Maribyrnong 2015-01-01 to 2025-06-01_VisualCrossing.csv")

# Drop 'severerisk' (mostly missing)
df.drop(columns=['severerisk'], inplace=True)

# Fill missing 'preciptype' with 'none'
df['preciptype'] = df['preciptype'].fillna('none')

# Fill missing 'visibility' with median
df['visibility'] = df['visibility'].fillna(df['visibility'].median())

# Drop unused columns
columns_to_drop = ['name', 'description', 'icon', 'stations', 'sunrise', 'sunset', 'snow', 'snowdepth']
df.drop(columns=columns_to_drop, inplace=True)

# Convert datetime and extract features
df['datetime'] = pd.to_datetime(df['datetime'])
df['month'] = df['datetime'].dt.month
df['day_of_week'] = df['datetime'].dt.dayofweek
df['year'] = df['datetime'].dt.year

# Create a flood_risk label (based on thresholds)
df['flood_risk'] = (
    (df['precip'] > 20) &
    (df['humidity'] > 85) &
    (df['visibility'] < 8)
).astype(int)

# Save the cleaned and engineered dataset
df.to_csv("Maribyrnong_Cleaned_Weather_FloodRisk.csv", index=False)

print("✅ Cleaned dataset saved as: Maribyrnong_Cleaned_Weather_FloodRisk.csv")
