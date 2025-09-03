import pandas as pd
import numpy as np

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

def create_risk_severity_labels(df):
    """
    Create 3-class risk severity labels based on weather patterns.
    
    Risk levels:
    1 = Low Risk (minimal flood conditions)
    2 = Medium Risk (elevated conditions, moderate concern)
    3 = High Risk (conditions similar to historical flood events)
    
    Based on analysis of historical flood events:
    - Precipitation: 26.2-34.6mm (mean: 30.5mm)
    - Humidity: 95.2-98.9% (mean: 97.6%)
    - Pressure: 1005.1-1007.2hPa (mean: 1006.2hPa)
    - Wind Speed: 13.4-37.1km/h (mean: 26.5km/h)
    """
    
    # Initialize all as low risk
    risk_score = np.zeros(len(df))
    
    # Define scoring weights for each factor
    
    # Precipitation scoring (highest weight - most critical factor)
    # High risk: >25mm (approaching flood levels)
    # Medium risk: 10-25mm (elevated precipitation)
    risk_score += np.where(df['precip'] > 25, 3.0,
                  np.where(df['precip'] > 10, 1.5, 0))
    
    # Humidity scoring (second most important)
    # High risk: >95% (flood event range)
    # Medium risk: >85% (elevated humidity)
    risk_score += np.where(df['humidity'] > 95, 2.5,
                  np.where(df['humidity'] > 85, 1.0, 0))
    
    # Pressure scoring (inverted - low pressure = higher risk)
    # High risk: <1008hPa (approaching flood event range)
    # Medium risk: <1012hPa (below average pressure)
    risk_score += np.where(df['sealevelpressure'] < 1008, 2.0,
                  np.where(df['sealevelpressure'] < 1012, 1.0, 0))
    
    # Wind speed scoring (moderate weight)
    # High risk: >35km/h (upper range of flood events)
    # Medium risk: >30km/h (elevated wind)
    risk_score += np.where(df['windspeed'] > 35, 1.5,
                  np.where(df['windspeed'] > 30, 0.75, 0))
    
    # Additional factors for more nuanced scoring
    
    # Temperature consideration (extreme temps can indicate severe weather)
    temp_range = df['tempmax'] - df['tempmin']
    risk_score += np.where(temp_range > 20, 0.5, 0)  # Large temperature swings
    
    # Cloud cover (heavy overcast can indicate severe weather systems)
    risk_score += np.where(df['cloudcover'] > 90, 0.5, 0)
    
    # Precipitation probability (even without actual precip, high probability is concerning)
    risk_score += np.where((df['precipprob'] > 80) & (df['precip'] > 5), 1.0,
                  np.where(df['precipprob'] > 90, 0.5, 0))
    
    # Convert risk scores to 3-class labels
    # Thresholds designed to create balanced distribution
    # High risk: score >= 6 (combinations of severe factors)
    # Medium risk: score >= 2.5 (some concerning factors)
    # Low risk: score < 2.5 (minimal risk factors)
    
    risk_severity = np.where(risk_score >= 6, 3,
                    np.where(risk_score >= 2.5, 2, 1))
    
    return risk_severity, risk_score

# Create 3-class risk severity labels
print("Creating 3-class risk severity labels...")
df['risk_severity'], df['risk_score'] = create_risk_severity_labels(df)

# Also keep a binary flood_risk for compatibility (high risk = 1, others = 0)
df['flood_risk'] = (df['risk_severity'] == 3).astype(int)

# Analyze risk distribution
print("\n=== RISK SEVERITY ANALYSIS ===")
print(f"Total records: {len(df)}")
print("\nRisk severity distribution:")

risk_dist = df['risk_severity'].value_counts().sort_index()
for level in [1, 2, 3]:
    count = risk_dist.get(level, 0)
    percentage = (count / len(df)) * 100
    risk_name = {1: "Low", 2: "Medium", 3: "High"}[level]
    print(f"  {level} ({risk_name} Risk): {count} records ({percentage:.1f}%)")

print(f"\nBinary flood_risk distribution (for compatibility):")
binary_dist = df['flood_risk'].value_counts().sort_index()
print(f"  0 (No Flood): {binary_dist.get(0, 0)} records")
print(f"  1 (Flood): {binary_dist.get(1, 0)} records")

# Save the cleaned and engineered dataset
df.to_csv("Maribyrnong_Cleaned_Weather_FloodRisk.csv", index=False)

print(f"\n[SUCCESS] Cleaned dataset saved as: Maribyrnong_Cleaned_Weather_FloodRisk.csv")
print("[SUCCESS] Dataset now includes:")
print("   - risk_severity: 3-class labels (1=Low, 2=Medium, 3=High)")
print("   - risk_score: Raw risk scores for analysis")
print("   - flood_risk: Binary labels for compatibility (High Risk only = 1)")
