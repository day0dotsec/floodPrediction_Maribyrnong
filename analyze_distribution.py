import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('Maribyrnong_Cleaned_Weather_FloodRisk.csv')

print('=== FULL DATASET DISTRIBUTION ===')
print(f'Total records: {len(df)}')
print()

# Risk severity distribution (3-class)
print('Risk Severity (3-class) Distribution:')
risk_dist = df['risk_severity'].value_counts().sort_index()
for level in [1, 2, 3]:
    count = risk_dist.get(level, 0)
    percentage = (count / len(df)) * 100
    risk_name = {1: 'Low Risk', 2: 'Medium Risk', 3: 'High Risk'}[level]
    print(f'  Class {level} ({risk_name:11}): {count:5} records ({percentage:5.2f}%)')

print()

# Binary flood_risk distribution
print('Binary Flood Risk Distribution:')
binary_dist = df['flood_risk'].value_counts().sort_index()
for val in [0, 1]:
    count = binary_dist.get(val, 0)
    percentage = (count / len(df)) * 100
    label = {0: 'No Flood', 1: 'Flood (High Risk)'}[val]
    print(f'  {val} ({label:17}): {count:5} records ({percentage:5.2f}%)')

print()
print('=== TRAIN/TEST SPLIT (80/20, stratified) ===')
print()

# Simulate the train/test split used in the app
X = df[['tempmax', 'tempmin', 'temp', 'humidity', 'precip', 'precipprob', 'windspeed', 'sealevelpressure', 'cloudcover']]
y = df['risk_severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print('TRAINING SET:')
print(f'Total training samples: {len(y_train)}')
train_dist = y_train.value_counts().sort_index()
for level in [1, 2, 3]:
    count = train_dist.get(level, 0)
    percentage = (count / len(y_train)) * 100
    risk_name = {1: 'Low Risk', 2: 'Medium Risk', 3: 'High Risk'}[level]
    print(f'  Class {level} ({risk_name:11}): {count:5} records ({percentage:5.2f}%)')

print()
print('TEST SET:')
print(f'Total test samples: {len(y_test)}')
test_dist = y_test.value_counts().sort_index()
for level in [1, 2, 3]:
    count = test_dist.get(level, 0)
    percentage = (count / len(y_test)) * 100
    risk_name = {1: 'Low Risk', 2: 'Medium Risk', 3: 'High Risk'}[level]
    print(f'  Class {level} ({risk_name:11}): {count:5} records ({percentage:5.2f}%)')

print()
print('Class Imbalance Ratio:')
print(f'  Low : Medium : High = {train_dist.get(1,0)} : {train_dist.get(2,0)} : {train_dist.get(3,0)}')
low_to_high = train_dist.get(1,0) / train_dist.get(3,0) if train_dist.get(3,0) > 0 else 0
print(f'  Low Risk is {low_to_high:.1f}x more common than High Risk')
