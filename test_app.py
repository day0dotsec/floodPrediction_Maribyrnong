#!/usr/bin/env python3
"""
Test script to validate the flood prediction application setup
"""

import sys
import pandas as pd
import numpy as np

def test_data_loading():
    """Test if the CSV data can be loaded properly"""
    try:
        df = pd.read_csv('Maribyrnong_Cleaned_Weather_FloodRisk.csv')
        print(f"[OK] Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check for flood events
        flood_events = df['flood_risk'].sum()
        print(f"[OK] Found {flood_events} flood events in the dataset")
        
        # Check date range
        df['datetime'] = pd.to_datetime(df['datetime'])
        date_range = f"{df['datetime'].min()} to {df['datetime'].max()}"
        print(f"[OK] Date range: {date_range}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return False

def test_basic_analysis():
    """Test basic data analysis functions"""
    try:
        df = pd.read_csv('Maribyrnong_Cleaned_Weather_FloodRisk.csv')
        
        # Basic statistics
        temp_avg = df['temp'].mean()
        precip_avg = df['precip'].mean()
        humidity_avg = df['humidity'].mean()
        
        print(f"[OK] Average temperature: {temp_avg:.2f}C")
        print(f"[OK] Average precipitation: {precip_avg:.2f}mm")
        print(f"[OK] Average humidity: {humidity_avg:.2f}%")
        
        # Correlation analysis (only numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_with_flood = df[numeric_cols].corr()['flood_risk'].abs().sort_values(ascending=False)
        print(f"[OK] Top 3 correlations with flood risk:")
        for i, (var, corr) in enumerate(corr_with_flood.head(4)[1:].items()):
            print(f"   {i+1}. {var}: {corr:.3f}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Error in analysis: {e}")
        return False

def test_sklearn_functionality():
    """Test scikit-learn model functionality"""
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.cluster import KMeans
        from sklearn.model_selection import train_test_split
        
        df = pd.read_csv('Maribyrnong_Cleaned_Weather_FloodRisk.csv')
        
        # Prepare data for modeling
        feature_cols = ['tempmax', 'tempmin', 'temp', 'humidity', 'precip', 
                       'precipprob', 'windspeed', 'sealevelpressure', 'cloudcover']
        
        X = df[feature_cols].fillna(0)  # Handle any missing values
        y = df['flood_risk']
        
        # Test data scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"[OK] Data scaling successful: {X_scaled.shape}")
        
        # Test train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        print(f"[OK] Train-test split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
        
        # Test Logistic Regression
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        lr_score = lr_model.score(X_test, y_test)
        print(f"[OK] Logistic Regression accuracy: {lr_score:.4f}")
        
        # Test K-Means Clustering
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        print(f"[OK] K-Means clustering successful: {len(np.unique(clusters))} clusters")
        
        return True
    except Exception as e:
        print(f"[ERROR] Error in sklearn functionality: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Flood Prediction App Test Suite ===\n")
    
    print("1. Testing data loading...")
    data_ok = test_data_loading()
    print()
    
    print("2. Testing basic analysis...")
    analysis_ok = test_basic_analysis()
    print()
    
    print("3. Testing sklearn functionality...")
    sklearn_ok = test_sklearn_functionality()
    print()
    
    # Summary
    all_tests = [data_ok, analysis_ok, sklearn_ok]
    passed = sum(all_tests)
    total = len(all_tests)
    
    print("=== Test Summary ===")
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("[SUCCESS] All tests passed! The application should work correctly.")
        print("\nTo run the Streamlit app:")
        print("streamlit run flood_prediction_app_enhanced.py")
    else:
        print("[FAILED] Some tests failed. Please check the errors above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)