import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
import os
warnings.filterwarnings('ignore')

# Configuration
API_KEY = "2M3N82RD42E4CWHEKB53PWSVW"
LOCATION = "Maribyrnong"
API_URL = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{LOCATION}?unitGroup=metric&key={API_KEY}&contentType=json"

# Page configuration
st.set_page_config(
    page_title="Maribyrnong Flood Prediction Dashboard with Alerts",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for alerts
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'email_config' not in st.session_state:
    st.session_state.email_config = {
        'enabled': False,
        'sender_email': '',
        'sender_password': '',
        'recipient_emails': []
    }
if 'alert_thresholds' not in st.session_state:
    st.session_state.alert_thresholds = {
        'high_precip': 25.0,
        'high_humidity': 95.0,
        'low_pressure': 1008.0,
        'high_windspeed': 35.0
    }

# Enhanced Custom CSS with alert styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding: 1rem 2rem 3rem 2rem;
        background: #ffffff;
        border-radius: 12px;
        margin: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1e293b;
        text-align: center;
        margin: 2rem 0;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
        animation: fadeInUp 1s ease-out;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Alert Styles */
    .alert-critical {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #dc2626;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .alert-info {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Enhanced Weather Cards */
    .weather-card {
        background: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        color: #1e293b;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        animation: slideIn 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .weather-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }
    
    .weather-card h3 {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .weather-value {
        font-size: 2rem;
        font-weight: 700;
        color: #3b82f6;
        margin: 0.5rem 0;
    }
    
    /* Risk Level Styling */
    .risk-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left-color: #dc2626;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left-color: #f59e0b;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #dcfce7 0%, #4ade80 100%);
        border-left-color: #16a34a;
    }
    
    /* Sidebar Enhancements - Force white background with comprehensive selectors */
    .css-1d391kg, section[data-testid="stSidebar"], .stSidebar, 
    [data-testid="stSidebar"] > div, section[data-testid="stSidebar"] > div > div,
    .css-1d391kg > div, section[data-testid="stSidebar"] .element-container {
        background-color: #ffffff !important;
        background: #ffffff !important;
    }
    
    /* Sidebar container styling */
    .css-1d391kg, section[data-testid="stSidebar"] {
        border-radius: 0 12px 12px 0;
        border-right: 1px solid #e2e8f0;
        box-shadow: 4px 0 16px rgba(0, 0, 0, 0.08);
    }
    
    /* Force all sidebar text to be dark with comprehensive selectors */
    .css-1d391kg *, section[data-testid="stSidebar"] *, [data-testid="stSidebar"] *,
    .stSidebar *, section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div, section[data-testid="stSidebar"] span,
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg p,
    .css-1d391kg div, .css-1d391kg span {
        color: #1e293b !important;
    }
    
    /* Sidebar selectbox styling */
    .css-1d391kg .stSelectbox label, section[data-testid="stSidebar"] .stSelectbox label {
        color: #1e293b !important;
    }
    
    .css-1d391kg .stSelectbox > div > div, section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* Override Streamlit's built-in theme detection */
    @media (prefers-color-scheme: dark) {
        section[data-testid="stSidebar"], .css-1d391kg {
            background-color: #ffffff !important;
        }
        section[data-testid="stSidebar"] *, .css-1d391kg * {
            color: #1e293b !important;
        }
    }
    
    /* Button Enhancements */
    .stButton > button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2);
    }
    
    .stButton > button:hover {
        background: #1d4ed8;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Selectbox Enhancements */
    .stSelectbox > div > div {
        background: #ffffff !important;
        border-radius: 8px;
        border: 1px solid #d1d5db !important;
        color: #1e293b !important;
    }
    
    .stSelectbox > div > div > div {
        color: #1e293b !important;
    }
    
    .stSelectbox [data-baseweb="select"] {
        color: #1e293b !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        color: #1e293b !important;
        background-color: #ffffff !important;
    }
    
    /* Dropdown options styling - more specific selectors */
    div[data-baseweb="popover"] {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
    }
    
    ul[data-baseweb="menu"] {
        background-color: #ffffff !important;
    }
    
    li[role="option"] {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    
    li[role="option"]:hover {
        background-color: #f1f5f9 !important;
        color: #1e293b !important;
    }
    
    /* DataFrames styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Plotly chart styling */
    .js-plotly-plot {
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
        border-radius: 4px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [aria-selected="true"] {
        background: #2563eb;
        color: white;
        border-color: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# Email notification functions
def send_email_alert(subject, body, recipient_emails):
    """Send email alert to specified recipients"""
    config = st.session_state.email_config
    
    if not config['enabled'] or not config['sender_email'] or not config['sender_password']:
        return False, "Email configuration not set"
    
    try:
        message = MIMEMultipart()
        message["From"] = config['sender_email']
        message["Subject"] = subject
        
        message.attach(MIMEText(body, "html"))
        
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(config['sender_email'], config['sender_password'])
        
        success_count = 0
        for recipient in recipient_emails:
            try:
                message["To"] = recipient
                text = message.as_string()
                server.sendmail(config['sender_email'], recipient, text)
                success_count += 1
                del message["To"]  # Remove for next iteration
            except Exception as e:
                st.error(f"Failed to send to {recipient}: {str(e)}")
        
        server.quit()
        return success_count > 0, f"Sent to {success_count}/{len(recipient_emails)} recipients"
        
    except Exception as e:
        return False, f"Email server error: {str(e)}"

def check_alert_conditions(weather_data, current_risk):
    """Check if current conditions warrant an alert"""
    thresholds = st.session_state.alert_thresholds
    alerts = []
    
    # Check precipitation
    if weather_data.get('precip', 0) > thresholds['high_precip']:
        alerts.append({
            'type': 'CRITICAL',
            'condition': 'High Precipitation',
            'value': weather_data.get('precip', 0),
            'threshold': thresholds['high_precip'],
            'message': f"Precipitation ({weather_data.get('precip', 0):.1f}mm) exceeds critical threshold ({thresholds['high_precip']:.1f}mm)"
        })
    
    # Check humidity
    if weather_data.get('humidity', 0) > thresholds['high_humidity']:
        alerts.append({
            'type': 'WARNING',
            'condition': 'High Humidity',
            'value': weather_data.get('humidity', 0),
            'threshold': thresholds['high_humidity'],
            'message': f"Humidity ({weather_data.get('humidity', 0):.1f}%) exceeds warning threshold ({thresholds['high_humidity']:.1f}%)"
        })
    
    # Check pressure
    if weather_data.get('sealevelpressure', 1013) < thresholds['low_pressure']:
        alerts.append({
            'type': 'WARNING',
            'condition': 'Low Pressure',
            'value': weather_data.get('sealevelpressure', 1013),
            'threshold': thresholds['low_pressure'],
            'message': f"Pressure ({weather_data.get('sealevelpressure', 1013):.1f} hPa) below warning threshold ({thresholds['low_pressure']:.1f} hPa)"
        })
    
    # Check wind speed
    if weather_data.get('windspeed', 0) > thresholds['high_windspeed']:
        alerts.append({
            'type': 'INFO',
            'condition': 'High Wind Speed',
            'value': weather_data.get('windspeed', 0),
            'threshold': thresholds['high_windspeed'],
            'message': f"Wind speed ({weather_data.get('windspeed', 0):.1f} km/h) exceeds threshold ({thresholds['high_windspeed']:.1f} km/h)"
        })
    
    # Overall flood risk assessment
    if current_risk == "High":
        alerts.append({
            'type': 'CRITICAL',
            'condition': 'High Flood Risk',
            'value': current_risk,
            'threshold': 'High Risk Conditions',
            'message': f"Overall flood risk is {current_risk} - Immediate attention required"
        })
    elif current_risk == "Medium":
        alerts.append({
            'type': 'WARNING',
            'condition': 'Medium Flood Risk',
            'value': current_risk,
            'threshold': 'Medium Risk Conditions',
            'message': f"Overall flood risk is {current_risk} - Monitor conditions closely"
        })
    
    return alerts

def log_alert(alert_type, condition, message):
    """Log alert to session state history"""
    alert_entry = {
        'timestamp': datetime.now(),
        'type': alert_type,
        'condition': condition,
        'message': message,
        'email_sent': False
    }
    st.session_state.alert_history.insert(0, alert_entry)
    
    # Keep only last 50 alerts
    if len(st.session_state.alert_history) > 50:
        st.session_state.alert_history = st.session_state.alert_history[:50]

def send_flood_alert_email(alerts, weather_data):
    """Send comprehensive flood alert email"""
    if not st.session_state.email_config['enabled'] or not st.session_state.email_config['recipient_emails']:
        return False, "Email not configured"
    
    # Determine highest severity
    severity_order = {'CRITICAL': 3, 'WARNING': 2, 'INFO': 1}
    highest_severity = max(alerts, key=lambda x: severity_order.get(x['type'], 0))['type']
    
    subject = f"🚨 Maribyrnong Flood Alert - {highest_severity} CONDITIONS"
    
    # Create HTML email body
    body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); color: white; padding: 20px; text-align: center;">
            <h1>🌊 Maribyrnong Flood Alert System</h1>
            <h2 style="color: #fbbf24;">{highest_severity} CONDITIONS DETECTED</h2>
        </div>
        
        <div style="padding: 20px; background: #f8fafc;">
            <h3>Current Weather Conditions:</h3>
            <ul style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <li><strong>Precipitation:</strong> {weather_data.get('precip', 'N/A')} mm</li>
                <li><strong>Humidity:</strong> {weather_data.get('humidity', 'N/A')}%</li>
                <li><strong>Temperature:</strong> {weather_data.get('temp', 'N/A')}°C</li>
                <li><strong>Pressure:</strong> {weather_data.get('sealevelpressure', 'N/A')} hPa</li>
                <li><strong>Wind Speed:</strong> {weather_data.get('windspeed', 'N/A')} km/h</li>
            </ul>
            
            <h3>Alert Conditions Triggered:</h3>
    """
    
    for alert in alerts:
        color = "#dc2626" if alert['type'] == 'CRITICAL' else "#f59e0b" if alert['type'] == 'WARNING' else "#3b82f6"
        body += f"""
            <div style="background: white; padding: 15px; margin: 10px 0; border-left: 4px solid {color}; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="color: {color}; margin: 0 0 5px 0;">{alert['type']}: {alert['condition']}</h4>
                <p style="margin: 0;">{alert['message']}</p>
            </div>
        """
    
    body += f"""
            <div style="background: #1e293b; color: white; padding: 15px; border-radius: 8px; margin-top: 20px; text-align: center;">
                <p><strong>Alert generated at:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><em>This is an automated alert from the Maribyrnong Flood Prediction System</em></p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return send_email_alert(subject, body, st.session_state.email_config['recipient_emails'])

# Data loading functions
@st.cache_data
def fetch_weather_data():
    """Fetch current weather data from API"""
    try:
        response = requests.get(API_URL)
        data = response.json()
        current_conditions = data['currentConditions']
        return current_conditions
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

@st.cache_data
def load_historical_data():
    """Load historical weather and flood data"""
    try:
        df = pd.read_csv('Maribyrnong_Cleaned_Weather_FloodRisk.csv')
        return df
    except FileNotFoundError:
        st.error("Historical data file not found. Please ensure 'Maribyrnong_Cleaned_Weather_FloodRisk.csv' is in the current directory.")
        return None
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return None

def assess_flood_risk(weather_data):
    """Assess current flood risk based on weather conditions"""
    if weather_data is None:
        return "Unknown", {}
    
    risk_factors = {
        'High Precipitation': weather_data.get('precip', 0) > 25.0,
        'High Humidity': weather_data.get('humidity', 0) > 95.0,
        'Low Pressure': weather_data.get('sealevelpressure', 1013) < 1008.0,
        'High Wind Speed': weather_data.get('windspeed', 0) > 35
    }
    
    risk_count = sum(risk_factors.values())
    
    if risk_count >= 3:
        return "High", risk_factors
    elif risk_count >= 2:
        return "Medium", risk_factors
    else:
        return "Low", risk_factors

# LSTM Neural Network Implementation
def create_lstm_features(df):
    """Create features for LSTM model"""
    feature_columns = ['temp', 'humidity', 'precip', 'windspeed', 'sealevelpressure']
    features = df[feature_columns].fillna(df[feature_columns].mean())
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, scaler, feature_columns

def build_lstm_model(X_train, y_train):
    """Build and train LSTM model using PyTorch"""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden_size, output_size)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                out = self.sigmoid(out)
                return out
        
        # Prepare data
        X_tensor = torch.FloatTensor(X_train.reshape(X_train.shape[0], 1, X_train.shape[1]))
        y_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
        
        # Model parameters
        input_size = X_train.shape[1]
        hidden_size = 50
        num_layers = 2
        output_size = 1
        
        model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training
        num_epochs = 100
        for epoch in range(num_epochs):
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return model
        
    except ImportError:
        st.warning("PyTorch not available. Using Logistic Regression instead.")
        return None

# Machine Learning Models
def train_models(df):
    """Train various ML models for flood prediction"""
    feature_columns = ['temp', 'humidity', 'precip', 'windspeed', 'sealevelpressure']
    X = df[feature_columns].fillna(df[feature_columns].mean())
    y = df['flood_risk']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    
    # Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_score = lr_model.score(X_test_scaled, y_test)
    models['Logistic Regression'] = {'model': lr_model, 'scaler': scaler, 'score': lr_score}
    
    # K-Means Clustering for risk zones
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)
    models['K-Means'] = {'model': kmeans, 'scaler': scaler}
    
    # LSTM Neural Network
    lstm_model = build_lstm_model(X_train_scaled, y_train.values)
    if lstm_model:
        models['LSTM'] = {'model': lstm_model, 'scaler': scaler}
    
    return models, X_test_scaled, y_test

def predict_flood_risk_ml(models, current_weather):
    """Predict flood risk using trained ML models"""
    predictions = {}
    
    if current_weather is None:
        return predictions
    
    # Prepare current weather data
    feature_values = [
        current_weather.get('temp', 20),
        current_weather.get('humidity', 60),
        current_weather.get('precip', 0),
        current_weather.get('windspeed', 10),
        current_weather.get('sealevelpressure', 1013)
    ]
    
    # Logistic Regression prediction
    if 'Logistic Regression' in models:
        lr_model = models['Logistic Regression']['model']
        lr_scaler = models['Logistic Regression']['scaler']
        scaled_features = lr_scaler.transform([feature_values])
        lr_pred = lr_model.predict(scaled_features)[0]
        lr_proba = lr_model.predict_proba(scaled_features)[0][1]
        predictions['Logistic Regression'] = {'prediction': lr_pred, 'probability': lr_proba}
    
    # K-Means clustering
    if 'K-Means' in models:
        kmeans_model = models['K-Means']['model']
        kmeans_scaler = models['K-Means']['scaler']
        scaled_features = kmeans_scaler.transform([feature_values])
        cluster = kmeans_model.predict(scaled_features)[0]
        predictions['K-Means'] = {'cluster': cluster}
    
    # LSTM prediction
    if 'LSTM' in models:
        try:
            import torch
            lstm_model = models['LSTM']['model']
            lstm_scaler = models['LSTM']['scaler']
            scaled_features = lstm_scaler.transform([feature_values])
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(scaled_features.reshape(1, 1, -1))
                lstm_pred = lstm_model(X_tensor).item()
                predictions['LSTM'] = {'probability': lstm_pred}
        except:
            pass
    
    return predictions

# Main Application
def main():
    # Header
    st.markdown('<h1 class="main-header">🌊 Maribyrnong Flood Prediction Dashboard with Email Alerts</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation and alert configuration
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 2rem; background: #ffffff !important;">
            <h2 style="color: #1e293b !important; font-weight: 600;">🌊 Navigation</h2>
            <p style="color: #64748b !important; font-size: 0.9rem;">Select a section to explore</p>
        </div>
        """, unsafe_allow_html=True)
        
        main_section = st.selectbox(
            "Main Section",
            ["🏠 Home & Current Weather", "📧 Alert Configuration", "🤖 Machine Learning Models", "📊 Data Analysis", "🔍 Insights & Predictions", "📝 Alert History"],
            format_func=lambda x: x,
            label_visibility="collapsed"
        )
    
    # Load data
    weather_data = fetch_weather_data()
    historical_df = load_historical_data()
    
    if historical_df is None:
        st.error("Unable to load historical data. Some features may not be available.")
        return
    
    # Main content based on selection
    if main_section == "🏠 Home & Current Weather":
        show_current_weather_with_alerts(weather_data)
    
    elif main_section == "📧 Alert Configuration":
        show_alert_configuration()
    
    elif main_section == "🤖 Machine Learning Models":
        show_ml_models(historical_df, weather_data)
    
    elif main_section == "📊 Data Analysis":
        show_data_analysis(historical_df)
    
    elif main_section == "🔍 Insights & Predictions":
        show_insights_predictions(historical_df, weather_data)
    
    elif main_section == "📝 Alert History":
        show_alert_history()

def show_current_weather_with_alerts(weather_data):
    """Display current weather with real-time alert monitoring"""
    
    # Get current risk assessment
    current_risk, risk_factors = assess_flood_risk(weather_data)
    
    # Check for alert conditions
    alerts = check_alert_conditions(weather_data, current_risk)
    
    # Display alerts if any
    if alerts:
        st.markdown("## 🚨 Active Alerts")
        
        for alert in alerts:
            # Log alert
            log_alert(alert['type'], alert['condition'], alert['message'])
            
            # Display alert
            if alert['type'] == 'CRITICAL':
                st.markdown(f"""
                <div class="alert-critical">
                    <h3 style="color: #dc2626; margin: 0 0 10px 0;">🚨 CRITICAL: {alert['condition']}</h3>
                    <p style="color: #1e293b; margin: 0; font-weight: 500;">{alert['message']}</p>
                </div>
                """, unsafe_allow_html=True)
            elif alert['type'] == 'WARNING':
                st.markdown(f"""
                <div class="alert-warning">
                    <h3 style="color: #f59e0b; margin: 0 0 10px 0;">⚠️ WARNING: {alert['condition']}</h3>
                    <p style="color: #1e293b; margin: 0; font-weight: 500;">{alert['message']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-info">
                    <h3 style="color: #3b82f6; margin: 0 0 10px 0;">ℹ️ INFO: {alert['condition']}</h3>
                    <p style="color: #1e293b; margin: 0; font-weight: 500;">{alert['message']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Send email alerts if enabled
        if st.session_state.email_config['enabled'] and st.session_state.email_config['recipient_emails']:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("📧 Send Alert Email Now", type="primary"):
                    with st.spinner("Sending alert email..."):
                        success, message = send_flood_alert_email(alerts, weather_data)
                        if success:
                            st.success(f"✅ Alert email sent! {message}")
                            # Update alert history to mark email as sent
                            for i, entry in enumerate(st.session_state.alert_history):
                                if not entry['email_sent']:
                                    st.session_state.alert_history[i]['email_sent'] = True
                        else:
                            st.error(f"❌ Failed to send alert email: {message}")
            
            with col2:
                auto_email = st.checkbox("🔄 Auto-send alert emails", value=False)
                if auto_email:
                    with st.spinner("Sending automatic alert email..."):
                        success, message = send_flood_alert_email(alerts, weather_data)
                        if success:
                            st.success("📧 Automatic alert email sent!")
                        else:
                            st.warning(f"Automatic email failed: {message}")
    
    # Current Weather Display
    st.markdown("## 🌤️ Current Weather Conditions")
    
    if weather_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="weather-card">
                <h3>🌡️ Temperature</h3>
                <div class="weather-value">{weather_data.get('temp', 'N/A')}°C</div>
                <p style="color: #64748b;">Feels like {weather_data.get('feelslike', 'N/A')}°C</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="weather-card">
                <h3>💧 Precipitation</h3>
                <div class="weather-value">{weather_data.get('precip', 'N/A')} mm</div>
                <p style="color: #64748b;">Last 24 hours</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="weather-card">
                <h3>💨 Humidity</h3>
                <div class="weather-value">{weather_data.get('humidity', 'N/A')}%</div>
                <p style="color: #64748b;">Relative humidity</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="weather-card">
                <h3>🌬️ Wind Speed</h3>
                <div class="weather-value">{weather_data.get('windspeed', 'N/A')} km/h</div>
                <p style="color: #64748b;">Current wind speed</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional weather details
        col5, col6 = st.columns(2)
        
        with col5:
            st.markdown(f"""
            <div class="weather-card">
                <h3>📊 Atmospheric Pressure</h3>
                <div class="weather-value">{weather_data.get('sealevelpressure', 'N/A')} hPa</div>
                <p style="color: #64748b;">Sea level pressure</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            st.markdown(f"""
            <div class="weather-card">
                <h3>👁️ Visibility</h3>
                <div class="weather-value">{weather_data.get('visibility', 'N/A')} km</div>
                <p style="color: #64748b;">Current visibility</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Flood Risk Assessment
        st.markdown("## 🌊 Current Flood Risk Assessment")
        
        risk_class = f"risk-{current_risk.lower()}"
        risk_color = "#dc2626" if current_risk == "High" else "#f59e0b" if current_risk == "Medium" else "#16a34a"
        
        st.markdown(f"""
        <div class="weather-card {risk_class}" style="border-left: 6px solid {risk_color};">
            <h3>Current Risk Level: <span style="color: {risk_color};">{current_risk.upper()}</span></h3>
            <p style="font-size: 1.1rem; margin: 1rem 0;">
                Based on current weather conditions, the flood risk for Maribyrnong River is assessed as <strong>{current_risk}</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk Factors Breakdown
        st.markdown("### Risk Factors Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Conditions:**")
            for factor, triggered in risk_factors.items():
                icon = "🔴" if triggered else "🟢"
                st.markdown(f"{icon} {factor}")
        
        with col2:
            st.markdown("**Thresholds:**")
            thresholds = st.session_state.alert_thresholds
            st.markdown(f"• Precipitation: >{thresholds['high_precip']}mm")
            st.markdown(f"• Humidity: >{thresholds['high_humidity']}%")
            st.markdown(f"• Pressure: <{thresholds['low_pressure']} hPa")
            st.markdown(f"• Wind Speed: >{thresholds['high_windspeed']} km/h")
    
    else:
        st.error("Unable to fetch current weather data. Please check your internet connection.")

def show_alert_configuration():
    """Display alert configuration interface"""
    st.markdown("## 📧 Email Alert Configuration")
    
    # Email Settings
    st.markdown("### Email Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_email = st.checkbox("Enable Email Alerts", value=st.session_state.email_config['enabled'])
        st.session_state.email_config['enabled'] = enable_email
        
        sender_email = st.text_input(
            "Sender Email (Gmail)", 
            value=st.session_state.email_config['sender_email'],
            help="Use your Gmail address. You'll need an App Password, not your regular password."
        )
        st.session_state.email_config['sender_email'] = sender_email
        
        sender_password = st.text_input(
            "App Password", 
            type="password",
            help="Generate an App Password in your Google Account settings"
        )
        if sender_password:
            st.session_state.email_config['sender_password'] = sender_password
    
    with col2:
        st.markdown("### 📧 Gmail App Password Setup")
        st.markdown("""
        1. Go to Google Account settings
        2. Enable 2-Factor Authentication
        3. Go to Security > App passwords
        4. Generate password for 'Mail'
        5. Use that 16-character password here
        """)
        
        if st.button("Test Email Configuration"):
            if enable_email and sender_email and sender_password:
                with st.spinner("Testing email configuration..."):
                    test_subject = "🌊 Maribyrnong Flood Alert System - Test"
                    test_body = """
                    <html><body style="font-family: Arial, sans-serif;">
                    <h2>Email Alert System Test</h2>
                    <p>This is a test email from the Maribyrnong Flood Prediction System.</p>
                    <p>If you received this email, your configuration is working correctly!</p>
                    <p><em>Test sent at: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</em></p>
                    </body></html>
                    """
                    
                    success, message = send_email_alert(test_subject, test_body, [sender_email])
                    
                    if success:
                        st.success("✅ Test email sent successfully!")
                    else:
                        st.error(f"❌ Test failed: {message}")
            else:
                st.warning("Please fill in all email configuration fields first.")
    
    # Recipient Management
    st.markdown("### 📬 Alert Recipients")
    
    new_recipient = st.text_input("Add recipient email:")
    if st.button("Add Recipient") and new_recipient:
        if new_recipient not in st.session_state.email_config['recipient_emails']:
            st.session_state.email_config['recipient_emails'].append(new_recipient)
            st.success(f"Added {new_recipient} to alert recipients")
        else:
            st.warning("Email already in recipient list")
    
    # Display current recipients
    if st.session_state.email_config['recipient_emails']:
        st.markdown("**Current Recipients:**")
        for i, email in enumerate(st.session_state.email_config['recipient_emails']):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"📧 {email}")
            with col2:
                if st.button(f"Remove", key=f"remove_{i}"):
                    st.session_state.email_config['recipient_emails'].remove(email)
                    st.rerun()
    else:
        st.info("No recipients configured yet.")
    
    # Alert Threshold Configuration
    st.markdown("### ⚠️ Alert Thresholds")
    st.markdown("Configure the conditions that will trigger alerts:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.alert_thresholds['high_precip'] = st.number_input(
            "High Precipitation (mm)", 
            min_value=0.0, 
            max_value=100.0,
            value=st.session_state.alert_thresholds['high_precip'],
            step=0.5,
            help="Precipitation threshold that triggers critical alerts"
        )
        
        st.session_state.alert_thresholds['high_humidity'] = st.number_input(
            "High Humidity (%)", 
            min_value=50.0, 
            max_value=100.0,
            value=st.session_state.alert_thresholds['high_humidity'],
            step=1.0,
            help="Humidity threshold that triggers warning alerts"
        )
    
    with col2:
        st.session_state.alert_thresholds['low_pressure'] = st.number_input(
            "Low Pressure (hPa)", 
            min_value=900.0, 
            max_value=1050.0,
            value=st.session_state.alert_thresholds['low_pressure'],
            step=1.0,
            help="Pressure threshold below which warnings are triggered"
        )
        
        st.session_state.alert_thresholds['high_windspeed'] = st.number_input(
            "High Wind Speed (km/h)", 
            min_value=0.0, 
            max_value=200.0,
            value=st.session_state.alert_thresholds['high_windspeed'],
            step=1.0,
            help="Wind speed threshold that triggers info alerts"
        )
    
    # Save Configuration
    if st.button("💾 Save Configuration", type="primary"):
        st.success("✅ Alert configuration saved!")
        
        # Display current configuration summary
        st.markdown("### 📋 Configuration Summary")
        st.json({
            "Email Alerts Enabled": st.session_state.email_config['enabled'],
            "Sender Email": st.session_state.email_config['sender_email'],
            "Recipients Count": len(st.session_state.email_config['recipient_emails']),
            "Alert Thresholds": st.session_state.alert_thresholds
        })

def show_ml_models(historical_df, current_weather):
    """Display machine learning models section with alert integration"""
    st.markdown("## 🤖 Machine Learning Models")
    
    if historical_df is None:
        st.error("Historical data not available for model training.")
        return
    
    # Train models
    with st.spinner("Training machine learning models..."):
        models, X_test, y_test = train_models(historical_df)
        predictions = predict_flood_risk_ml(models, current_weather)
    
    # Model Performance
    st.markdown("### 📊 Model Performance")
    
    if 'Logistic Regression' in models:
        lr_score = models['Logistic Regression']['score']
        st.metric("Logistic Regression Accuracy", f"{lr_score:.3f}")
        
        if lr_score > 0.8:
            st.success("✅ High accuracy model")
        elif lr_score > 0.6:
            st.warning("⚠️ Moderate accuracy model")
        else:
            st.error("❌ Low accuracy model - consider more data")
    
    # Current Predictions
    st.markdown("### 🔮 Current Weather Predictions")
    
    if predictions:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'Logistic Regression' in predictions:
                lr_pred = predictions['Logistic Regression']
                risk_text = "High Risk" if lr_pred['prediction'] == 1 else "Low Risk"
                probability = lr_pred['probability']
                
                color = "#dc2626" if lr_pred['prediction'] == 1 else "#16a34a"
                
                st.markdown(f"""
                <div class="weather-card" style="border-left: 4px solid {color};">
                    <h3>🧠 Logistic Regression</h3>
                    <div class="weather-value" style="color: {color};">{risk_text}</div>
                    <p style="color: #64748b;">Probability: {probability:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Check if ML model predicts high risk - add to alerts
                if lr_pred['prediction'] == 1 and probability > 0.7:
                    st.warning(f"⚠️ ML Model Alert: High flood risk detected (probability: {probability:.3f})")
        
        with col2:
            if 'K-Means' in predictions:
                cluster = predictions['K-Means']['cluster']
                cluster_names = {0: "Low Risk Zone", 1: "Medium Risk Zone", 2: "High Risk Zone"}
                cluster_name = cluster_names.get(cluster, f"Cluster {cluster}")
                colors = {0: "#16a34a", 1: "#f59e0b", 2: "#dc2626"}
                color = colors.get(cluster, "#64748b")
                
                st.markdown(f"""
                <div class="weather-card" style="border-left: 4px solid {color};">
                    <h3>🎯 K-Means Clustering</h3>
                    <div class="weather-value" style="color: {color};">{cluster_name}</div>
                    <p style="color: #64748b;">Cluster: {cluster}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if 'LSTM' in predictions:
                lstm_prob = predictions['LSTM']['probability']
                risk_text = "High Risk" if lstm_prob > 0.5 else "Low Risk"
                color = "#dc2626" if lstm_prob > 0.5 else "#16a34a"
                
                st.markdown(f"""
                <div class="weather-card" style="border-left: 4px solid {color};">
                    <h3>🧬 LSTM Neural Network</h3>
                    <div class="weather-value" style="color: {color};">{risk_text}</div>
                    <p style="color: #64748b;">Probability: {lstm_prob:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Feature Importance (for Logistic Regression)
    if 'Logistic Regression' in models:
        st.markdown("### 📈 Feature Importance Analysis")
        
        lr_model = models['Logistic Regression']['model']
        feature_names = ['Temperature', 'Humidity', 'Precipitation', 'Wind Speed', 'Pressure']
        coefficients = lr_model.coef_[0]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(coefficients),
            'Effect': ['Positive' if c > 0 else 'Negative' for c in coefficients]
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, 
                    x='Importance', 
                    y='Feature', 
                    color='Effect',
                    orientation='h',
                    title="Feature Importance in Flood Risk Prediction",
                    color_discrete_map={'Positive': '#dc2626', 'Negative': '#16a34a'})
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#1e293b')
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_data_analysis(historical_df):
    """Display comprehensive data analysis with alert context"""
    st.markdown("## 📊 Historical Data Analysis")
    
    if historical_df is None:
        st.error("Historical data not available.")
        return
    
    # Data overview
    st.markdown("### 📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(historical_df))
    with col2:
        flood_events = historical_df['flood_risk'].sum()
        st.metric("Flood Events", flood_events)
    with col3:
        flood_rate = (flood_events / len(historical_df)) * 100
        st.metric("Flood Rate", f"{flood_rate:.2f}%")
    with col4:
        date_range = f"{len(pd.date_range(historical_df['datetime'].min(), historical_df['datetime'].max()))} days"
        st.metric("Date Range", date_range)
    
    # Flood events analysis
    st.markdown("### 🌊 Flood Events Analysis")
    
    flood_data = historical_df[historical_df['flood_risk'] == 1].copy()
    
    if len(flood_data) > 0:
        st.markdown("**Recorded Flood Events:**")
        
        # Display flood events with alert-style formatting
        for idx, event in flood_data.iterrows():
            event_date = pd.to_datetime(event['datetime']).strftime('%Y-%m-%d')
            
            st.markdown(f"""
            <div class="alert-warning">
                <h4 style="color: #f59e0b; margin: 0 0 10px 0;">🌊 Flood Event: {event_date}</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-top: 10px;">
                    <div><strong>Precipitation:</strong> {event.get('precip', 'N/A')} mm</div>
                    <div><strong>Humidity:</strong> {event.get('humidity', 'N/A')}%</div>
                    <div><strong>Temperature:</strong> {event.get('temp', 'N/A')}°C</div>
                    <div><strong>Pressure:</strong> {event.get('sealevelpressure', 'N/A')} hPa</div>
                    <div><strong>Wind Speed:</strong> {event.get('windspeed', 'N/A')} km/h</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Flood conditions analysis
        st.markdown("### 📈 Flood Condition Patterns")
        
        # Create comparison between flood and non-flood days
        flood_stats = flood_data[['temp', 'humidity', 'precip', 'windspeed', 'sealevelpressure']].mean()
        normal_stats = historical_df[historical_df['flood_risk'] == 0][['temp', 'humidity', 'precip', 'windspeed', 'sealevelpressure']].mean()
        
        comparison_df = pd.DataFrame({
            'Flood Days': flood_stats,
            'Normal Days': normal_stats,
            'Difference': flood_stats - normal_stats
        }).round(2)
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Alert threshold comparison
        st.markdown("### ⚠️ Current Alert Thresholds vs Historical Flood Data")
        
        thresholds = st.session_state.alert_thresholds
        threshold_analysis = []
        
        for param, threshold in [
            ('precip', thresholds['high_precip']),
            ('humidity', thresholds['high_humidity']),
            ('sealevelpressure', thresholds['low_pressure']),
            ('windspeed', thresholds['high_windspeed'])
        ]:
            if param in flood_data.columns:
                flood_values = flood_data[param].dropna()
                if len(flood_values) > 0:
                    min_val = flood_values.min()
                    max_val = flood_values.max()
                    mean_val = flood_values.mean()
                    
                    if param == 'sealevelpressure':
                        # For pressure, we check if threshold is above maximum (since low pressure triggers alerts)
                        threshold_status = "✅ Appropriate" if threshold > max_val else "⚠️ May need adjustment"
                    else:
                        # For other parameters, check if threshold is below minimum
                        threshold_status = "✅ Appropriate" if threshold < min_val else "⚠️ May need adjustment"
                    
                    threshold_analysis.append({
                        'Parameter': param.title(),
                        'Alert Threshold': f"{threshold}",
                        'Flood Min': f"{min_val:.2f}",
                        'Flood Max': f"{max_val:.2f}",
                        'Flood Mean': f"{mean_val:.2f}",
                        'Status': threshold_status
                    })
        
        threshold_df = pd.DataFrame(threshold_analysis)
        st.dataframe(threshold_df, use_container_width=True)
    
    # Interactive flood events exploration
    st.markdown("### 🔍 Interactive Flood Events Explorer")
    
    # Create a chart showing all weather conditions with flood events highlighted
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Precipitation vs Flood Risk', 'Humidity vs Flood Risk', 
                       'Temperature vs Flood Risk', 'Pressure vs Flood Risk'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Precipitation
    fig.add_trace(
        go.Scatter(
            x=historical_df['datetime'],
            y=historical_df['precip'],
            mode='markers',
            marker=dict(color=historical_df['flood_risk'], 
                       colorscale=['lightblue', 'red'],
                       size=8),
            name='Precipitation',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add threshold line for precipitation
    fig.add_hline(
        y=thresholds['high_precip'], 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Alert Threshold: {thresholds['high_precip']}mm",
        row=1, col=1
    )
    
    # Humidity
    fig.add_trace(
        go.Scatter(
            x=historical_df['datetime'],
            y=historical_df['humidity'],
            mode='markers',
            marker=dict(color=historical_df['flood_risk'], 
                       colorscale=['lightgreen', 'red'],
                       size=8),
            name='Humidity',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_hline(
        y=thresholds['high_humidity'], 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Alert Threshold: {thresholds['high_humidity']}%",
        row=1, col=2
    )
    
    # Temperature
    fig.add_trace(
        go.Scatter(
            x=historical_df['datetime'],
            y=historical_df['temp'],
            mode='markers',
            marker=dict(color=historical_df['flood_risk'], 
                       colorscale=['lightcyan', 'red'],
                       size=8),
            name='Temperature',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Pressure
    fig.add_trace(
        go.Scatter(
            x=historical_df['datetime'],
            y=historical_df['sealevelpressure'],
            mode='markers',
            marker=dict(color=historical_df['flood_risk'], 
                       colorscale=['lightyellow', 'red'],
                       size=8),
            name='Pressure',
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.add_hline(
        y=thresholds['low_pressure'], 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Alert Threshold: {thresholds['low_pressure']} hPa",
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        title_text="Weather Conditions vs Flood Events (Red = Flood Day, Blue/Green/Cyan/Yellow = Normal Day)",
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Precipitation (mm)", row=1, col=1)
    fig.update_yaxes(title_text="Humidity (%)", row=1, col=2)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Pressure (hPa)", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

def show_insights_predictions(historical_df, current_weather):
    """Display insights and predictions with alert integration"""
    st.markdown("## 🔍 Insights & Predictions")
    
    if historical_df is None:
        st.error("Historical data not available for analysis.")
        return
    
    # Current conditions analysis
    st.markdown("### 🌤️ Current Conditions Analysis")
    
    if current_weather:
        # Compare current conditions to historical flood events
        flood_data = historical_df[historical_df['flood_risk'] == 1]
        
        if len(flood_data) > 0:
            current_conditions = {
                'precip': current_weather.get('precip', 0),
                'humidity': current_weather.get('humidity', 60),
                'temp': current_weather.get('temp', 20),
                'windspeed': current_weather.get('windspeed', 10),
                'sealevelpressure': current_weather.get('sealevelpressure', 1013)
            }
            
            # Calculate similarity to historical flood events
            similarities = []
            for idx, event in flood_data.iterrows():
                similarity_score = 0
                comparisons = 0
                
                for param in current_conditions.keys():
                    if param in event and not pd.isna(event[param]):
                        current_val = current_conditions[param]
                        flood_val = event[param]
                        
                        # Normalize the difference (0-1 scale where 1 is identical)
                        if param == 'temp':
                            max_diff = 20  # Temperature difference threshold
                        elif param == 'humidity':
                            max_diff = 30  # Humidity difference threshold
                        elif param == 'precip':
                            max_diff = 50  # Precipitation difference threshold
                        elif param == 'windspeed':
                            max_diff = 40  # Wind speed difference threshold
                        else:  # pressure
                            max_diff = 50  # Pressure difference threshold
                        
                        diff = abs(current_val - flood_val)
                        similarity = max(0, 1 - (diff / max_diff))
                        similarity_score += similarity
                        comparisons += 1
                
                if comparisons > 0:
                    avg_similarity = similarity_score / comparisons
                    event_date = pd.to_datetime(event['datetime']).strftime('%Y-%m-%d')
                    similarities.append({
                        'date': event_date,
                        'similarity': avg_similarity,
                        'conditions': event[list(current_conditions.keys())].to_dict()
                    })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Display most similar historical event
            if similarities:
                most_similar = similarities[0]
                similarity_percent = most_similar['similarity'] * 100
                
                if similarity_percent > 70:
                    alert_class = "alert-critical"
                    alert_icon = "🚨"
                    alert_type = "HIGH SIMILARITY"
                elif similarity_percent > 50:
                    alert_class = "alert-warning"
                    alert_icon = "⚠️"
                    alert_type = "MODERATE SIMILARITY"
                else:
                    alert_class = "alert-info"
                    alert_icon = "ℹ️"
                    alert_type = "LOW SIMILARITY"
                
                st.markdown(f"""
                <div class="{alert_class}">
                    <h3 style="margin: 0 0 10px 0;">{alert_icon} {alert_type} TO FLOOD EVENT</h3>
                    <p><strong>Most similar flood event:</strong> {most_similar['date']}</p>
                    <p><strong>Similarity score:</strong> {similarity_percent:.1f}%</p>
                    <p style="margin: 10px 0 0 0;"><em>Current conditions compared to historical flood event on {most_similar['date']}</em></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Current Conditions:**")
                    for param, value in current_conditions.items():
                        st.markdown(f"• {param.title()}: {value}")
                
                with col2:
                    st.markdown(f"**{most_similar['date']} Flood Event:**")
                    for param, value in most_similar['conditions'].items():
                        if not pd.isna(value):
                            st.markdown(f"• {param.title()}: {value}")
    
    # Predictive insights
    st.markdown("### 🔮 Predictive Insights")
    
    # Seasonal risk analysis
    historical_df['month'] = pd.to_datetime(historical_df['datetime']).dt.month
    monthly_flood_risk = historical_df.groupby('month')['flood_risk'].agg(['sum', 'count', 'mean']).reset_index()
    monthly_flood_risk['risk_percentage'] = (monthly_flood_risk['mean'] * 100).round(2)
    
    # Create monthly risk chart
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_flood_risk['month_name'] = monthly_flood_risk['month'].apply(lambda x: month_names[x-1])
    
    fig = px.bar(monthly_flood_risk, 
                 x='month_name', 
                 y='risk_percentage',
                 title='Monthly Flood Risk Percentage',
                 labels={'risk_percentage': 'Flood Risk (%)', 'month_name': 'Month'},
                 color='risk_percentage',
                 color_continuous_scale='Reds')
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#1e293b')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Current month risk
    current_month = datetime.now().month
    current_month_risk = monthly_flood_risk[monthly_flood_risk['month'] == current_month]
    
    if len(current_month_risk) > 0:
        risk_pct = current_month_risk.iloc[0]['risk_percentage']
        month_name = month_names[current_month - 1]
        
        if risk_pct > 5:
            risk_class = "alert-warning"
            risk_icon = "⚠️"
        elif risk_pct > 0:
            risk_class = "alert-info"
            risk_icon = "ℹ️"
        else:
            risk_class = "alert-info"
            risk_icon = "✅"
        
        st.markdown(f"""
        <div class="{risk_class}">
            <h4 style="margin: 0 0 10px 0;">{risk_icon} Current Month ({month_name}) Risk Assessment</h4>
            <p>Historical flood risk for {month_name}: <strong>{risk_pct:.1f}%</strong></p>
            <p>Based on {current_month_risk.iloc[0]['count']} historical records for this month.</p>
        </div>
        """, unsafe_allow_html=True)

def show_alert_history():
    """Display alert history and statistics"""
    st.markdown("## 📝 Alert History & Statistics")
    
    if not st.session_state.alert_history:
        st.info("No alerts have been generated yet. Visit the Home section to see real-time monitoring.")
        return
    
    # Alert statistics
    st.markdown("### 📊 Alert Statistics")
    
    total_alerts = len(st.session_state.alert_history)
    critical_alerts = len([a for a in st.session_state.alert_history if a['type'] == 'CRITICAL'])
    warning_alerts = len([a for a in st.session_state.alert_history if a['type'] == 'WARNING'])
    info_alerts = len([a for a in st.session_state.alert_history if a['type'] == 'INFO'])
    emails_sent = len([a for a in st.session_state.alert_history if a['email_sent']])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Alerts", total_alerts)
    with col2:
        st.metric("Critical", critical_alerts)
    with col3:
        st.metric("Warnings", warning_alerts)
    with col4:
        st.metric("Info", info_alerts)
    with col5:
        st.metric("Emails Sent", emails_sent)
    
    # Alert timeline
    st.markdown("### ⏰ Recent Alerts")
    
    # Display recent alerts with proper formatting
    for alert in st.session_state.alert_history[:20]:  # Show last 20 alerts
        timestamp = alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        email_status = "📧 Email sent" if alert['email_sent'] else "📭 No email"
        
        if alert['type'] == 'CRITICAL':
            alert_class = "alert-critical"
            icon = "🚨"
            color = "#dc2626"
        elif alert['type'] == 'WARNING':
            alert_class = "alert-warning"
            icon = "⚠️"
            color = "#f59e0b"
        else:
            alert_class = "alert-info"
            icon = "ℹ️"
            color = "#3b82f6"
        
        st.markdown(f"""
        <div class="{alert_class}" style="margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="color: {color}; margin: 0;">{icon} {alert['type']}: {alert['condition']}</h4>
                <small style="color: #64748b;">{timestamp}</small>
            </div>
            <p style="margin: 5px 0;">{alert['message']}</p>
            <small style="color: #64748b;">{email_status}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Alert management
    st.markdown("### 🛠️ Alert Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Alert History"):
            st.session_state.alert_history = []
            st.success("Alert history cleared!")
            st.rerun()
    
    with col2:
        if st.button("📧 Test Alert Email"):
            if st.session_state.email_config['enabled'] and st.session_state.email_config['recipient_emails']:
                # Create a test alert
                test_alerts = [{
                    'type': 'INFO',
                    'condition': 'System Test',
                    'message': 'This is a test alert to verify the email notification system.',
                    'value': 'Test',
                    'threshold': 'Test Threshold'
                }]
                
                test_weather = {
                    'temp': 25.0,
                    'humidity': 60.0,
                    'precip': 0.0,
                    'windspeed': 15.0,
                    'sealevelpressure': 1013.0
                }
                
                with st.spinner("Sending test alert email..."):
                    success, message = send_flood_alert_email(test_alerts, test_weather)
                    
                    if success:
                        st.success("✅ Test alert email sent successfully!")
                        # Log the test
                        log_alert("INFO", "System Test", "Test alert email sent successfully")
                    else:
                        st.error(f"❌ Test alert email failed: {message}")
            else:
                st.warning("Please configure email settings first in the Alert Configuration section.")
    
    # Export alert history
    if st.session_state.alert_history:
        st.markdown("### 💾 Export Alert Data")
        
        # Convert alert history to DataFrame
        alert_df = pd.DataFrame(st.session_state.alert_history)
        alert_df['timestamp'] = alert_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Display as table
        st.dataframe(alert_df, use_container_width=True)
        
        # Download button
        csv_data = alert_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Alert History (CSV)",
            data=csv_data,
            file_name=f"flood_alert_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()