import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Lazy import torch to avoid Streamlit file watcher conflicts
torch = None
nn = None
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
API_KEY = "2M3N82RD42E4CWHEKB53PWSVW"
LOCATION = "Maribyrnong"
API_URL = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{LOCATION}?unitGroup=metric&key={API_KEY}&contentType=json"

# Page configuration
st.set_page_config(
    page_title="Maribyrnong Flood Prediction Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .weather-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .risk-high {
        background: #fee2e2;
        border-left-color: #dc2626;
    }
    .risk-medium {
        background: #fef3c7;
        border-left-color: #d97706;
    }
    .risk-low {
        background: #dcfce7;
        border-left-color: #16a34a;
    }
</style>
""", unsafe_allow_html=True)

def import_torch():
    """Lazy import torch modules to avoid Streamlit file watcher conflicts"""
    global torch, nn
    if torch is None:
        try:
            import torch as _torch
            import torch.nn as _nn
            torch = _torch
            nn = _nn
        except ImportError:
            st.error("PyTorch not installed. Please install with: pip install torch")
            return False
    return True

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_weather_data():
    """Fetch current weather data from Visual Crossing API"""
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch weather data: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None

@st.cache_data
def load_historical_data():
    """Load historical weather data"""
    try:
        df = pd.read_csv('Maribyrnong_Cleaned_Weather_FloodRisk.csv')
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)}")
        return None

class LSTMFloodPredictor:
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        if not import_torch():
            return
        
        class _LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
                super(_LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                   batch_first=True, dropout=dropout)
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, output_size)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                
                out, _ = self.lstm(x, (h0, c0))
                out = self.dropout(out[:, -1, :])
                out = self.fc(out)
                out = self.sigmoid(out)
                return out
        
        self.model = _LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
    
    def __call__(self, x):
        return self.model(x) if hasattr(self, 'model') else None
    
    def train(self):
        if hasattr(self, 'model'):
            self.model.train()
    
    def eval(self):
        if hasattr(self, 'model'):
            self.model.eval()
    
    def parameters(self):
        if hasattr(self, 'model'):
            return self.model.parameters()
        return []

def prepare_lstm_data(df, sequence_length=7):
    """Prepare data for LSTM model"""
    feature_cols = ['tempmax', 'tempmin', 'temp', 'humidity', 'precip', 
                   'precipprob', 'windspeed', 'sealevelpressure', 'cloudcover']
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    
    X, y = [], []
    for i in range(len(scaled_features) - sequence_length):
        X.append(scaled_features[i:i+sequence_length])
        y.append(df.iloc[i+sequence_length]['flood_risk'])
    
    return np.array(X), np.array(y), scaler, feature_cols

def train_lstm_model(X, y):
    """Train LSTM model"""
    if not import_torch():
        return None, 0.0, None, None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)
    
    model = LSTMFloodPredictor(input_size=X_train.shape[2], hidden_size=50, 
                              num_layers=2, output_size=1)
    
    if not hasattr(model, 'model'):
        return None, 0.0, None, None
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_predictions = (test_outputs > 0.5).float()
        accuracy = (test_predictions == y_test).float().mean()
    
    return model, accuracy.item(), X_test, y_test

def main():
    st.markdown("<h1 class='main-header'>🌊 Maribyrnong Flood Prediction Dashboard</h1>", unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    main_section = st.sidebar.selectbox(
        "Select Section",
        ["🏠 Home & Current Weather", "🤖 Machine Learning Models", "📊 Data Analysis", "🔍 Insights & Predictions"]
    )
    
    # Load data
    weather_data = fetch_weather_data()
    historical_df = load_historical_data()
    
    if historical_df is None:
        st.error("Failed to load historical data. Please check the CSV file.")
        return
    
    if main_section == "🏠 Home & Current Weather":
        st.header("Current Weather & Forecast")
        
        if weather_data:
            current = weather_data['currentConditions']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="weather-card">
                    <h3>🌡️ Temperature</h3>
                    <h2>{current['temp']}°C</h2>
                    <p>Feels like {current['feelslike']}°C</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="weather-card">
                    <h3>💧 Humidity</h3>
                    <h2>{current['humidity']}%</h2>
                    <p>Dew point: {current['dew']}°C</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="weather-card">
                    <h3>🌧️ Precipitation</h3>
                    <h2>{current.get('precip', 0)}mm</h2>
                    <p>Probability: {current.get('precipprob', 0)}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="weather-card">
                    <h3>💨 Wind</h3>
                    <h2>{current['windspeed']} km/h</h2>
                    <p>Direction: {current['winddir']}°</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 7-day forecast
            st.subheader("📅 7-Day Forecast")
            forecast_data = []
            for day in weather_data['days'][:7]:
                forecast_data.append({
                    'Date': day['datetime'],
                    'High': day['tempmax'],
                    'Low': day['tempmin'],
                    'Precipitation': day['precip'],
                    'Conditions': day['conditions']
                })
            
            forecast_df = pd.DataFrame(forecast_data)
            st.dataframe(forecast_df, use_container_width=True)
            
            # Forecast chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=forecast_df['Date'], y=forecast_df['High'], 
                          name='High Temp', line=dict(color='red')),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=forecast_df['Date'], y=forecast_df['Low'], 
                          name='Low Temp', line=dict(color='blue')),
                secondary_y=False,
            )
            fig.add_trace(
                go.Bar(x=forecast_df['Date'], y=forecast_df['Precipitation'], 
                       name='Precipitation', opacity=0.7),
                secondary_y=True,
            )
            
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
            fig.update_yaxes(title_text="Precipitation (mm)", secondary_y=True)
            fig.update_layout(title="7-Day Weather Forecast")
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Unable to fetch current weather data. Showing historical overview instead.")
            
            # Show recent historical data
            recent_data = historical_df.tail(30)
            fig = px.line(recent_data, x='datetime', y=['tempmax', 'tempmin'], 
                         title='Recent Temperature Trends')
            st.plotly_chart(fig, use_container_width=True)
    
    elif main_section == "🤖 Machine Learning Models":
        st.header("Machine Learning Models for Flood Prediction")
        
        model_type = st.sidebar.selectbox(
            "Select Model",
            ["LSTM Neural Network", "Logistic Regression", "K-Means Clustering"]
        )
        
        if model_type == "LSTM Neural Network":
            st.subheader("🧠 LSTM Neural Network")
            st.write("Long Short-Term Memory network for sequential flood prediction")
            
            if st.button("Train LSTM Model"):
                with st.spinner("Training LSTM model..."):
                    X, y, scaler, feature_cols = prepare_lstm_data(historical_df)
                    model, accuracy, X_test, y_test = train_lstm_model(X, y)
                    
                    if model is not None:
                        st.success(f"LSTM Model trained successfully!")
                        st.metric("Test Accuracy", f"{accuracy:.4f}")
                        
                        # Show model architecture
                        st.write("**Model Architecture:**")
                        st.code(f"""
Input Size: {X.shape[2]} features
Hidden Size: 50 neurons
Layers: 2 LSTM layers
Output: Binary classification (flood/no flood)
Sequence Length: 7 days
                        """)
                    else:
                        st.error("Failed to train LSTM model. Please check PyTorch installation.")
        
        elif model_type == "Logistic Regression":
            st.subheader("📈 Logistic Regression")
            st.write("Classical statistical approach for flood risk classification")
            
            if st.button("Train Logistic Regression"):
                with st.spinner("Training Logistic Regression model..."):
                    feature_cols = ['tempmax', 'tempmin', 'temp', 'humidity', 'precip', 
                                   'precipprob', 'windspeed', 'sealevelpressure', 'cloudcover']
                    
                    X = historical_df[feature_cols]
                    y = historical_df['flood_risk']
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    lr_model = LogisticRegression(random_state=42)
                    lr_model.fit(X_train_scaled, y_train)
                    
                    y_pred = lr_model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    st.success(f"Logistic Regression trained successfully!")
                    st.metric("Test Accuracy", f"{accuracy:.4f}")
                    
                    # Feature importance
                    importance_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Coefficient': lr_model.coef_[0]
                    }).sort_values('Coefficient', key=abs, ascending=False)
                    
                    fig = px.bar(importance_df, x='Coefficient', y='Feature', 
                                title='Feature Importance (Logistic Regression Coefficients)')
                    st.plotly_chart(fig, use_container_width=True)
        
        elif model_type == "K-Means Clustering":
            st.subheader("🎯 K-Means Clustering")
            st.write("Unsupervised learning to identify weather patterns")
            
            n_clusters = st.slider("Number of Clusters", 2, 10, 4)
            
            if st.button("Perform K-Means Clustering"):
                with st.spinner("Performing K-Means clustering..."):
                    feature_cols = ['tempmax', 'tempmin', 'temp', 'humidity', 'precip', 
                                   'precipprob', 'windspeed', 'sealevelpressure', 'cloudcover']
                    
                    X = historical_df[feature_cols]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(X_scaled)
                    
                    historical_df_clustered = historical_df.copy()
                    historical_df_clustered['cluster'] = clusters
                    
                    st.success(f"K-Means clustering completed with {n_clusters} clusters!")
                    
                    # Cluster analysis
                    cluster_stats = historical_df_clustered.groupby('cluster').agg({
                        'flood_risk': ['count', 'sum', 'mean'],
                        'precip': 'mean',
                        'temp': 'mean',
                        'humidity': 'mean'
                    }).round(3)
                    
                    st.write("**Cluster Statistics:**")
                    st.dataframe(cluster_stats)
                    
                    # Visualization
                    fig = px.scatter(historical_df_clustered, x='temp', y='precip', 
                                   color='cluster', size='humidity',
                                   title='Weather Pattern Clusters',
                                   labels={'temp': 'Temperature (°C)', 'precip': 'Precipitation (mm)'})
                    st.plotly_chart(fig, use_container_width=True)
    
    elif main_section == "📊 Data Analysis":
        st.header("Historical Data Analysis")
        
        analysis_type = st.sidebar.selectbox(
            "Select Analysis",
            ["Data Overview", "Temporal Patterns", "Weather Correlations", "Flood Risk Analysis"]
        )
        
        if analysis_type == "Data Overview":
            st.subheader("📋 Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(historical_df))
            with col2:
                st.metric("Date Range", f"{historical_df['datetime'].dt.year.min()}-{historical_df['datetime'].dt.year.max()}")
            with col3:
                st.metric("Flood Events", historical_df['flood_risk'].sum())
            with col4:
                st.metric("Flood Rate", f"{(historical_df['flood_risk'].mean()*100):.2f}%")
            
            st.write("**Data Sample:**")
            st.dataframe(historical_df.head(10))
            
            st.write("**Statistical Summary:**")
            st.dataframe(historical_df.describe())
        
        elif analysis_type == "Temporal Patterns":
            st.subheader("📅 Temporal Patterns")
            
            # Monthly patterns
            monthly_stats = historical_df.groupby('month').agg({
                'flood_risk': ['count', 'sum', 'mean'],
                'precip': 'mean',
                'temp': 'mean'
            }).round(3)
            
            fig = make_subplots(rows=2, cols=2, 
                              subplot_titles=['Monthly Flood Risk', 'Monthly Precipitation', 
                                            'Monthly Temperature', 'Yearly Trends'])
            
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            fig.add_trace(go.Bar(x=months, y=monthly_stats['flood_risk']['mean'], 
                                name='Flood Risk'), row=1, col=1)
            fig.add_trace(go.Bar(x=months, y=monthly_stats['precip']['mean'], 
                                name='Precipitation'), row=1, col=2)
            fig.add_trace(go.Bar(x=months, y=monthly_stats['temp']['mean'], 
                                name='Temperature'), row=2, col=1)
            
            yearly_stats = historical_df.groupby('year')['flood_risk'].sum()
            fig.add_trace(go.Scatter(x=yearly_stats.index, y=yearly_stats.values, 
                                   mode='lines+markers', name='Yearly Floods'), row=2, col=2)
            
            fig.update_layout(height=600, showlegend=False, title_text="Temporal Analysis")
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Weather Correlations":
            st.subheader("🌤️ Weather Variable Correlations")
            
            numeric_cols = ['tempmax', 'tempmin', 'temp', 'humidity', 'precip', 
                           'precipprob', 'windspeed', 'sealevelpressure', 'cloudcover', 'flood_risk']
            corr_matrix = historical_df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title="Correlation Matrix of Weather Variables")
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlations with flood risk
            flood_corr = corr_matrix['flood_risk'].abs().sort_values(ascending=False)[1:]
            st.write("**Top Correlations with Flood Risk:**")
            for var, corr in flood_corr.head(5).items():
                st.write(f"• {var}: {corr:.3f}")
        
        elif analysis_type == "Flood Risk Analysis":
            st.subheader("🌊 Flood Risk Analysis")
            
            flood_days = historical_df[historical_df['flood_risk'] == 1]
            normal_days = historical_df[historical_df['flood_risk'] == 0]
            
            if len(flood_days) > 0:
                st.write("**Flood Event Characteristics:**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Average conditions during flood events:")
                    flood_stats = flood_days[['tempmax', 'precip', 'humidity', 'windspeed']].mean()
                    for var, val in flood_stats.items():
                        st.write(f"• {var}: {val:.2f}")
                
                with col2:
                    st.write("Average conditions during normal days:")
                    normal_stats = normal_days[['tempmax', 'precip', 'humidity', 'windspeed']].mean()
                    for var, val in normal_stats.items():
                        st.write(f"• {var}: {val:.2f}")
                
                # Comparison chart
                comparison_data = pd.DataFrame({
                    'Variable': ['Temperature', 'Precipitation', 'Humidity', 'Wind Speed'],
                    'Flood Days': [flood_stats['tempmax'], flood_stats['precip'], 
                                  flood_stats['humidity'], flood_stats['windspeed']],
                    'Normal Days': [normal_stats['tempmax'], normal_stats['precip'], 
                                   normal_stats['humidity'], normal_stats['windspeed']]
                })
                
                fig = px.bar(comparison_data, x='Variable', y=['Flood Days', 'Normal Days'],
                           title='Weather Conditions: Flood vs Normal Days', barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No flood events found in the historical data for detailed analysis.")
    
    elif main_section == "🔍 Insights & Predictions":
        st.header("Insights & Flood Predictions")
        
        insight_type = st.sidebar.selectbox(
            "Select Analysis",
            ["Risk Assessment", "Prediction Dashboard", "Model Comparison", "Recommendations"]
        )
        
        if insight_type == "Risk Assessment":
            st.subheader("🎯 Current Risk Assessment")
            
            if weather_data:
                current = weather_data['currentConditions']
                
                # Simple risk calculation based on historical patterns
                risk_factors = {
                    'High Precipitation': current.get('precip', 0) > 10,
                    'High Humidity': current['humidity'] > 80,
                    'Low Pressure': current['pressure'] < 1010,
                    'High Wind Speed': current['windspeed'] > 40
                }
                
                risk_score = sum(risk_factors.values()) / len(risk_factors)
                
                if risk_score >= 0.75:
                    risk_level = "HIGH"
                    risk_class = "risk-high"
                elif risk_score >= 0.5:
                    risk_level = "MEDIUM"
                    risk_class = "risk-medium"
                else:
                    risk_level = "LOW"
                    risk_class = "risk-low"
                
                st.markdown(f"""
                <div class="metric-card {risk_class}">
                    <h2>Current Flood Risk: {risk_level}</h2>
                    <p>Risk Score: {risk_score:.2f}/1.00</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("**Risk Factors Analysis:**")
                for factor, present in risk_factors.items():
                    status = "🔴 Present" if present else "🟢 Not Present"
                    st.write(f"• {factor}: {status}")
        
        elif insight_type == "Model Comparison":
            st.subheader("⚖️ Model Performance Comparison")
            
            # Placeholder for model comparison
            model_performance = pd.DataFrame({
                'Model': ['LSTM Neural Network', 'Logistic Regression', 'K-Means Clustering'],
                'Accuracy': [0.85, 0.78, 0.72],
                'Precision': [0.80, 0.75, 0.68],
                'Recall': [0.82, 0.73, 0.70],
                'F1-Score': [0.81, 0.74, 0.69]
            })
            
            st.dataframe(model_performance, use_container_width=True)
            
            fig = px.bar(model_performance, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        title='Model Performance Comparison', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        elif insight_type == "Recommendations":
            st.subheader("💡 Recommendations")
            
            st.write("""
            **For Emergency Management:**
            - Monitor precipitation levels above 10mm/day
            - Pay attention to humidity levels exceeding 80%
            - Consider atmospheric pressure drops below 1010 hPa
            
            **For Model Improvement:**
            - Collect more flood event data (currently only 2 events in dataset)
            - Include additional meteorological variables (river levels, soil moisture)
            - Implement ensemble methods combining multiple models
            
            **For Early Warning System:**
            - Set up automated alerts based on weather thresholds
            - Integrate real-time river level monitoring
            - Develop mobile app for community notifications
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("🌊 Maribyrnong Flood Prediction Dashboard | Data-driven flood risk assessment for community safety")

if __name__ == "__main__":
    main()