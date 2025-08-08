
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix

# 🔧 Set Streamlit page config (MUST be first Streamlit command)
st.set_page_config(page_title="Flood Risk Web App", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Maribyrnong_Cleaned_Weather_FloodRisk.csv")

df = load_data()

# Sidebar
st.sidebar.title("📊 Weather Flood Risk Tool")
section = st.sidebar.selectbox("Select Section", [
    "🌤️ Weather Dashboard",
    "🔍 Data Exploration",
    "🧠 ML Models",
    "📈 Predictions",
    "ℹ️ About"
])

# Section 1: Weather Dashboard
if section == "🌤️ Weather Dashboard":
    sub = st.sidebar.selectbox("Dashboard Options", ["Today's Weather", "7-Day Forecast"])
    st.title("🌦️ Live Weather - Maribyrnong")

    API_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Maribyrnong/today?unitGroup=metric&key=2M3N82RD42E4CWHEKB53PWSVW&contentType=json"
    response = requests.get(API_URL)

    if response.status_code == 200:
        weather = response.json()["currentConditions"]
        st.metric("🌡️ Temperature", f"{weather['temp']} °C")
        st.metric("💧 Humidity", f"{weather['humidity']}%")
        st.metric("🌧️ Precipitation", f"{weather['precip']} mm")
        st.metric("💨 Wind Speed", f"{weather['windspeed']} km/h")
        st.metric("☀️ UV Index", f"{weather['uvindex']}")
        st.metric("☁️ Cloud Cover", f"{weather['cloudcover']}%")
    else:
        st.error("Failed to fetch weather data.")

# Section 2: Data Exploration
elif section == "🔍 Data Exploration":
    sub = st.sidebar.selectbox("Explore Options", ["Summary Statistics", "Correlation Matrix", "Visualisations"])
    st.title("📊 Data Exploration")

    if sub == "Summary Statistics":
        st.write(df.describe())

    elif sub == "Correlation Matrix":
        corr = df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=False, cmap='coolwarm')
        st.pyplot(plt)

    elif sub == "Visualisations":
        col = st.selectbox("Select column to visualise", df.columns)
        st.line_chart(df[col])

# Section 3: ML Models
elif section == "🧠 ML Models":
    model_option = st.sidebar.selectbox("Model Options", ["Logistic Regression", "K-Means Clustering"])
    st.title("🤖 Machine Learning Models")

    # Prepare data
    features = df.drop(columns=['datetime', 'preciptype', 'conditions', 'flood_risk'])
    target = df['flood_risk']

    if model_option == "Logistic Regression":
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        st.pyplot(plt)

    elif model_option == "K-Means Clustering":
        kmeans = KMeans(n_clusters=3, random_state=0)
        kmeans.fit(features)
        df['Cluster'] = kmeans.labels_

        st.subheader("Cluster Distribution")
        st.bar_chart(df['Cluster'].value_counts())

# Section 4: Predictions
elif section == "📈 Predictions":
    st.title("📤 Upload New Data for Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)
        new_features = new_data.drop(columns=['datetime', 'preciptype', 'conditions'], errors='ignore')

        model = LogisticRegression(max_iter=1000)
        model.fit(df.drop(columns=['datetime', 'preciptype', 'conditions', 'flood_risk']), df['flood_risk'])

        predictions = model.predict(new_features)
        new_data['flood_risk_prediction'] = predictions
        st.write(new_data)

        csv = new_data.to_csv(index=False).encode()
        st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

# Section 5: About
elif section == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    This Streamlit web app visualizes and predicts flood risks in Maribyrnong using historical weather data 
    and live updates from Visual Crossing API.

    ### Project Features
    - View today's weather and forecast
    - Explore historical trends and patterns
    - Predict flood risk with Logistic Regression
    - Cluster days by weather similarity using K-Means

    **Developed by Bahram Azami, Alex Blomberg, Joeben Buena, and Rhys Chisholm**
    """)
