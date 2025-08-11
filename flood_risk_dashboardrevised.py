import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# ------------------------
# MUST be first Streamlit command
# ------------------------
st.set_page_config(page_title="Flood Risk Web App", layout="wide")

# ------------------------
# Config
# ------------------------
API_KEY = "2M3N82RD42E4CWHEKB53PWSVW"
LOCATION = "Maribyrnong"
API_URL = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{LOCATION}?unitGroup=metric&key={API_KEY}&contentType=json"

# ------------------------
# Helper Functions
# ------------------------
def get_weather_data():
    """Fetch weather data from Visual Crossing API."""
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

def calculate_flood_risk(rain, temp, wind=None):
    """
    Determine flood risk level.
    Example thresholds (tune for your needs):
    - High: rain > 30mm OR temp < 5°C
    - Medium: rain 10–30mm
    - Low: rain < 10mm
    """
    if rain > 30 or (temp is not None and temp < 5):
        return "High"
    elif rain >= 10:
        return "Medium"
    else:
        return "Low"

def display_weather_card(title, date_str, risk_level, attributes):
    """Display weather and flood risk info in a card-like format."""
    with st.container():
        st.subheader(title)
        st.write(f"📅 {date_str}")
        st.markdown(f"**Flood Risk:** {risk_level}")
        st.write(attributes)

# ------------------------
# Sidebar Navigation
# ------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Forecast", "About"])

# ------------------------
# Fetch Data
# ------------------------
data = get_weather_data()

if data:
    days_data = data.get("days", [])
    today_data = days_data[0] if days_data else None

    if page == "Dashboard":
        st.title("🌊 Flood Risk Dashboard - Maribyrnong")

        if today_data:
            today_date = datetime.strptime(today_data["datetime"], "%Y-%m-%d")
            today_day = today_date.strftime("%A")
            rain = today_data.get("precip", 0)
            temp = today_data.get("temp", None)
            wind = today_data.get("windspeed", None)
            risk = calculate_flood_risk(rain, temp, wind)

            display_weather_card(
                "Today's Weather",
                f"{today_day}, {today_date.strftime('%d %B %Y')}",
                risk,
                {
                    "Rainfall (mm)": rain,
                    "Temperature (°C)": temp,
                    "Wind Speed (km/h)": wind
                }
            )

            # Hourly forecast
            st.markdown("### Hourly Forecast")
            hourly_data = today_data.get("hours", [])
            if hourly_data:
                hourly_df = pd.DataFrame(hourly_data)[["datetime", "temp", "precip", "windspeed"]]
                st.dataframe(hourly_df)
            else:
                st.info("No hourly forecast available.")

    elif page == "Forecast":
        st.title("📅 Upcoming Flood Risk Forecast")

        if days_data:
            forecast_list = []
            for day in days_data[1:6]:  # Next 5 days
                date_obj = datetime.strptime(day["datetime"], "%Y-%m-%d")
                day_name = date_obj.strftime("%A")
                rain = day.get("precip", 0)
                temp = day.get("temp", None)
                wind = day.get("windspeed", None)
                risk = calculate_flood_risk(rain, temp, wind)

                forecast_list.append({
                    "Date": f"{day_name}, {date_obj.strftime('%d %B %Y')}",
                    "Rainfall (mm)": rain,
                    "Temp (°C)": temp,
                    "Wind (km/h)": wind,
                    "Flood Risk": risk
                })

            forecast_df = pd.DataFrame(forecast_list)
            st.dataframe(forecast_df)
        else:
            st.info("No forecast data available.")

    elif page == "About":
        st.title("ℹ️ About this App")
        st.write("""
        This app retrieves live weather data for Maribyrnong from the Visual Crossing API
        and uses simple thresholds to assess potential flood risk for today and the upcoming days.
        
        **Risk levels:**
        - **Low:** Rain < 10 mm  
        - **Medium:** Rain 10–30 mm  
        - **High:** Rain > 30 mm or very low temperatures (< 5°C)  
        """)

else:
    st.error("No weather data available.")
