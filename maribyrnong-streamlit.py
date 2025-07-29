import requests
import streamlit as st

# Set page title and favicon
st.set_page_config(page_title="Maribyrnong City Council: Flood and Overflow Prediction", page_icon="🌧️")

st.sidebar.markdown("# **Maribyrnong City Council: Flood and Overflow Prediction**")
# Radio button for selecting section
selected_section = st.sidebar.radio("Select Section", ["Historical Data Overview",
                                                       "Data Preparation",
                                                       "Modeling",
                                                       "Test"])

data = requests.get("https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Maribyrnong/today?unitGroup=metric&key=2M3N82RD42E4CWHEKB53PWSVW&contentType=json").json()

st.write(data)