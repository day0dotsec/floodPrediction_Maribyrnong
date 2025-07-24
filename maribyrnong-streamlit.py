import requests
import streamlit as st

data = requests.get("https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Maribyrnong/today?unitGroup=metric&key=2M3N82RD42E4CWHEKB53PWSVW&contentType=json").json()

st.write(data)