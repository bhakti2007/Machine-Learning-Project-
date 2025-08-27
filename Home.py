import streamlit as st
import pandas as pd
import os

st.title("Sales Forecasting App")

st.write("This app predicts sales based on advertising and product features. "
         "Use the sidebar to navigate between pages.")

# Sidebar summary if history exists
if os.path.exists("history.csv"):
    hist = pd.read_csv("history.csv")

    st.sidebar.subheader("Dashboard Summary")
    st.sidebar.metric("Total Predictions", len(hist))
    st.sidebar.metric("Avg Sales", round(hist["Predicted_Sales"].mean(), 2))
    st.sidebar.metric("Last Prediction", round(hist["Predicted_Sales"].iloc[-1], 2))

# Background style
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #ffe6e6, #ffffff);
    }
    </style>
""", unsafe_allow_html=True)
