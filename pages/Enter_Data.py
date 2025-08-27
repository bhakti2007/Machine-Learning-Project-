import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# Load trained model
model = joblib.load("models/linear_sales_forecast.pkl")

# Features UI (underscored names for input)
features_ui = ["TV", "Radio", "Newspaper", "Oven", "Fan", "AC",
               "Projector", "Heater", "Table_Fan", "Mobile_Phone",
               "Laptop", "Tab"]

# Map UI names -> Model feature names
rename_map = {"Mobile_Phone": "Mobile Phone", "Table_Fan": "Table Fan"}

st.title("Enter Features")

# Input fields
inputs = {}
for feat in features_ui:
    inputs[feat] = st.number_input(feat, min_value=0.0, step=0.1)

# Predict button
if st.button("Predict"):
    X_new = pd.DataFrame([inputs])
    X_new.rename(columns=rename_map, inplace=True)

    prediction = model.predict(X_new)[0]

    st.session_state["prediction"] = prediction
    st.session_state["inputs"] = inputs

    # Save history with timestamp
    history_file = "history.csv"
    new_entry = pd.DataFrame([{**inputs, "Predicted_Sales": prediction,
                               "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])
    new_entry.rename(columns=rename_map, inplace=True)

    if os.path.exists(history_file):
        old = pd.read_csv(history_file)
        updated = pd.concat([old, new_entry], ignore_index=True)
    else:
        updated = new_entry
    updated.to_csv(history_file, index=False)

    st.success(f"Predicted Sales: {prediction:.2f}")
    st.button("Go to Results", on_click=lambda: st.experimental_set_query_params(page="Results"))

# Bulk upload
st.subheader("Upload CSV/Excel")
file = st.file_uploader("Upload", type=["csv", "xlsx"])

if file:
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    st.write("Preview:", df.head())

    if st.button("Predict Uploaded Data"):
        df.rename(columns=rename_map, inplace=True)
        preds = model.predict(df[model.feature_names_in_])
        df["Predicted_Sales"] = preds
        df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.dataframe(df)

        # Save to history
        if os.path.exists("history.csv"):
            old = pd.read_csv("history.csv")
            updated = pd.concat([old, df], ignore_index=True)
        else:
            updated = df
        updated.to_csv("history.csv", index=False)

        st.session_state["prediction"] = preds.tolist()
        st.session_state["inputs"] = df.to_dict(orient="records")

        st.download_button("Download Predictions",
                           df.to_csv(index=False), "predictions.csv", "text/csv")
