import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.title("Prediction History")

if os.path.exists("history.csv"):
    history = pd.read_csv("history.csv")
    st.dataframe(history)

    if "Timestamp" in history.columns:
        timestamps = history["Timestamp"].unique()
        choice = st.selectbox("Select record", timestamps)

        if choice:
            # Get the row safely as a DataFrame
            record = history[history["Timestamp"] == choice].head(1)

            st.write("Record Details:")
            st.dataframe(record)

            # Drop non-numeric columns
            features_only = record.drop(columns=["Predicted_Sales", "Timestamp"], errors="ignore")
            features_only = features_only.apply(pd.to_numeric, errors="coerce")

            if not features_only.empty and features_only.shape[1] > 0:
                st.subheader("Feature Values")
                fig, ax = plt.subplots()
                features_only.T.plot(kind="bar", ax=ax, color="pink", legend=False)
                ax.set_ylabel("Value")
                st.pyplot(fig)

            st.success(f"Predicted Sales: {record['Predicted_Sales'].values[0]:.2f}")
else:
    st.warning("No history found. Make predictions first.")
