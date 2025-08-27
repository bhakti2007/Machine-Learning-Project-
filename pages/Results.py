import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Prediction Results")

if "prediction" in st.session_state:
    preds = st.session_state["prediction"]

    if isinstance(preds, (float, int)):
        st.success(f"Predicted Sales: {preds:.2f}")

        if "inputs" in st.session_state:
            inputs = st.session_state["inputs"]

            st.subheader("Input Features")
            fig, ax = plt.subplots()
            pd.Series(inputs).astype(float).plot(kind="bar", ax=ax, color="pink")
            st.pyplot(fig)

    elif isinstance(preds, list):
        st.success("Bulk Predictions Done")

        if "inputs" in st.session_state:
            df = pd.DataFrame(st.session_state["inputs"])
            df["Predicted_Sales"] = preds
            st.dataframe(df)
else:
    st.info("Go to 'Enter Data' page to make predictions first.")
