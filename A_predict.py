import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/sales_forecast_model.pkl")

# Load new input data
new_data = pd.read_csv("data/new_sales.csv")

# ✅ Rename columns if needed (must match training features)
new_data = new_data.rename(columns={
    "Mobile Phone": "Mobile_Phone",
    "Table Fan": "Table_Fan"
})

# Make predictions
predictions = model.predict(new_data)

# Save results
new_data["Predicted_Sales"] = predictions
new_data.to_csv("data/predictions.csv", index=False)

print("✅ Predictions saved to data/predictions.csv")
