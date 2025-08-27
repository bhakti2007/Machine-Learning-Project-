import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load your dataset (update path if needed)
df = pd.read_csv("data/raw/sales.csv")

# Define X (features) and y (target)
X = df.drop(columns=["Sales"])
y = df["Sales"]

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Save model and features
model_data = {
    "model": model,
    "features": list(X.columns)
}
joblib.dump(model_data, "models/sales_forecast.pkl")

print("✅ Linear Regression model trained and saved successfully!")
