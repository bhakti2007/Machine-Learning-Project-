import joblib
import pandas as pd
import os

MODEL_PATH = "models/linear_sales_forecast.pkl"
INPUT_PATH = "data/new_sales.csv"
OUTPUT_PATH = "data/predictions.csv"

def predict():
    # Load model
    model = joblib.load(MODEL_PATH)

    # Load new data
    df_new = pd.read_csv(INPUT_PATH)

    # Keep a copy of date if available
    date_col = None
    for col in df_new.columns:
        if col.lower() == "date":
            date_col = col
            break

    # Drop target column if it exists (for new unseen data, sales may not exist)
    target_col = None
    for col in df_new.columns:
        if col.lower() == "sales":
            target_col = col
            break

    X_new = df_new.drop(columns=[target_col]) if target_col else df_new

    # Predictions
    y_pred = model.predict(X_new)

    # Build output DataFrame
    output = pd.DataFrame({
        "y_pred": y_pred
    })

    # Add date if present
    if date_col:
        output["date"] = df_new[date_col]

    # Add y_true if present (for evaluation)
    if target_col:
        output["y_true"] = df_new[target_col]

    # Save predictions
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    predict()
