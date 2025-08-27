from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

METRICS_PATH = "reports/figures/metrics.txt"

def evaluate_model(y_true, y_pred):
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Print to console (for quick debugging)
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    # Save metrics to file
    with open(METRICS_PATH, "w") as f:
        f.write("Evaluation Metrics\n")
        f.write("=================\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R²: {r2:.4f}\n")

if __name__ == "__main__":
    import pandas as pd

    # Load predictions
    preds = pd.read_csv("data/predictions.csv")

    # y_true must exist in predictions file
    if "y_true" not in preds.columns or "y_pred" not in preds.columns:
        raise ValueError("data/predictions.csv must have columns: y_true, y_pred")

    # Evaluate
    evaluate_model(preds["y_true"], preds["y_pred"])
