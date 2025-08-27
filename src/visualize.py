import pandas as pd
import matplotlib.pyplot as plt
import os

PRED_PATH = "data/predictions.csv"
FIG_DIR = "reports/figures"

def visualize_predictions():
    os.makedirs(FIG_DIR, exist_ok=True)

    preds = pd.read_csv(PRED_PATH)

    # -------- Scatter: Actual vs Predicted --------
    plt.figure(figsize=(8,6))
    plt.scatter(preds["y_true"], preds["y_pred"], alpha=0.6)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.plot([preds["y_true"].min(), preds["y_true"].max()],
             [preds["y_true"].min(), preds["y_true"].max()], "r--")  # perfect prediction line
    plt.savefig(os.path.join(FIG_DIR, "actual_vs_pred.png"))
    plt.close()

    # -------- Residuals --------
    residuals = preds["y_true"] - preds["y_pred"]
    plt.figure(figsize=(8,6))
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel("Residuals (y_true - y_pred)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.savefig(os.path.join(FIG_DIR, "residuals.png"))
    plt.close()

    # -------- Time-Series (if date exists) --------
    if "date" in preds.columns:
        preds["date"] = pd.to_datetime(preds["date"])
        preds = preds.sort_values("date")

        plt.figure(figsize=(12,6))
        plt.plot(preds["date"], preds["y_true"], label="Actual Sales", marker="o")
        plt.plot(preds["date"], preds["y_pred"], label="Predicted Sales", marker="x")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.title("Sales Forecast Over Time")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "time_series_forecast.png"))
        plt.close()

    print(f"✅ Figures saved in {FIG_DIR}")

if __name__ == "__main__":
    visualize_predictions()
