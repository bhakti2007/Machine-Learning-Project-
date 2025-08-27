import pandas as pd
from src import train, evaluate, visualize, data_preprocessing

RAW_DATA_PATH = "data/raw/sales.csv"
PROCESSED_DATA_PATH = "data/processed/sales_processed.csv"

def main():
    # Load data
    df = pd.read_csv(RAW_DATA_PATH)

    # Preprocess
    df = data_preprocessing.preprocess_data(df)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    # Train
    model, X_test, y_test, y_pred = train.train_linear(df)

    # Evaluate
    evaluate.evaluate_model(y_test, y_pred)

    # Visualize
    visualize.run_visualizations(df)

if __name__ == "__main__":
    main()
