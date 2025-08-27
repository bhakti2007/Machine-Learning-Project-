import pandas as pd

print("=== RAW DATA ===")
df = pd.read_csv("data/raw/sales.csv")
print(df.columns.tolist())
print(df.head())

print("\n=== PROCESSED DATA ===")
df_processed = pd.read_csv("data/processed/sales_processed.csv")
print(df_processed.columns.tolist())
print(df_processed.head())
