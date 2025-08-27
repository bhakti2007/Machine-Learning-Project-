import pandas as pd

def clean_data(df):
 def load_and_preprocess(path: str, freq="D"):
    # Remove missing values
    df = df.dropna()

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    df = df.asfreq(freq)   # enforce regular frequency
    df["sales"] = df["sales"].fillna(0)  # or interpolation

    # Ensure correct datetime format
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing:
    - Fill missing values with 0
    - Remove duplicates
    - Ensure correct datatypes
    """
    # Drop duplicates
    df = df.drop_duplicates()

    # Fill missing values (numeric with 0, categorical with mode)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(0)

    # Convert column names to consistent format (no spaces)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    return df
