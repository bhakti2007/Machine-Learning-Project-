def create_features(df):
    if "Date" in df.columns:
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["Weekday"] = df["Date"].dt.weekday

# src/feature_engineering.py
def create_features(df, lags=[1,7,30]):
    for lag in lags:
        df[f"lag_{lag}"] = df["sales"].shift(lag)
    df["rolling_mean_7"] = df["sales"].shift(1).rolling(7).mean()
    df = df.dropna()
    return df

