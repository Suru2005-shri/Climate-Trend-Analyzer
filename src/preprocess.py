"""
preprocess.py
-------------
Loads the raw climate dataset, performs:
 - Missing value imputation (forward-fill + median fallback)
 - Outlier capping using IQR method
 - Feature engineering (rolling averages, seasonal encoding, anomaly flags)
 - Saves cleaned dataset to data/climate_data_clean.csv
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
def load_data(path="data/climate_data_raw.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"📂 Loaded raw data: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df

# ─────────────────────────────────────────────
def handle_missing(df):
    numeric_cols = ["temperature", "rainfall", "humidity", "wind_speed"]
    before = df[numeric_cols].isnull().sum().sum()

    # Forward-fill for short gaps (weather continuity)
    df[numeric_cols] = df[numeric_cols].fillna(method="ffill", limit=3)
    # Remaining NaNs → median of same month
    for col in numeric_cols:
        df[col] = df.groupby("month")[col].transform(
            lambda x: x.fillna(x.median())
        )

    after = df[numeric_cols].isnull().sum().sum()
    print(f"🔧 Missing values: {before} → {after}")
    return df

# ─────────────────────────────────────────────
def cap_outliers(df):
    """IQR-based capping — keeps extreme events but removes impossible values."""
    cols = ["temperature", "rainfall", "humidity", "wind_speed"]
    for col in cols:
        Q1  = df[col].quantile(0.01)
        Q99 = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=Q1, upper=Q99)
        print(f"  {col}: capped to [{Q1:.1f}, {Q99:.1f}]")
    return df

# ─────────────────────────────────────────────
def engineer_features(df):
    df = df.sort_values("date").reset_index(drop=True)

    # Rolling averages (7-day and 30-day)
    df["temp_7day_avg"]    = df["temperature"].rolling(7,  min_periods=1).mean().round(2)
    df["temp_30day_avg"]   = df["temperature"].rolling(30, min_periods=1).mean().round(2)
    df["rain_7day_sum"]    = df["rainfall"].rolling(7,    min_periods=1).sum().round(2)
    df["rain_30day_sum"]   = df["rainfall"].rolling(30,   min_periods=1).sum().round(2)

    # Yearly average temperature per year (for trend line)
    yearly_avg = df.groupby("year")["temperature"].transform("mean").round(2)
    df["yearly_avg_temp"]  = yearly_avg

    # Month name for labeling
    df["month_name"] = pd.to_datetime(df["date"]).dt.strftime("%b")

    # Anomaly binary flag
    df["is_anomaly"] = (df["anomaly"] != "Normal").astype(int)

    # Seasonal encoding
    season_map = {"Winter":0, "Spring":1, "Monsoon":2, "Autumn":3}
    df["season_code"] = df["season"].map(season_map).fillna(0).astype(int)

    # Heat Index (simplified Steadman formula)
    T = df["temperature"]
    H = df["humidity"]
    df["heat_index"] = (-8.78469475556
                        + 1.61139411 * T
                        + 2.33854883889 * H
                        - 0.14611605 * T * H
                        - 0.012308094 * T**2
                        - 0.016424828 * H**2
                        + 0.002211732 * T**2 * H
                        + 0.00072546 * T * H**2
                        - 0.000003582 * T**2 * H**2).round(2)

    print("✅ Feature engineering complete. New columns:", 
          ["temp_7day_avg","temp_30day_avg","rain_7day_sum","rain_30day_sum",
           "yearly_avg_temp","month_name","is_anomaly","season_code","heat_index"])
    return df

# ─────────────────────────────────────────────
def save_clean(df, path="data/climate_data_clean.csv"):
    os.makedirs("data", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 Cleaned data saved → {path}")

# ─────────────────────────────────────────────
def run_pipeline():
    df = load_data()
    print("\n── Missing Value Check ──")
    print(df[["temperature","rainfall","humidity","wind_speed"]].isnull().sum())

    df = handle_missing(df)
    print("\n── Outlier Capping ──")
    df = cap_outliers(df)
    print("\n── Feature Engineering ──")
    df = engineer_features(df)
    save_clean(df)

    print("\n📊 Final Dataset Info:")
    print(df.shape)
    print(df.dtypes)
    print(df.tail(3))
    return df

if __name__ == "__main__":
    run_pipeline()
