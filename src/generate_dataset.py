"""
generate_dataset.py
-------------------
Generates a realistic synthetic climate dataset for 30 years (1994–2024).
Simulates daily temperature, rainfall, humidity, and wind speed with:
- Long-term warming trend (climate change simulation)
- Seasonal variation
- Random noise
- Injected anomalies (heatwaves, cold snaps, extreme rainfall)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

START_DATE = "1994-01-01"
END_DATE   = "2024-12-31"
CITY       = "Mumbai"         # Change as desired
LAT        = 19.076           # Latitude (affects seasonality)
WARMING_RATE = 0.04           # °C per year (IPCC-inspired value)

# ─────────────────────────────────────────────
# DATE RANGE
# ─────────────────────────────────────────────
dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
n     = len(dates)
years = dates.year
doy   = dates.dayofyear   # Day of year (1–365/366)

# ─────────────────────────────────────────────
# TEMPERATURE  (°C)
# Seasonal cycle + long-term warming + noise
# ─────────────────────────────────────────────
seasonal_temp   = 8 * np.sin(2 * np.pi * (doy - 80) / 365)   # seasonal swing
warming_signal  = WARMING_RATE * (years - years.min())        # gradual warming
base_temp       = 27.0                                         # Mumbai baseline
noise_temp      = np.random.normal(0, 1.5, n)
temperature     = base_temp + seasonal_temp + warming_signal + noise_temp

# ─────────────────────────────────────────────
# RAINFALL  (mm/day)
# Monsoon-heavy (June–September), near-zero otherwise
# ─────────────────────────────────────────────
monsoon_mask = (dates.month >= 6) & (dates.month <= 9)
rainfall = np.where(
    monsoon_mask,
    np.random.exponential(scale=12, size=n),   # monsoon season
    np.random.exponential(scale=1.2, size=n)   # dry season
)

# ─────────────────────────────────────────────
# HUMIDITY  (%)
# Higher during monsoon
# ─────────────────────────────────────────────
humidity = np.where(
    monsoon_mask,
    np.clip(np.random.normal(85, 8, n), 55, 100),
    np.clip(np.random.normal(60, 10, n), 30, 90)
)

# ─────────────────────────────────────────────
# WIND SPEED  (km/h)
# ─────────────────────────────────────────────
wind_speed = np.clip(np.random.gamma(shape=2.5, scale=6, size=n), 0, 80)

# ─────────────────────────────────────────────
# INJECT ANOMALIES
# ─────────────────────────────────────────────
anomaly_labels = ["Normal"] * n

# Heatwave anomalies (random ~15 events, 3–7 days each)
for _ in range(15):
    idx = np.random.randint(0, n - 7)
    dur = np.random.randint(3, 8)
    temperature[idx:idx+dur] += np.random.uniform(4, 8)
    for j in range(idx, min(idx+dur, n)):
        anomaly_labels[j] = "Heatwave"

# Cold snap anomalies (~8 events)
for _ in range(8):
    idx = np.random.randint(0, n - 5)
    dur = np.random.randint(3, 6)
    temperature[idx:idx+dur] -= np.random.uniform(4, 7)
    for j in range(idx, min(idx+dur, n)):
        anomaly_labels[j] = "Cold Snap"

# Extreme rainfall events (~20 days)
extreme_rain_idx = np.random.choice(np.where(monsoon_mask)[0], 20, replace=False)
rainfall[extreme_rain_idx] += np.random.uniform(80, 200, 20)
for j in extreme_rain_idx:
    anomaly_labels[j] = "Extreme Rainfall"

# ─────────────────────────────────────────────
# BUILD DATAFRAME
# ─────────────────────────────────────────────
df = pd.DataFrame({
    "date"        : dates,
    "year"        : dates.year,
    "month"       : dates.month,
    "day"         : dates.day,
    "season"      : pd.cut(
                        dates.month,
                        bins=[0,2,5,8,11,12],
                        labels=["Winter","Spring","Monsoon","Autumn","Winter2"]
                    ).astype(str).replace("Winter2","Winter"),
    "temperature" : np.round(temperature, 2),
    "rainfall"    : np.round(np.clip(rainfall, 0, None), 2),
    "humidity"    : np.round(humidity, 1),
    "wind_speed"  : np.round(wind_speed, 1),
    "anomaly"     : anomaly_labels,
    "city"        : CITY,
})

# ─────────────────────────────────────────────
# INTRODUCE REALISTIC MISSING VALUES (~1%)
# ─────────────────────────────────────────────
for col in ["temperature", "rainfall", "humidity", "wind_speed"]:
    missing_idx = np.random.choice(n, size=int(0.01 * n), replace=False)
    df.loc[missing_idx, col] = np.nan

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv("data/climate_data_raw.csv", index=False)
print(f"✅ Dataset generated: {len(df):,} rows | {df.columns.tolist()}")
print(df.head())
print(df.describe())
