"""
forecasting.py
--------------
Forecasts future climate trends using:
 1. Linear Regression on yearly averages (simple, explainable)
 2. ARIMA / SARIMAX on monthly averages (time-series model)
Saves forecast plots and prediction tables.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings
warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("⚠️  statsmodels not installed — using Linear Regression only")

OUT = "outputs/figures"
os.makedirs(OUT, exist_ok=True)
os.makedirs("reports", exist_ok=True)

PALETTE = {
    "bg"     : "#0D1117",
    "panel"  : "#161B22",
    "text"   : "#E6EDF3",
    "grid"   : "#21262D",
    "hist"   : "#3A7BD5",
    "pred"   : "#E84040",
    "ci"     : "#FF8C00",
    "trend"  : "#00B09B",
}

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5, linestyle="--")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)

# ─────────────────────────────────────────────
# 1. LINEAR REGRESSION FORECAST (Yearly)
# ─────────────────────────────────────────────
def linear_forecast(df, forecast_years=10):
    yearly = df.groupby("year")["temperature"].mean().reset_index()
    X = yearly["year"].values.reshape(-1, 1)
    y = yearly["temperature"].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred_train = model.predict(X)

    # Future years
    future_years = np.arange(2025, 2025 + forecast_years).reshape(-1, 1)
    future_pred  = model.predict(future_years)

    # 95% Prediction Interval (approximate)
    n    = len(y)
    se   = np.sqrt(np.sum((y - y_pred_train)**2) / (n - 2))
    t_ci = 1.96
    margin = t_ci * se * np.sqrt(1 + 1/n + (future_years.flatten() - X.flatten().mean())**2
                                  / np.sum((X.flatten() - X.flatten().mean())**2))

    # Metrics
    mae  = mean_absolute_error(y, y_pred_train)
    rmse = np.sqrt(mean_squared_error(y, y_pred_train))
    r2   = model.score(X, y)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax.fill_between(future_years.flatten(), future_pred - margin, future_pred + margin,
                    color=PALETTE["ci"], alpha=0.15, label="95% Prediction Interval")
    ax.plot(yearly["year"], y, color=PALETTE["hist"], linewidth=2,
            marker="o", markersize=5, label="Historical (1994–2024)")
    ax.plot(yearly["year"], y_pred_train, color=PALETTE["trend"],
            linewidth=1.5, linestyle="--", alpha=0.8, label="Fitted Trend")
    ax.plot(future_years.flatten(), future_pred, color=PALETTE["pred"],
            linewidth=2.5, linestyle="--", marker="s", markersize=6,
            label=f"Forecast (2025–{2025+forecast_years-1})")
    ax.axvline(2024, color=PALETTE["text"], linewidth=1, linestyle=":", alpha=0.5)
    ax.text(2024.3, ax.get_ylim()[0]+0.2, "Forecast →",
            color=PALETTE["text"], fontsize=9, alpha=0.7)

    style_ax(ax, "🔮 Temperature Forecast — Linear Regression (2025–2034)",
             "Year", "Temperature (°C)")
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], fontsize=9)

    ax.text(0.02, 0.95, f"MAE={mae:.3f}°C | RMSE={rmse:.3f}°C | R²={r2:.4f}",
            transform=ax.transAxes, color=PALETTE["text"], fontsize=9,
            verticalalignment="top",
            bbox=dict(facecolor=PALETTE["bg"], alpha=0.6, edgecolor="none"))

    plt.tight_layout()
    path = f"{OUT}/08_temperature_forecast_lr.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")

    # Save predictions
    pred_df = pd.DataFrame({
        "year"       : future_years.flatten(),
        "predicted_temp" : np.round(future_pred, 3),
        "lower_95"   : np.round(future_pred - margin, 3),
        "upper_95"   : np.round(future_pred + margin, 3),
    })
    pred_df.to_csv("reports/forecast_linear.csv", index=False)
    print(f"  📄 Forecast saved → reports/forecast_linear.csv")
    print(pred_df.to_string(index=False))
    return pred_df, {"mae": mae, "rmse": rmse, "r2": r2}

# ─────────────────────────────────────────────
# 2. SARIMA FORECAST (Monthly)
# ─────────────────────────────────────────────
def sarima_forecast(df, steps=24):
    if not HAS_STATSMODELS:
        print("  ⚠️  Skipping SARIMA — statsmodels not available")
        return None, {}

    monthly = df.groupby(["year","month"])["temperature"].mean().reset_index()
    monthly["ds"] = pd.to_datetime(monthly[["year","month"]].assign(day=1))
    monthly = monthly.sort_values("ds").set_index("ds")["temperature"]

    # ADF Test for stationarity
    adf_result = adfuller(monthly.dropna())
    print(f"  ADF Statistic: {adf_result[0]:.4f} | p-value: {adf_result[1]:.4f}")

    # SARIMAX model (seasonal period = 12 months)
    model = SARIMAX(monthly, order=(1,1,1), seasonal_order=(1,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False)
    print(f"  AIC: {result.aic:.2f} | BIC: {result.bic:.2f}")

    forecast = result.get_forecast(steps=steps)
    pred_mean = forecast.predicted_mean
    conf_int  = forecast.conf_int()

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax.plot(monthly.index, monthly.values, color=PALETTE["hist"],
            linewidth=1.5, label="Historical Monthly Avg")
    ax.plot(pred_mean.index, pred_mean.values, color=PALETTE["pred"],
            linewidth=2.5, linestyle="--", label=f"SARIMA Forecast (+{steps} months)")
    ax.fill_between(pred_mean.index,
                    conf_int.iloc[:,0], conf_int.iloc[:,1],
                    color=PALETTE["ci"], alpha=0.2, label="95% Confidence Interval")
    ax.axvline(monthly.index[-1], color=PALETTE["text"],
               linewidth=1, linestyle=":", alpha=0.5)

    style_ax(ax, "🔮 Monthly Temperature Forecast — SARIMA",
             "Date", "Temperature (°C)")
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], fontsize=9)

    plt.tight_layout()
    path = f"{OUT}/09_temperature_forecast_sarima.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")

    # Save
    forecast_df = pd.DataFrame({
        "date"          : pred_mean.index,
        "predicted_temp": pred_mean.values.round(3),
        "lower_95"      : conf_int.iloc[:,0].values.round(3),
        "upper_95"      : conf_int.iloc[:,1].values.round(3),
    })
    forecast_df.to_csv("reports/forecast_sarima.csv", index=False)
    print(f"  📄 SARIMA forecast saved → reports/forecast_sarima.csv")
    return forecast_df, {"aic": result.aic, "bic": result.bic}

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_forecasting(df=None):
    if df is None:
        df = pd.read_csv("data/climate_data_clean.csv", parse_dates=["date"])

    print("\n═══════════════════════════════════════")
    print("   FORECASTING")
    print("═══════════════════════════════════════")

    print("\n 1. Linear Regression Forecast (Yearly)...")
    lr_pred, lr_metrics = linear_forecast(df, forecast_years=10)

    print("\n 2. SARIMA Monthly Forecast...")
    sarima_pred, sarima_metrics = sarima_forecast(df, steps=24)

    print("\n Forecasting complete.")
    return lr_pred, sarima_pred

if __name__ == "__main__":
    run_forecasting()
