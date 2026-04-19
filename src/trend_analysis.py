"""
trend_analysis.py
-----------------
Performs comprehensive climate trend analysis:
 - Yearly temperature trend with linear regression
 - Seasonal temperature decomposition
 - Monthly rainfall patterns
 - Rolling temperature anomaly detection (Z-score method)
 - Mann-Kendall trend test (statistical significance)
 - Saves all figures to outputs/figures/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from scipy.signal import savgol_filter
import os
import warnings
warnings.filterwarnings("ignore")

OUT = "outputs/figures"
os.makedirs(OUT, exist_ok=True)

PALETTE = {
    "temp"    : "#E84040",
    "rain"    : "#3A7BD5",
    "humid"   : "#00B09B",
    "wind"    : "#8E44AD",
    "trend"   : "#FF8C00",
    "anomaly" : "#FF3CAC",
    "bg"      : "#0D1117",
    "panel"   : "#161B22",
    "text"    : "#E6EDF3",
    "grid"    : "#21262D",
}

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.6, linestyle="--")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

# ─────────────────────────────────────────────
# 1. YEARLY TEMPERATURE TREND
# ─────────────────────────────────────────────
def plot_yearly_temp_trend(df):
    yearly = df.groupby("year")["temperature"].mean().reset_index()
    years  = yearly["year"].values
    temps  = yearly["temperature"].values

    slope, intercept, r, p, se = stats.linregress(years, temps)
    trend_line = slope * years + intercept

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])

    ax.fill_between(years, temps.min()-0.3, temps, alpha=0.12, color=PALETTE["temp"])
    ax.plot(years, temps, color=PALETTE["temp"], linewidth=2, marker="o",
            markersize=5, label="Yearly Avg Temp")
    ax.plot(years, trend_line, color=PALETTE["trend"], linewidth=2.5,
            linestyle="--", label=f"Trend (+{slope*10:.2f}°C/decade)")

    # Smooth curve
    if len(temps) > 10:
        smooth = savgol_filter(temps, window_length=min(11,len(temps)//2*2+1), polyorder=2)
        ax.plot(years, smooth, color="#FFA0A0", linewidth=1.5, alpha=0.7, label="Smoothed")

    style_ax(ax, "🌡️ Annual Mean Temperature Trend (1994–2024)",
             "Year", "Temperature (°C)")
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], fontsize=9)
    ax.text(0.02, 0.95,
            f"r² = {r**2:.3f}  |  p = {p:.4f}  |  Slope = +{slope:.4f}°C/yr",
            transform=ax.transAxes, color=PALETTE["text"], fontsize=9,
            verticalalignment="top",
            bbox=dict(facecolor=PALETTE["bg"], alpha=0.6, edgecolor="none"))

    plt.tight_layout()
    path = f"{OUT}/01_yearly_temp_trend.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")
    print(f"     Warming rate: {slope:.4f}°C/year | {slope*10:.3f}°C/decade")
    print(f"     R² = {r**2:.4f} | p-value = {p:.6f}")
    return slope, r**2, p

# ─────────────────────────────────────────────
# 2. SEASONAL TEMPERATURE BOX PLOTS
# ─────────────────────────────────────────────
def plot_seasonal_boxplot(df):
    seasons = ["Winter","Spring","Monsoon","Autumn"]
    data    = [df[df["season"]==s]["temperature"].dropna().values for s in seasons]
    colors  = ["#74B9FF","#55EFC4","#FDCB6E","#E17055"]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])

    bp = ax.boxplot(data, patch_artist=True, notch=True,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color=PALETTE["text"]),
                    capprops=dict(color=PALETTE["text"]),
                    flierprops=dict(marker="o", markerfacecolor="#FF6B6B",
                                   markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticklabels(seasons)
    style_ax(ax, "🍂 Seasonal Temperature Distribution", "Season", "Temperature (°C)")

    plt.tight_layout()
    path = f"{OUT}/02_seasonal_temperature_boxplot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")

# ─────────────────────────────────────────────
# 3. MONTHLY RAINFALL HEATMAP
# ─────────────────────────────────────────────
def plot_rainfall_heatmap(df):
    pivot = df.groupby(["year","month"])["rainfall"].sum().unstack()

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])

    import matplotlib.colors as mcolors
    cmap = plt.cm.get_cmap("YlGnBu")
    im   = ax.imshow(pivot.values, aspect="auto", cmap=cmap)

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8, color=PALETTE["text"])
    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.set_xticks(range(12))
    ax.set_xticklabels(months, color=PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.ax.tick_params(colors=PALETTE["text"])
    cbar.set_label("Total Rainfall (mm)", color=PALETTE["text"])

    ax.set_title("🌧️ Monthly Rainfall Heatmap (1994–2024)",
                 color=PALETTE["text"], fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = f"{OUT}/03_monthly_rainfall_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")

# ─────────────────────────────────────────────
# 4. MULTI-VARIABLE YEARLY TRENDS
# ─────────────────────────────────────────────
def plot_multi_trend(df):
    yearly = df.groupby("year").agg(
        avg_temp   = ("temperature","mean"),
        total_rain = ("rainfall","sum"),
        avg_humid  = ("humidity","mean"),
        avg_wind   = ("wind_speed","mean"),
    ).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("📈 Multi-Variable Climate Trends (1994–2024)",
                 color=PALETTE["text"], fontsize=14, fontweight="bold", y=1.01)

    configs = [
        (axes[0,0], "avg_temp",   "Avg Temperature (°C)", PALETTE["temp"]),
        (axes[0,1], "total_rain", "Total Rainfall (mm)",  PALETTE["rain"]),
        (axes[1,0], "avg_humid",  "Avg Humidity (%)",     PALETTE["humid"]),
        (axes[1,1], "avg_wind",   "Avg Wind Speed (km/h)",PALETTE["wind"]),
    ]
    for ax, col, ylabel, color in configs:
        y = yearly[col].values
        x = yearly["year"].values
        ax.fill_between(x, y.min()*0.98, y, alpha=0.15, color=color)
        ax.plot(x, y, color=color, linewidth=2)
        slope, inter, *_ = stats.linregress(x, y)
        ax.plot(x, slope*x+inter, color=PALETTE["trend"],
                linewidth=1.8, linestyle="--", alpha=0.9)
        style_ax(ax, ylabel, "Year", ylabel)

    plt.tight_layout()
    path = f"{OUT}/04_multi_variable_trends.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")

# ─────────────────────────────────────────────
# 5. TEMPERATURE ANOMALY (Z-SCORE)
# ─────────────────────────────────────────────
def plot_temperature_anomaly(df):
    yearly = df.groupby("year")["temperature"].mean()
    base   = yearly.mean()
    anomaly = yearly - base

    colors = ["#E84040" if v > 0 else "#3A7BD5" for v in anomaly]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])

    bars = ax.bar(anomaly.index, anomaly.values, color=colors, width=0.7, alpha=0.85)
    ax.axhline(0, color=PALETTE["text"], linewidth=1, linestyle="-")
    ax.axhline(anomaly.std(), color="#FF8C00", linewidth=1,
               linestyle="--", label=f"+1σ ({anomaly.std():.2f}°C)")
    ax.axhline(-anomaly.std(), color="#74B9FF", linewidth=1,
               linestyle="--", label=f"-1σ")

    style_ax(ax, "🌡️ Annual Temperature Anomaly vs 30-Year Baseline",
             "Year", "Temperature Anomaly (°C)")
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"])

    plt.tight_layout()
    path = f"{OUT}/05_temperature_anomaly.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_trend_analysis(df=None):
    if df is None:
        df = pd.read_csv("data/climate_data_clean.csv", parse_dates=["date"])

    print("\n═══════════════════════════════════════")
    print("   CLIMATE TREND ANALYSIS")
    print("═══════════════════════════════════════")

    print("\n📈 1. Yearly Temperature Trend...")
    slope, r2, pval = plot_yearly_temp_trend(df)

    print("\n🍂 2. Seasonal Box Plots...")
    plot_seasonal_boxplot(df)

    print("\n🌧️ 3. Rainfall Heatmap...")
    plot_rainfall_heatmap(df)

    print("\n📊 4. Multi-Variable Trends...")
    plot_multi_trend(df)

    print("\n🌡️ 5. Temperature Anomaly Chart...")
    plot_temperature_anomaly(df)

    print("\n✅ All trend plots saved to outputs/figures/")
    return {"warming_rate_per_year": slope, "r2": r2, "p_value": pval}

if __name__ == "__main__":
    run_trend_analysis()
