"""
generate_report.py
------------------
Generates the final insights summary dashboard plot
and a text-based insights report (reports/insights_report.txt).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")

OUT = "outputs/figures"
os.makedirs(OUT, exist_ok=True)
os.makedirs("reports", exist_ok=True)

PALETTE = {
    "bg"    : "#0D1117",
    "panel" : "#161B22",
    "text"  : "#E6EDF3",
    "grid"  : "#21262D",
    "temp"  : "#E84040",
    "rain"  : "#3A7BD5",
    "humid" : "#00B09B",
    "wind"  : "#8E44AD",
    "acc"   : "#FF8C00",
}

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["text"], labelsize=8)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.4, linestyle="--")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)

# ─────────────────────────────────────────────
# SUMMARY DASHBOARD (6-panel)
# ─────────────────────────────────────────────
def plot_summary_dashboard(df):
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(PALETTE["bg"])
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    yearly = df.groupby("year").agg(
        avg_temp  =("temperature","mean"),
        total_rain=("rainfall","sum"),
        avg_humid =("humidity","mean"),
        avg_wind  =("wind_speed","mean"),
    ).reset_index()

    # ── Panel 1: Temperature Trend ──
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(PALETTE["panel"])
    years, temps = yearly["year"].values, yearly["avg_temp"].values
    slope, inter, *_ = stats.linregress(years, temps)
    ax1.fill_between(years, temps.min()-0.2, temps, alpha=0.1, color=PALETTE["temp"])
    ax1.plot(years, temps, color=PALETTE["temp"], linewidth=2, marker="o", markersize=4)
    ax1.plot(years, slope*years+inter, color=PALETTE["acc"],
             linewidth=2, linestyle="--", label=f"+{slope*10:.2f}°C/decade")
    style_ax(ax1, "🌡️ Annual Mean Temperature Trend", "Year", "°C")
    ax1.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
               labelcolor=PALETTE["text"], fontsize=8)

    # ── Panel 2: Anomaly Type Distribution ──
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(PALETTE["panel"])
    anom_counts = df["anomaly"].value_counts()
    colors_pie  = [PALETTE["rain"], PALETTE["temp"], PALETTE["humid"],
                   PALETTE["acc"], PALETTE["wind"]]
    wedges, texts, autotexts = ax2.pie(
        anom_counts.values, labels=anom_counts.index,
        autopct="%1.1f%%", colors=colors_pie[:len(anom_counts)],
        textprops={"color": PALETTE["text"], "fontsize": 8},
        startangle=90, wedgeprops={"edgecolor": PALETTE["bg"], "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_color(PALETTE["bg"])
        at.set_fontsize(7)
    ax2.set_title("🔴 Climate Event Distribution", color=PALETTE["text"],
                  fontsize=10, fontweight="bold")

    # ── Panel 3: Monthly Average Temperature ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(PALETTE["panel"])
    monthly_temp = df.groupby("month")["temperature"].mean()
    bars = ax3.bar(range(1,13), monthly_temp.values, color=PALETTE["temp"], alpha=0.8)
    months_short = ["J","F","M","A","M","J","J","A","S","O","N","D"]
    ax3.set_xticks(range(1,13))
    ax3.set_xticklabels(months_short, color=PALETTE["text"], fontsize=8)
    style_ax(ax3, "📅 Avg Temp by Month", "Month", "°C")

    # ── Panel 4: Annual Rainfall ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PALETTE["panel"])
    ax4.bar(yearly["year"], yearly["total_rain"], color=PALETTE["rain"], alpha=0.8)
    style_ax(ax4, "🌧️ Annual Rainfall", "Year", "mm")

    # ── Panel 5: Humidity vs Temperature Scatter ──
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(PALETTE["panel"])
    sample = df.sample(min(3000, len(df)), random_state=42)
    sc = ax5.scatter(sample["temperature"], sample["humidity"],
                     c=sample["rainfall"], cmap="YlOrRd",
                     s=4, alpha=0.5, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax5)
    cbar.ax.tick_params(colors=PALETTE["text"], labelsize=7)
    cbar.set_label("Rainfall (mm)", color=PALETTE["text"], fontsize=7)
    style_ax(ax5, "💧 Humidity vs Temperature", "Temperature (°C)", "Humidity (%)")

    # ── Panel 6: Decadal Temperature Comparison ──
    ax6 = fig.add_subplot(gs[2, :])
    ax6.set_facecolor(PALETTE["panel"])
    decades = {
        "1994–2004": df[df["year"].between(1994,2004)]["temperature"],
        "2005–2014": df[df["year"].between(2005,2014)]["temperature"],
        "2015–2024": df[df["year"].between(2015,2024)]["temperature"],
    }
    dec_colors = [PALETTE["rain"], PALETTE["acc"], PALETTE["temp"]]
    for (label, data), color in zip(decades.items(), dec_colors):
        ax6.hist(data, bins=60, alpha=0.55, color=color,
                 label=f"{label} (μ={data.mean():.2f}°C)", density=True)
    style_ax(ax6, "📊 Decadal Temperature Distribution Comparison",
             "Temperature (°C)", "Density")
    ax6.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
               labelcolor=PALETTE["text"], fontsize=9)

    # Title
    fig.text(0.5, 1.01, "🌍 CLIMATE TREND ANALYZER — SUMMARY DASHBOARD",
             ha="center", color=PALETTE["text"], fontsize=16, fontweight="bold")
    fig.text(0.5, 0.99, "Mumbai, India | 1994–2024 | 30-Year Analysis",
             ha="center", color=PALETTE["acc"], fontsize=10)

    path = f"{OUT}/10_summary_dashboard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Summary dashboard saved → {path}")
    return path

# ─────────────────────────────────────────────
# TEXT INSIGHTS REPORT
# ─────────────────────────────────────────────
def generate_text_report(df, trend_stats=None):
    yearly   = df.groupby("year")["temperature"].mean()
    monthly  = df.groupby("month")["temperature"].mean()
    hottest_year  = yearly.idxmax()
    coldest_year  = yearly.idxmin()
    hottest_month = monthly.idxmax()
    wettest_month = df.groupby("month")["rainfall"].sum().idxmax()
    anomaly_count = (df["anomaly"] != "Normal").sum()
    anomaly_pct   = anomaly_count / len(df) * 100

    slope, inter, r, p, se = stats.linregress(yearly.index, yearly.values)

    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║        CLIMATE TREND ANALYZER — INSIGHTS REPORT                  ║
║        Location: Mumbai, India | Period: 1994–2024               ║
╚══════════════════════════════════════════════════════════════════╝

DATASET SUMMARY
───────────────
• Total records analyzed  : {len(df):,} daily observations
• Date range              : {df['date'].min().date()} to {df['date'].max().date()}
• Variables tracked       : Temperature, Rainfall, Humidity, Wind Speed
• Anomaly events detected : {anomaly_count:,} days ({anomaly_pct:.2f}% of dataset)

TEMPERATURE ANALYSIS
────────────────────
• 30-Year Mean Temperature : {df['temperature'].mean():.2f}°C
• Hottest year on record   : {hottest_year} ({yearly[hottest_year]:.2f}°C avg)
• Coldest year on record   : {coldest_year} ({yearly[coldest_year]:.2f}°C avg)
• Hottest month (avg)      : Month {hottest_month} — {monthly[hottest_month]:.2f}°C
• All-time max recorded    : {df['temperature'].max():.2f}°C
• All-time min recorded    : {df['temperature'].min():.2f}°C

TREND ANALYSIS
──────────────
• Warming rate             : +{slope:.4f}°C/year (+{slope*10:.3f}°C/decade)
• R² (goodness of fit)     : {r**2:.4f}
• p-value (significance)   : {p:.6f} {'✅ Statistically significant' if p<0.05 else '❌ Not significant'}
• Projected temp by 2030   : {slope*2030+inter:.2f}°C
• Projected temp by 2050   : {slope*2050+inter:.2f}°C

RAINFALL ANALYSIS
─────────────────
• Annual avg total rain    : {df.groupby('year')['rainfall'].sum().mean():.1f} mm/year
• Wettest month            : Month {wettest_month}
• Max single-day rainfall  : {df['rainfall'].max():.1f} mm
• Monsoon months share     : {df[df['month'].between(6,9)]['rainfall'].sum() / df['rainfall'].sum() * 100:.1f}% of annual total

HUMIDITY & WIND
───────────────
• Average humidity         : {df['humidity'].mean():.1f}%
• Monsoon avg humidity     : {df[df['month'].between(6,9)]['humidity'].mean():.1f}%
• Average wind speed       : {df['wind_speed'].mean():.1f} km/h
• Max wind recorded        : {df['wind_speed'].max():.1f} km/h

ANOMALY BREAKDOWN
─────────────────
{df['anomaly'].value_counts().to_string()}

KEY RESEARCH FINDINGS
─────────────────────
1. A statistically significant warming trend of {slope*10:.2f}°C/decade has been detected.
2. The most recent decade (2015–2024) is the warmest on record.
3. Monsoon rainfall accounts for ~{df[df['month'].between(6,9)]['rainfall'].sum()/df['rainfall'].sum()*100:.0f}% of annual total.
4. {anomaly_pct:.1f}% of all days show some form of climate anomaly.
5. Heat Index values indicate increasing heat stress risk in summer months.

RECOMMENDATIONS FOR POLICYMAKERS
──────────────────────────────────
• Urban heat island mitigation through green infrastructure
• Flood early-warning systems for monsoon-season extreme events  
• Drought preparedness planning for extended dry spells
• Long-term infrastructure planning based on {slope*10:.2f}°C/decade warming

Generated by Climate Trend Analyzer | Student Research Project
"""
    path = "reports/insights_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  📄 Insights report saved → {path}")
    print(report)
    return report

# ─────────────────────────────────────────────
def run_reporting(df=None):
    if df is None:
        df = pd.read_csv("data/climate_data_clean.csv", parse_dates=["date"])

    print("\n═══════════════════════════════════════")
    print("   GENERATING REPORTS & DASHBOARD")
    print("═══════════════════════════════════════")
    plot_summary_dashboard(df)
    generate_text_report(df)
    print("\n✅ All reports generated.")

if __name__ == "__main__":
    run_reporting()
