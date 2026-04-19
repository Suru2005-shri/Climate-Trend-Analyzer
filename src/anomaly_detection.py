"""
anomaly_detection.py
--------------------
Detects climate anomalies using multiple methods:
 1. Z-Score method (statistical deviation)
 2. IQR method (robust outlier detection)
 3. Rolling Z-Score (contextual anomaly detection)
 4. Isolation Forest (ML-based unsupervised detection)
Saves anomaly summary table and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings("ignore")

OUT = "outputs/figures"
os.makedirs(OUT, exist_ok=True)
os.makedirs("reports", exist_ok=True)

PALETTE = {
    "bg"      : "#0D1117",
    "panel"   : "#161B22",
    "text"    : "#E6EDF3",
    "grid"    : "#21262D",
    "normal"  : "#3A7BD5",
    "anomaly" : "#E84040",
    "warn"    : "#FF8C00",
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
# METHOD 1: Z-SCORE
# ─────────────────────────────────────────────
def zscore_detection(df, col="temperature", threshold=2.5):
    z = np.abs(stats.zscore(df[col].dropna()))
    idx = df[col].dropna().index
    flags = pd.Series(False, index=df.index)
    flags.loc[idx[z > threshold]] = True
    return flags

# ─────────────────────────────────────────────
# METHOD 2: IQR
# ─────────────────────────────────────────────
def iqr_detection(df, col="temperature"):
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 2.0 * IQR, Q3 + 2.0 * IQR
    return (df[col] < lower) | (df[col] > upper)

# ─────────────────────────────────────────────
# METHOD 3: ROLLING Z-SCORE (Contextual)
# ─────────────────────────────────────────────
def rolling_zscore_detection(df, col="temperature", window=90, threshold=2.8):
    roll_mean = df[col].rolling(window, center=True, min_periods=1).mean()
    roll_std  = df[col].rolling(window, center=True, min_periods=1).std()
    z = (df[col] - roll_mean) / (roll_std + 1e-6)
    return z.abs() > threshold

# ─────────────────────────────────────────────
# METHOD 4: ISOLATION FOREST (ML)
# ─────────────────────────────────────────────
def isolation_forest_detection(df, contamination=0.03):
    features = ["temperature", "rainfall", "humidity", "wind_speed"]
    X = df[features].fillna(df[features].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=42, n_jobs=-1)
    preds = clf.fit_predict(X_scaled)
    scores = clf.score_samples(X_scaled)
    return pd.Series(preds == -1, index=df.index), pd.Series(-scores, index=df.index)

# ─────────────────────────────────────────────
# PLOT: ANOMALY TIMELINE
# ─────────────────────────────────────────────
def plot_anomaly_timeline(df, anomaly_mask, method_name="Z-Score"):
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    fig.patch.set_facecolor(PALETTE["bg"])

    # Temperature
    ax = axes[0]
    ax.set_facecolor(PALETTE["panel"])
    normal  = df[~anomaly_mask]
    flagged = df[anomaly_mask]
    ax.plot(df["date"], df["temperature"], color=PALETTE["normal"],
            linewidth=0.6, alpha=0.6, label="Normal")
    ax.scatter(flagged["date"], flagged["temperature"],
               color=PALETTE["anomaly"], s=12, zorder=5, label="Anomaly", alpha=0.85)
    style_ax(ax, f"🚨 Temperature Anomalies — {method_name} Method",
             "", "Temperature (°C)")
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], fontsize=9)

    # Rainfall
    ax2 = axes[1]
    ax2.set_facecolor(PALETTE["panel"])
    ax2.bar(df["date"], df["rainfall"], color=PALETTE["normal"],
            alpha=0.4, width=1, label="Normal Rainfall")
    ax2.bar(flagged["date"], flagged["rainfall"], color=PALETTE["anomaly"],
            alpha=0.85, width=2, label="Anomaly Days")
    style_ax(ax2, "🌧️ Rainfall on Anomaly Days",
             "Date", "Rainfall (mm)")
    ax2.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
               labelcolor=PALETTE["text"], fontsize=9)

    plt.tight_layout()
    name = method_name.lower().replace(" ", "_")
    path = f"{OUT}/06_anomaly_timeline_{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")

# ─────────────────────────────────────────────
# PLOT: ANOMALY COUNT PER YEAR
# ─────────────────────────────────────────────
def plot_anomaly_per_year(df, anomaly_mask):
    yearly_count = df[anomaly_mask].groupby("year").size().reindex(
        df["year"].unique(), fill_value=0
    ).sort_index()

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])
    bars = ax.bar(yearly_count.index, yearly_count.values,
                  color=PALETTE["anomaly"], alpha=0.8, width=0.7)
    # Color bars by intensity
    max_val = yearly_count.max()
    for bar, val in zip(bars, yearly_count.values):
        alpha = 0.3 + 0.7 * (val / max_val)
        bar.set_alpha(alpha)

    style_ax(ax, "📅 Anomaly Events Per Year", "Year", "Number of Anomaly Days")
    ax.axhline(yearly_count.mean(), color=PALETTE["warn"], linestyle="--",
               linewidth=1.5, label=f"Mean = {yearly_count.mean():.1f}")
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"])

    plt.tight_layout()
    path = f"{OUT}/07_anomaly_count_per_year.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {path}")

# ─────────────────────────────────────────────
# SAVE ANOMALY REPORT TABLE
# ─────────────────────────────────────────────
def save_anomaly_report(df, anomaly_mask, if_mask, if_scores):
    report = df[anomaly_mask | if_mask].copy()
    report["zscore_flag"]  = anomaly_mask[report.index]
    report["iforest_flag"] = if_mask[report.index]
    report["anomaly_score"] = if_scores[report.index].round(4)
    report = report[["date","year","month","temperature","rainfall",
                      "humidity","anomaly","zscore_flag","iforest_flag","anomaly_score"]]
    report = report.sort_values("anomaly_score", ascending=False)
    report.to_csv("reports/anomaly_report.csv", index=False)
    print(f"  📄 Anomaly report saved → reports/anomaly_report.csv")
    print(f"     Total flagged rows: {len(report):,}")
    print(report.head(10).to_string())
    return report

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_anomaly_detection(df=None):
    if df is None:
        df = pd.read_csv("data/climate_data_clean.csv", parse_dates=["date"])

    print("\n═══════════════════════════════════════")
    print("   ANOMALY DETECTION")
    print("═══════════════════════════════════════")

    print("\n📐 1. Z-Score Detection...")
    zscore_mask = zscore_detection(df, "temperature", threshold=2.5)
    print(f"   → {zscore_mask.sum()} anomalies found (Z > 2.5)")

    print("\n📦 2. IQR Detection...")
    iqr_mask = iqr_detection(df, "temperature")
    print(f"   → {iqr_mask.sum()} anomalies found (IQR method)")

    print("\n🌀 3. Rolling Z-Score Detection...")
    rolling_mask = rolling_zscore_detection(df, "temperature")
    print(f"   → {rolling_mask.sum()} contextual anomalies found")

    print("\n🤖 4. Isolation Forest (ML)...")
    if_mask, if_scores = isolation_forest_detection(df)
    print(f"   → {if_mask.sum()} anomalies found (Isolation Forest)")

    combined_mask = zscore_mask | rolling_mask

    print("\n🖼️  Generating Plots...")
    plot_anomaly_timeline(df, combined_mask, "Z-Score + Rolling")
    plot_anomaly_per_year(df, combined_mask)

    print("\n📋 Saving Report...")
    report = save_anomaly_report(df, combined_mask, if_mask, if_scores)

    summary = {
        "zscore_anomalies"    : int(zscore_mask.sum()),
        "iqr_anomalies"       : int(iqr_mask.sum()),
        "rolling_anomalies"   : int(rolling_mask.sum()),
        "isolation_forest"    : int(if_mask.sum()),
        "combined_anomalies"  : int(combined_mask.sum()),
    }
    print("\n📊 Anomaly Detection Summary:")
    for k, v in summary.items():
        print(f"   {k}: {v}")
    return summary, report

if __name__ == "__main__":
    run_anomaly_detection()
