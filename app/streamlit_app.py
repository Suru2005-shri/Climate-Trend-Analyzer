"""
app/streamlit_app.py
--------------------
Interactive Streamlit dashboard for the Climate Trend Analyzer.
Lets users explore trends, filter by year/season, and view anomalies.

Run with:  streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os, sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Climate Trend Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0D1117; }
    .stApp { background-color: #0D1117; }
    h1, h2, h3 { color: #E6EDF3 !important; }
    .metric-card {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #E84040; }
    .metric-label { font-size: 0.85rem; color: #8B949E; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

PALETTE = {
    "bg"    : "#0D1117", "panel": "#161B22", "text": "#E6EDF3",
    "grid"  : "#21262D", "temp": "#E84040",  "rain": "#3A7BD5",
    "humid" : "#00B09B", "wind": "#8E44AD",  "acc": "#FF8C00",
}

# ─────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    path = "data/climate_data_clean.csv"
    if not os.path.exists(path):
        st.error("❌ Data not found. Please run: python main.py first.")
        st.stop()
    df = pd.read_csv(path, parse_dates=["date"])
    return df

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def dark_fig(w=14, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])
    return fig, ax

def style(ax, title="", xlabel="", ylabel=""):
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    for s in ax.spines.values(): s.set_edgecolor(PALETTE["grid"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5, linestyle="--")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)

# ═════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════
def main():
    df = load_data()

    # ── SIDEBAR ──────────────────────────────
    with st.sidebar:
        st.image("https://via.placeholder.com/280x80/0D1117/E84040?text=🌍+Climate+Analyzer",
                 use_column_width=True)
        st.markdown("---")
        st.markdown("### 🎛️ Filters")

        year_range = st.slider(
            "📅 Year Range",
            int(df["year"].min()), int(df["year"].max()),
            (int(df["year"].min()), int(df["year"].max()))
        )
        seasons = st.multiselect(
            "🍂 Seasons",
            options=["Winter","Spring","Monsoon","Autumn"],
            default=["Winter","Spring","Monsoon","Autumn"]
        )
        variable = st.selectbox(
            "📊 Primary Variable",
            ["temperature","rainfall","humidity","wind_speed"],
            index=0
        )
        st.markdown("---")
        st.markdown("### 📁 Project Links")
        st.markdown("🔗 [GitHub Repository](#)")
        st.markdown("📄 [View Report](reports/insights_report.txt)")
        st.markdown("---")
        st.caption("Climate Trend Analyzer | Mumbai, India | 1994–2024")

    # Filter data
    mask = (
        (df["year"] >= year_range[0]) &
        (df["year"] <= year_range[1]) &
        (df["season"].isin(seasons))
    )
    fdf = df[mask].copy()

    if fdf.empty:
        st.warning("No data matches filters. Please adjust the sidebar.")
        return

    # ── HEADER ───────────────────────────────
    st.markdown("""
    <h1 style='text-align:center; color:#E6EDF3; font-size:2.5rem; margin-bottom:0'>
        🌍 Climate Trend Analyzer
    </h1>
    <p style='text-align:center; color:#8B949E; font-size:1rem; margin-top:0'>
        Mumbai, India &nbsp;|&nbsp; 30-Year Historical Analysis (1994–2024) &nbsp;|&nbsp; Python Data Science Project
    </p>
    <hr style='border-color:#21262D; margin:16px 0'>
    """, unsafe_allow_html=True)

    # ── KPI METRICS ──────────────────────────
    yearly = fdf.groupby("year")["temperature"].mean()
    slope, inter, r, p, _ = stats.linregress(yearly.index, yearly.values)
    anomaly_days = (fdf["anomaly"] != "Normal").sum()

    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        (c1, f"{fdf['temperature'].mean():.2f}°C", "Avg Temperature"),
        (c2, f"{fdf['rainfall'].sum()/len(fdf['year'].unique()):.0f}mm", "Avg Annual Rainfall"),
        (c3, f"+{slope*10:.3f}°C", "Warming / Decade"),
        (c4, f"{r**2:.4f}", "R² (Trend Fit)"),
        (c5, f"{anomaly_days:,}", "Anomaly Days"),
    ]
    for col, val, lbl in kpis:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TAB LAYOUT ───────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Trends", "🌧️ Rainfall", "🚨 Anomalies", "🔮 Forecast", "📊 Dashboard"
    ])

    # ────────────────────────────────────────
    # TAB 1 — TRENDS
    # ────────────────────────────────────────
    with tab1:
        st.subheader(f"Annual {variable.replace('_',' ').title()} Trend")
        yearly_v = fdf.groupby("year")[variable].mean()
        x, y = yearly_v.index.values, yearly_v.values
        sl, ic, *_ = stats.linregress(x, y)
        fig, ax = dark_fig(14, 5)
        ax.fill_between(x, y.min()*0.98, y, alpha=0.1, color=PALETTE["temp"])
        ax.plot(x, y, color=PALETTE["temp"], linewidth=2, marker="o", markersize=5, label="Annual Avg")
        ax.plot(x, sl*x+ic, color=PALETTE["acc"], linewidth=2.5, linestyle="--",
                label=f"Trend: {sl*10:+.3f}/decade")
        style(ax, f"Annual {variable} Trend", "Year", variable)
        ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
                  labelcolor=PALETTE["text"], fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Seasonal Distribution")
            seasons_order = ["Winter","Spring","Monsoon","Autumn"]
            data = [fdf[fdf["season"]==s][variable].dropna().values for s in seasons_order]
            colors = ["#74B9FF","#55EFC4","#FDCB6E","#E17055"]
            fig2, ax2 = dark_fig(7, 4)
            bp = ax2.boxplot(data, patch_artist=True,
                             medianprops=dict(color="white", linewidth=2),
                             whiskerprops=dict(color=PALETTE["text"]),
                             capprops=dict(color=PALETTE["text"]),
                             flierprops=dict(marker="o", markerfacecolor="#FF6B6B", markersize=3))
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c); patch.set_alpha(0.8)
            ax2.set_xticklabels(seasons_order)
            style(ax2, "Seasonal Box Plot", "Season", variable)
            st.pyplot(fig2, use_container_width=True)
            plt.close()
        with col2:
            st.subheader("Monthly Average")
            monthly_v = fdf.groupby("month")[variable].mean()
            months_abbr = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            fig3, ax3 = dark_fig(7, 4)
            bar_colors = [PALETTE["temp"] if v > monthly_v.mean() else PALETTE["rain"]
                          for v in monthly_v.values]
            ax3.bar(range(1,13), monthly_v.values, color=bar_colors, alpha=0.85)
            ax3.set_xticks(range(1,13))
            ax3.set_xticklabels(months_abbr, fontsize=8)
            style(ax3, "Monthly Pattern", "Month", variable)
            st.pyplot(fig3, use_container_width=True)
            plt.close()

    # ────────────────────────────────────────
    # TAB 2 — RAINFALL
    # ────────────────────────────────────────
    with tab2:
        st.subheader("Rainfall Heatmap — Year × Month")
        pivot = fdf.groupby(["year","month"])["rainfall"].sum().unstack()
        fig4, ax4 = dark_fig(16, 8)
        im = ax4.imshow(pivot.values, aspect="auto", cmap=plt.cm.YlGnBu)
        ax4.set_yticks(range(len(pivot.index)))
        ax4.set_yticklabels(pivot.index, fontsize=8, color=PALETTE["text"])
        months_short = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        ax4.set_xticks(range(12))
        ax4.set_xticklabels(months_short, color=PALETTE["text"])
        for sp in ax4.spines.values(): sp.set_edgecolor(PALETTE["grid"])
        cbar = fig4.colorbar(im, ax=ax4, pad=0.01)
        cbar.ax.tick_params(colors=PALETTE["text"])
        cbar.set_label("Rainfall (mm)", color=PALETTE["text"])
        ax4.set_title("Monthly Total Rainfall Heatmap", color=PALETTE["text"], fontsize=13, fontweight="bold")
        st.pyplot(fig4, use_container_width=True)
        plt.close()

        st.subheader("Annual Total Rainfall")
        annual_rain = fdf.groupby("year")["rainfall"].sum()
        fig5, ax5 = dark_fig(14, 4)
        ax5.bar(annual_rain.index, annual_rain.values, color=PALETTE["rain"], alpha=0.8)
        sl2, ic2, *_ = stats.linregress(annual_rain.index, annual_rain.values)
        ax5.plot(annual_rain.index, sl2*annual_rain.index+ic2, color=PALETTE["acc"],
                 linewidth=2, linestyle="--")
        style(ax5, "Annual Total Rainfall", "Year", "mm")
        st.pyplot(fig5, use_container_width=True)
        plt.close()

    # ────────────────────────────────────────
    # TAB 3 — ANOMALIES
    # ────────────────────────────────────────
    with tab3:
        st.subheader("🚨 Climate Anomaly Detection")
        anom_counts = fdf["anomaly"].value_counts().reset_index()
        anom_counts.columns = ["Event Type","Count"]
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(
                anom_counts.style.background_gradient(cmap="Reds"),
                use_container_width=True
            )
            st.metric("Total Anomaly Days",
                      f"{(fdf['anomaly']!='Normal').sum():,}",
                      f"{(fdf['anomaly']!='Normal').mean()*100:.2f}% of dataset")
        with col2:
            yearly_anom = fdf[fdf["anomaly"]!="Normal"].groupby("year").size().reindex(
                range(year_range[0], year_range[1]+1), fill_value=0)
            fig6, ax6 = dark_fig(10, 4)
            ax6.bar(yearly_anom.index, yearly_anom.values, color=PALETTE["temp"], alpha=0.8)
            ax6.axhline(yearly_anom.mean(), color=PALETTE["acc"], linestyle="--",
                        linewidth=1.5, label=f"Mean = {yearly_anom.mean():.1f}")
            style(ax6, "Anomaly Events Per Year", "Year", "Count")
            ax6.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])
            st.pyplot(fig6, use_container_width=True)
            plt.close()

        st.subheader("Anomaly Timeline")
        sample = fdf.sample(min(2000, len(fdf)), random_state=42).sort_values("date")
        anom_mask = sample["anomaly"] != "Normal"
        fig7, ax7 = dark_fig(14, 4)
        ax7.plot(sample["date"], sample["temperature"], color=PALETTE["rain"],
                 linewidth=0.5, alpha=0.5, label="Temperature")
        ax7.scatter(sample[anom_mask]["date"], sample[anom_mask]["temperature"],
                    color=PALETTE["temp"], s=20, zorder=5, label="Anomaly", alpha=0.9)
        style(ax7, "Temperature with Anomaly Markers", "Date", "°C")
        ax7.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])
        st.pyplot(fig7, use_container_width=True)
        plt.close()

        st.subheader("📋 Anomaly Records")
        anom_df = fdf[fdf["anomaly"] != "Normal"][["date","year","month","temperature","rainfall","humidity","anomaly"]].copy()
        anom_df = anom_df.sort_values("temperature", ascending=False).reset_index(drop=True)
        st.dataframe(anom_df.head(50).style.background_gradient(subset=["temperature"], cmap="Reds"),
                     use_container_width=True)

    # ────────────────────────────────────────
    # TAB 4 — FORECAST
    # ────────────────────────────────────────
    with tab4:
        st.subheader("🔮 Temperature Forecast (2025–2034)")
        forecast_path = "reports/forecast_linear.csv"
        if os.path.exists(forecast_path):
            fc = pd.read_csv(forecast_path)
            yearly_hist = fdf.groupby("year")["temperature"].mean().reset_index()
            fig8, ax8 = dark_fig(14, 6)
            ax8.fill_between(fc["year"], fc["lower_95"], fc["upper_95"],
                             alpha=0.15, color=PALETTE["acc"], label="95% Confidence Interval")
            ax8.plot(yearly_hist["year"], yearly_hist["temperature"], color=PALETTE["rain"],
                     linewidth=2, marker="o", markersize=5, label="Historical")
            ax8.plot(fc["year"], fc["predicted_temp"], color=PALETTE["temp"],
                     linewidth=2.5, marker="s", markersize=6, linestyle="--", label="Forecast")
            ax8.axvline(2024, color=PALETTE["text"], linewidth=1, linestyle=":", alpha=0.5)
            style(ax8, "10-Year Temperature Forecast", "Year", "°C")
            ax8.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])
            st.pyplot(fig8, use_container_width=True)
            plt.close()
            st.subheader("Forecast Table")
            st.dataframe(fc.style.background_gradient(subset=["predicted_temp"], cmap="Reds"),
                         use_container_width=True)
        else:
            st.warning("Run forecasting module first: python src/forecasting.py")

    # ────────────────────────────────────────
    # TAB 5 — DASHBOARD
    # ────────────────────────────────────────
    with tab5:
        st.subheader("📊 Summary Dashboard")
        dash_path = "outputs/figures/10_summary_dashboard.png"
        if os.path.exists(dash_path):
            st.image(dash_path, use_column_width=True,
                     caption="Climate Trend Analyzer — 6-Panel Summary Dashboard")
        else:
            st.warning("Run: python src/generate_report.py to generate the dashboard.")

        st.subheader("📄 Insights Report")
        report_path = "reports/insights_report.txt"
        if os.path.exists(report_path):
            with open(report_path) as f:
                content = f.read()
            st.code(content, language="")
        else:
            st.warning("Run the full pipeline first: python main.py")

if __name__ == "__main__":
    main()
