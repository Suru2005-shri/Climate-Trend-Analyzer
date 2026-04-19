# 🌍 Climate Trend Analyzer

> **A comprehensive 30-year climate data analysis system** built with Python, featuring statistical trend detection, ML-based anomaly detection, temperature forecasting, and interactive visualizations.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0-green)](https://pandas.pydata.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()

---

## 📌 Project Overview

The **Climate Trend Analyzer** is an end-to-end data science project that ingests historical daily climate data (temperature, rainfall, humidity, wind speed) for **Mumbai, India (1994–2024)** and performs:

- **Long-term trend analysis** with statistical significance testing
- **Seasonal decomposition** and monthly pattern analysis
- **Multi-method anomaly detection** (Z-Score, IQR, Rolling Z-Score, Isolation Forest)
- **Temperature forecasting** using Linear Regression and SARIMA models
- **Automated insights reporting** with policy recommendations

This project demonstrates a real-world workflow used by climate scientists, environmental agencies, and smart city planners.

---

## 🧩 Problem Statement

Climate change is one of the most critical challenges of our time. However:

- Raw climate data is **noisy, incomplete, and complex**
- Identifying **genuine warming trends** vs. natural variability requires rigorous statistical methods
- **Anomalous events** (heatwaves, extreme rainfall) need automated detection for early warning systems
- Governments and urban planners need **forward-looking projections** to design resilient infrastructure

This project solves all of the above using only **public data and open-source Python tools**.

---

## 🏢 Industry Relevance

| Sector | Use Case |
|---|---|
| **Government / IPCC** | National climate reports, carbon policy evidence |
| **Smart Cities** | Urban heat island planning, flood risk mapping |
| **Agriculture** | Crop season forecasting, drought alerts |
| **Insurance** | Extreme weather risk pricing |
| **Research Institutions** | Peer-reviewed climate trend publications |
| **Environmental NGOs** | Advocacy data, awareness campaigns |

---

## ⚙️ Tech Stack

| Category | Tools Used |
|---|---|
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | Scikit-learn (Isolation Forest) |
| **Statistical Analysis** | SciPy (Linear Regression, Z-Score, ADF Test) |
| **Time-Series Forecasting** | Statsmodels (SARIMA), Linear Regression |
| **Dashboard** | Streamlit |
| **Environment** | Jupyter Notebook, Virtual Environment |

---

## 🏗️ Project Architecture

```
                    ┌─────────────────────────┐
                    │   RAW CLIMATE DATA       │
                    │  (CSV / Synthetic Gen)   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   DATA PREPROCESSING     │
                    │  • Missing value impute  │
                    │  • IQR outlier capping   │
                    │  • Feature engineering   │
                    └────────────┬────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                       │
┌─────────▼────────┐  ┌──────────▼────────┐  ┌─────────▼────────┐
│  TREND ANALYSIS  │  │ ANOMALY DETECTION  │  │   FORECASTING    │
│ • Yearly trends  │  │ • Z-Score          │  │ • Linear Regr.   │
│ • Seasonal split │  │ • Rolling Z-Score  │  │ • SARIMA         │
│ • Mann-Kendall   │  │ • Isolation Forest │  │ • 10-yr outlook  │
└─────────┬────────┘  └──────────┬────────┘  └─────────┬────────┘
          │                      │                       │
          └──────────────────────┼───────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   VISUALIZATION ENGINE   │
                    │  10 Publication-quality  │
                    │  charts + Dashboard       │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   INSIGHTS REPORT        │
                    │  Text + CSV + PNG outputs│
                    └─────────────────────────┘
```

---

## 📁 Folder Structure

```
Climate-Trend-Analyzer/
│
├── data/
│   ├── climate_data_raw.csv          # Raw synthetic dataset (11,323 rows)
│   └── climate_data_clean.csv        # Cleaned & feature-engineered dataset
│
├── src/
│   ├── __init__.py
│   ├── generate_dataset.py           # Synthetic data generator (30 years)
│   ├── preprocess.py                 # Data cleaning pipeline
│   ├── trend_analysis.py             # Statistical trend analysis + 5 charts
│   ├── anomaly_detection.py          # 4-method anomaly detection
│   ├── forecasting.py                # LR + SARIMA forecasting
│   └── generate_report.py           # Dashboard + insights report
│
├── outputs/
│   └── figures/                      # All 10 generated charts (PNG)
│       ├── 01_yearly_temp_trend.png
│       ├── 02_seasonal_temperature_boxplot.png
│       ├── 03_monthly_rainfall_heatmap.png
│       ├── 04_multi_variable_trends.png
│       ├── 05_temperature_anomaly.png
│       ├── 06_anomaly_timeline_*.png
│       ├── 07_anomaly_count_per_year.png
│       ├── 08_temperature_forecast_lr.png
│       ├── 09_temperature_forecast_sarima.png
│       └── 10_summary_dashboard.png
│
├── reports/
│   ├── anomaly_report.csv            # Flagged anomaly records
│   ├── forecast_linear.csv           # 2025–2034 temperature predictions
│   └── insights_report.txt          # Full text insights + recommendations
│
├── notebooks/
│   └── Climate_Analysis_EDA.ipynb   # Jupyter notebook walkthrough
│
├── app/
│   └── streamlit_app.py             # Interactive Streamlit dashboard
│
├── docs/
│   └── project_explanation.md       # Detailed project documentation
│
├── main.py                           # Master pipeline runner
├── requirements.txt                  # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip

### Step 1: Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Climate-Trend-Analyzer.git
cd Climate-Trend-Analyzer
```

### Step 2: Create a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify installation
```bash
python -c "import pandas, numpy, matplotlib, scipy, sklearn; print('✅ All packages ready')"
```

---

## ▶️ How to Run

### Option A: Run the complete pipeline (recommended)
```bash
python main.py
```

### Option B: Run individual modules
```bash
# Step 1 — Generate dataset
python src/generate_dataset.py

# Step 2 — Preprocess data
python src/preprocess.py

# Step 3 — Trend analysis
python src/trend_analysis.py

# Step 4 — Anomaly detection
python src/anomaly_detection.py

# Step 5 — Forecasting
python src/forecasting.py

# Step 6 — Reports & dashboard
python src/generate_report.py
```

### Option C: Launch Streamlit app
```bash
streamlit run app/streamlit_app.py
```

### Option D: Run in Jupyter Notebook
```bash
jupyter notebook notebooks/Climate_Analysis_EDA.ipynb
```

---

## 📊 Dataset Details

| Field | Description | Range |
|---|---|---|
| `date` | Daily timestamp | 1994-01-01 to 2024-12-31 |
| `temperature` | Daily avg temperature (°C) | 14–46°C |
| `rainfall` | Daily rainfall (mm) | 0–200mm |
| `humidity` | Relative humidity (%) | 30–100% |
| `wind_speed` | Wind speed (km/h) | 0–80 km/h |
| `season` | Season label | Winter/Spring/Monsoon/Autumn |
| `anomaly` | Event type | Normal/Heatwave/Cold Snap/Extreme Rainfall |

**Dataset size:** 11,323 daily observations | ~1% missing values (realistic)

**Simulated features:**
- 0.04°C/year warming trend (IPCC-aligned)
- Realistic monsoon seasonality
- 15 heatwave events, 8 cold snaps, 20 extreme rainfall days

---

## 📈 Key Results

| Metric | Value |
|---|---|
| 30-Year Mean Temperature | 27.61°C |
| Detected Warming Rate | **+0.366°C/decade** |
| R² of Trend (p < 0.001) | **0.9356** (statistically significant) |
| Hottest Year on Record | **2024** (28.20°C avg) |
| Monsoon Share of Annual Rainfall | **83.4%** |
| Projected Temperature by 2030 | **28.37°C** |
| Projected Temperature by 2050 | **29.10°C** |
| Anomaly Events Detected | **344 days** (Isolation Forest) |

---

## 🖼️ Output Visualizations

| Chart | Description |
|---|---|
| `01_yearly_temp_trend.png` | Annual mean temperature with trend line |
| `02_seasonal_temperature_boxplot.png` | Temperature spread across 4 seasons |
| `03_monthly_rainfall_heatmap.png` | Year × Month rainfall heatmap |
| `04_multi_variable_trends.png` | Temp, Rain, Humidity, Wind — all trends |
| `05_temperature_anomaly.png` | Yearly deviation from 30-year baseline |
| `06_anomaly_timeline_*.png` | Daily anomaly scatter overlay |
| `07_anomaly_count_per_year.png` | Frequency of anomaly events per year |
| `08_temperature_forecast_lr.png` | 10-year forecast with confidence interval |
| `10_summary_dashboard.png` | Full 6-panel summary dashboard |

---

## 🔮 Future Improvements

- [ ] Integrate real NASA GISS or NOAA open climate data
- [ ] Add CO₂ concentration as a feature variable
- [ ] Implement LSTM deep learning forecasting
- [ ] Build interactive Plotly/Dash web dashboard
- [ ] Add multi-city comparison module
- [ ] Deploy on Streamlit Cloud or Hugging Face Spaces
- [ ] Add spatial visualization using GeoPandas + Folium

---

## 🎓 Learning Outcomes

By studying or building this project, you will learn:
- End-to-end data science pipeline design
- Time-series data cleaning and feature engineering
- Statistical trend analysis (Linear Regression, Mann-Kendall)
- Unsupervised ML anomaly detection (Isolation Forest)
- Publication-quality data visualization with Matplotlib
- ARIMA/SARIMA time-series forecasting
- Professional GitHub project structure and documentation

---

## 👤 Author

**Shruti Srivastava**  
*Data Science & Analytics Enthusiast*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](www.linkedin.com/in/shruti-srivastava-36b26232a)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Suru2005-shri)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

> *"Climate change is a complex problem that requires data-driven solutions. This project demonstrates how Python and open-source tools can turn raw numbers into meaningful environmental insights."*
