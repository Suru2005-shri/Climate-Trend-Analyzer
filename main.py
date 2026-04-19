"""
main.py
-------
Master pipeline for Climate Trend Analyzer.
Runs all modules in sequence:
  1. Generate synthetic dataset
  2. Preprocess & clean data
  3. Trend analysis
  4. Anomaly detection
  5. Forecasting
  6. Report generation

Usage:
  python main.py
  python main.py --skip-generate   (if data already exists)
"""

import argparse
import os
import sys
import time

# ── Banner ────────────────────────────────────
BANNER = """
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   🌍  CLIMATE TREND ANALYZER                         ║
║       A 30-Year Climate Analysis System               ║
║       Built with Python | Pandas | Scikit-learn       ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
"""

def main():
    print(BANNER)
    parser = argparse.ArgumentParser(description="Climate Trend Analyzer Pipeline")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip dataset generation if data already exists")
    args = parser.parse_args()

    start_time = time.time()
    results    = {}

    # ── Step 1: Generate Dataset ──────────────
    if not args.skip_generate or not os.path.exists("data/climate_data_raw.csv"):
        print("\n" + "="*55)
        print("  STEP 1: GENERATING SYNTHETIC CLIMATE DATASET")
        print("="*55)
        import importlib, src.generate_dataset as gd
        importlib.reload(gd)
    else:
        print("\n✅ Step 1: Using existing dataset (--skip-generate)")

    # ── Step 2: Preprocess ────────────────────
    print("\n" + "="*55)
    print("  STEP 2: DATA CLEANING & PREPROCESSING")
    print("="*55)
    from src.preprocess import run_pipeline
    df = run_pipeline()
    results["rows"] = len(df)

    # ── Step 3: Trend Analysis ────────────────
    print("\n" + "="*55)
    print("  STEP 3: CLIMATE TREND ANALYSIS")
    print("="*55)
    from src.trend_analysis import run_trend_analysis
    trend_stats = run_trend_analysis(df)
    results["trend"] = trend_stats

    # ── Step 4: Anomaly Detection ─────────────
    print("\n" + "="*55)
    print("  STEP 4: ANOMALY DETECTION")
    print("="*55)
    from src.anomaly_detection import run_anomaly_detection
    anomaly_summary, anomaly_report = run_anomaly_detection(df)
    results["anomaly"] = anomaly_summary

    # ── Step 5: Forecasting ───────────────────
    print("\n" + "="*55)
    print("  STEP 5: FORECASTING")
    print("="*55)
    from src.forecasting import run_forecasting
    lr_pred, sarima_pred = run_forecasting(df)
    results["forecast_2030"] = (lr_pred[lr_pred["year"]==2030]["predicted_temp"].values[0]
                                 if lr_pred is not None and 2030 in lr_pred["year"].values
                                 else "N/A")

    # ── Step 6: Reports ───────────────────────
    print("\n" + "="*55)
    print("  STEP 6: GENERATING REPORTS & DASHBOARD")
    print("="*55)
    from src.generate_report import run_reporting
    run_reporting(df)

    # ── Final Summary ─────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "═"*55)
    print("  ✅  PIPELINE COMPLETE")
    print("═"*55)
    print(f"\n  📊 Records processed     : {results['rows']:,}")
    print(f"  🌡️  Warming rate detected  : {results['trend']['warming_rate_per_year']:.4f}°C/year")
    print(f"  🚨 Anomalies detected     : {results['anomaly']['combined_anomalies']}")
    print(f"  ⏱️  Total runtime          : {elapsed:.1f}s")
    print("\n  📁 Output files created:")
    for root, _, files in os.walk("outputs"):
        for f in sorted(files):
            print(f"     outputs/{f}")
    for root, _, files in os.walk("reports"):
        for f in sorted(files):
            print(f"     reports/{f}")
    print("\n  🚀 Ready to push to GitHub!\n")

if __name__ == "__main__":
    main()
