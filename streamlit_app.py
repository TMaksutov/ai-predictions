import os
import random
import subprocess
import io
import zipfile
from pathlib import Path

import streamlit as st
import pandas as pd
from ts_core import (
    load_table,
    infer_date_and_target,
    forecast_linear_safe,
    forecast_multiple_models,
    DataError,
    detect_interval,
)


def get_deploy_time():
    """Return commit datetime (truncated to minutes) without timezone."""
    try:
        deploy_time = subprocess.check_output(
            [
                "git",
                "show",
                "-s",
                "--format=%cd",
                "--date=format:%Y-%m-%d %H:%M",
                "HEAD",
            ]
        ).decode().strip()
    except Exception:
        deploy_time = "unknown"
    return deploy_time


# Page configuration
st.set_page_config(page_title="Simple Time-Series Predictor", layout="wide")
st.header("Simple Time-Series Predictor (Baseline)")
st.write("Upload CSV/XLSX, choose columns, and get a baseline forecast with safety checks.")

# Constants
DATA_DIR = Path(__file__).parent / "test_files"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Sidebar
with st.sidebar:
    st.subheader("Data Input")
    
    # Example files download
    example_files = sorted(DATA_DIR.glob("*"))
    if example_files:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            for example in example_files:
                zf.write(example, arcname=example.name)
        buffer.seek(0)
        st.download_button(
            "Download Examples",
            buffer.getvalue(),
            "examples.zip",
            help="Sample datasets for testing"
        )
    
    # File uploader
    uploaded = st.file_uploader(
        "Upload your data file",
        type=["csv", "xlsx", "xls"],
        help="CSV or Excel files up to 10 MB"
    )
    
    # Forecast horizon
    horizon = st.number_input(
        "Forecast horizon (steps)",
        min_value=1,
        max_value=1000,
        value=12,
        help="Number of future periods to forecast"
    )

    # Monash dataset filter setting
    monash_max_kb = st.number_input(
        "Monash max file size (KB)",
        min_value=1,
        max_value=1000,
        value=10,
        step=1,
        help="Only include Monash series/files smaller than this size to speed up the benchmark"
    )

# File size check
if uploaded is not None and uploaded.size > MAX_FILE_SIZE:
    st.error(f"File too large. Limit is {MAX_FILE_SIZE // (1024*1024)} MB.")
    st.stop()

# Load data
if uploaded is None:
    # Use sample data
    sample_files = list(DATA_DIR.glob("*.csv"))
    if not sample_files:
        with st.sidebar:
            st.info("Upload a file to begin.")
        st.stop()
    
    choice = random.choice(sample_files)
    with st.sidebar:
        st.info(f"Using sample data: {choice.name}")
    
    with choice.open("rb") as f:
        df = load_table(f)
else:
    try:
        df = load_table(uploaded)
    except DataError as e:
        st.error(str(e))
        st.stop()

# Data preview
with st.expander("Data Preview"):
    st.dataframe(df.head(20))
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Auto-detect columns
auto_date, auto_target = infer_date_and_target(df)

# Column selection
with st.sidebar:
    st.subheader("Column Selection")
    
    date_col = st.selectbox(
        "Date/time column",
        df.columns.tolist(),
        index=(df.columns.get_loc(auto_date) if auto_date in df.columns else 0),
        help="Column containing date/time information"
    )
    
    candidates = [c for c in df.columns if c != date_col]
    target_col = st.selectbox(
        "Target column (numeric)",
        candidates,
        index=(candidates.index(auto_target) if auto_target in candidates else 0),
        help="Column to forecast (must be numeric)"
    )
    
    # Show detected interval
    if date_col:
        interval = detect_interval(df[date_col])
        st.caption(f"Detected interval: {interval}")

# Forecasting
try:
    with st.spinner("Generating forecast with multiple models..."):
        result, metrics = forecast_multiple_models(df, date_col, target_col, int(horizon))

    st.subheader("Forecast Results - Multiple Models")

    # Show model performance metrics
    st.subheader("Model Performance (20% Test Data)")
    st.dataframe(metrics, use_container_width=True)

    # Chart with all models
    st.subheader("Forecast Comparison")

    # Prepare data for plotting
    plot_data = result.set_index("date")
    # Determine historical length from 'kind' column to be robust against cleaning
    hist_len = (plot_data["kind"] == "historical").sum()

    # Get forecast columns
    forecast_cols = [col for col in plot_data.columns if col.endswith('_forecast')]

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot historical data
    ax.plot(plot_data.index[:hist_len], plot_data['actual'][:hist_len], 
            label='Historical', color='black', linewidth=2)

    # Plot forecasts
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i, col in enumerate(forecast_cols):
        model_name = col.replace('_forecast', '')
        color = colors[i % len(colors)]
        ax.plot(plot_data.index[hist_len:], plot_data[col][hist_len:], 
                label=f'{model_name}', color=color, linewidth=2, linestyle='--')

    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Time Series Forecast - Multiple Models')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

    # Details
    with st.expander("Forecast Details"):
        st.dataframe(result.tail(min(50, len(result))))

    # Download
    st.download_button(
        "Download Predictions",
        result.to_csv(index=False).encode("utf-8"),
        file_name="predictions_multiple_models.csv",
        mime="text/csv",
    )

    # Show best model
    if not metrics.empty and 'RMSE' in metrics.columns:
        best_model = metrics.iloc[0]['Model']
        st.success(f"Best performing model: {best_model}")

except DataError as e:
    # Fallback to simple linear forecast for small or problematic datasets
    try:
        with st.spinner("Using simple forecast due to dataset constraints..."):
            simple = forecast_linear_safe(df, date_col, target_col, int(horizon))

        st.subheader("Forecast Results - Simple Model")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        # Determine historical length from 'kind'
        hist_len_simple = (simple['kind'] == 'historical').sum()
        ax.plot(simple['date'].iloc[:hist_len_simple], simple['y'].iloc[:hist_len_simple], label='Historical', color='black', linewidth=2)
        ax.plot(simple['date'].iloc[hist_len_simple:], simple['yhat'].iloc[hist_len_simple:], label='Forecast', color='blue', linewidth=2, linestyle='--')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title('Time Series Forecast - Simple Model')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("Forecast Details"):
            st.dataframe(simple.tail(min(50, len(simple))))

        st.download_button(
            "Download Predictions",
            simple.to_csv(index=False).encode("utf-8"),
            file_name="predictions_simple.csv",
            mime="text/csv",
        )
    except Exception as inner_e:
        st.error(f"Unexpected error during fallback: {inner_e}")
except Exception as e:
    st.error(f"Unexpected error: {e}")

# Benchmark: Test files + Monash (sample)
st.subheader("Benchmark: Test Files + Monash (RMSE by model)")
st.markdown(
    "Compare RMSE across all built-in models on bundled test files and a Monash sample. "
    "Learn more about the Monash Time Series Forecasting Archive: "
    "[Paper](https://arxiv.org/abs/2005.06643), "
    "[GitHub](https://github.com/robjhyndman/tsdl), "
    "[Data source list](https://forecastingdata.org/)."
)

with st.spinner("Computing benchmark table..."):
    model_order = ['Naive', 'Seasonal Naive', 'Linear Trend', 'Exponential Smoothing', 'Moving Average', 'Polynomial Trend']
    rows = []

    # Evaluate on bundled test files
    for path in sorted(DATA_DIR.glob("*.csv")):
        # Skip monash sample here; we'll add it explicitly later
        if path.name.lower() == "monash_sample.csv":
            continue
        try:
            with path.open("rb") as fh:
                df_file = load_table(fh)
            dcol, tcol = infer_date_and_target(df_file)
            if dcol is None or tcol is None:
                continue
            _, mtx = forecast_multiple_models(df_file, dcol, tcol, int(horizon))
            rmse_map = {row['Model']: row['RMSE'] for _, row in mtx.iterrows()}
            row = {"Dataset": path.name}
            for m in model_order:
                row[m] = rmse_map.get(m, float("nan"))
            rows.append(row)
        except Exception:
            # Skip files that cannot be processed
            continue

    # Add Monash sample dataset (filtered by size)
    monash_path = DATA_DIR / "monash_sample.csv"
    try:
        if monash_path.exists():
            size_ok = monash_path.stat().st_size <= int(monash_max_kb) * 1024
            if size_ok:
                with monash_path.open("rb") as fh:
                    df_monash = load_table(fh)
                dcol_m, tcol_m = infer_date_and_target(df_monash)
                if dcol_m is not None and tcol_m is not None:
                    _, mtx_m = forecast_multiple_models(df_monash, dcol_m, tcol_m, int(horizon))
                    rmse_map_m = {row['Model']: row['RMSE'] for _, row in mtx_m.iterrows()}
                    row_m = {"Dataset": "Monash (sample)"}
                    for m in model_order:
                        row_m[m] = rmse_map_m.get(m, float("nan"))
                    rows.append(row_m)
            else:
                st.caption(
                    f"Monash sample excluded: file size {monash_path.stat().st_size} bytes exceeds limit of {int(monash_max_kb)*1024} bytes."
                )
    except Exception as e:
        st.caption(f"Monash sample could not be evaluated: {e}")

    if rows:
        benchmark_df = pd.DataFrame(rows)
        st.dataframe(benchmark_df, use_container_width=True)
    else:
        st.info("No datasets available for benchmarking.")

# Footer
st.caption("Multiple models: Naive, Seasonal Naive, Linear Trend, Exponential Smoothing, Moving Average, and Polynomial Trend. Models are evaluated on 20% test data for stability comparison.")

# Deploy time
deploy_time = get_deploy_time()
st.markdown(
    f"<div style='position: fixed; bottom: 0; left: 50%; transform: translateX(-50%); font-size:0.75rem; color: gray;'>Deploy: {deploy_time}</div>",
    unsafe_allow_html=True,
)
