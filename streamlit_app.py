import os
import random
import subprocess
import io
import zipfile
from pathlib import Path

import streamlit as st
from ts_core import (
    load_table,
    infer_date_and_target,
    forecast_linear_safe,
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
    
    # File uploader
    uploaded = st.file_uploader(
        "Upload your data file",
        type=["csv", "xlsx", "xls"],
        help="CSV or Excel files up to 10 MB"
    )
    
    # Example files download
    example_files = sorted(DATA_DIR.glob("*"))
    if example_files:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            for example in example_files:
                zf.write(example, arcname=example.name)
        buffer.seek(0)
        st.download_button(
            "Examples",
            buffer.getvalue(),
            "examples.zip",
            help="Sample datasets for testing"
        )
    
    # Forecast horizon
    horizon = st.number_input(
        "Forecast horizon (steps)",
        min_value=1,
        max_value=1000,
        value=12,
        help="Number of future periods to forecast"
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
if st.sidebar.button("Generate Forecast", type="primary"):
    try:
        with st.spinner("Generating forecast..."):
            result = forecast_linear_safe(df, date_col, target_col, int(horizon))
        
        st.subheader("Forecast Results")
        
        # Chart
        st.line_chart(result.set_index("date")[["y", "yhat"]])
        
        # Details
        with st.expander("Forecast Details"):
            st.dataframe(result.tail(min(50, len(result))))
        
        # Download
        st.download_button(
            "Download Predictions",
            result.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )
        
    except DataError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unexpected error: {e}")

# Footer
st.caption("Baseline uses scikit-learn LinearRegression with fallback to last value if modeling fails.")

# Deploy time
deploy_time = get_deploy_time()
st.markdown(
    f"<div style='position: fixed; bottom: 0; left: 50%; transform: translateX(-50%); font-size:0.75rem; color: gray;'>Deploy: {deploy_time}</div>",
    unsafe_allow_html=True,
)
