import os
import random
import subprocess
import io
import zipfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
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

st.set_page_config(page_title="Simple Time-Series Predictor", layout="wide")
st.header("Simple Time-Series Predictor (Baseline)")
st.write("Upload CSV/XLSX, choose columns, and get a small baseline forecast with strong safety checks.")

DATA_DIR = Path(__file__).parent / "test_files"

with st.sidebar:
    example_files = sorted(DATA_DIR.glob("*"))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for example in example_files:
            zf.write(example, arcname=example.name)
    buffer.seek(0)
    st.download_button(
        "Examples",
        buffer.getvalue(),
        "examples.zip",
    )
    uploaded = st.file_uploader(
        "",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )
    components.html(
        """
        <script>
        const fu = window.parent.document.querySelector('[data-testid="stFileUploader"]');
        if (fu) {
            const btn = fu.querySelector('button');
            if (btn) { btn.textContent = 'Browse'; }
            const help = fu.querySelector('small');
            if (help) {
                help.textContent = 'CSV or Excel (.csv, .xlsx, .xls) (max 10 MB)';
                help.style.display = 'block';
            }
        }
        </script>
        """,
        height=0,
    )
    horizon = st.number_input(
        "Forecast horizon (steps)",
        min_value=1,
        max_value=1000,
        value=12,
    )

if uploaded is not None and uploaded.size > 10 * 1024 * 1024:
    st.error("File too large. Limit is 10 MB.")
    st.stop()

if uploaded is None:
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
        st.error(str(e)); st.stop()

with st.expander("Preview"):
    st.dataframe(df.head(20))
auto_date, auto_target = infer_date_and_target(df)

with st.sidebar:
    st.subheader("Select columns")
    date_col = st.selectbox(
        "Date/time column",
        df.columns.tolist(),
        index=(df.columns.get_loc(auto_date) if auto_date in df.columns else 0),
    )
    candidates = [c for c in df.columns if c != date_col]
    target_col = st.selectbox(
        "Target (numeric)",
        candidates,
        index=(candidates.index(auto_target) if auto_target in candidates else 0),
    )
    interval_full = detect_interval(df[date_col])
    human = interval_full.split("(")[-1].strip(") ") if "(" in interval_full else interval_full
    st.caption(f"Detected interval: {human.capitalize()}")

try:
    out = forecast_linear_safe(df, date_col, target_col, int(horizon))
    st.subheader("Forecast")
    st.line_chart(out.set_index("date")[["y", "yhat"]])
    with st.expander("Details"):
        st.dataframe(out.tail(min(50, len(out))))
    st.download_button(
        "Download predictions",
        out.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )
except DataError as e:
    st.error(str(e))
except Exception as e:
    st.error(f"Unexpected error: {e}")

st.caption("Baseline uses scikit-learn LinearRegression with a safe fallback to last value if modeling fails.")


deploy_time = get_deploy_time()
st.markdown(
    f"<div style='position: fixed; bottom: 0; left: 50%; transform: translateX(-50%); font-size:0.75rem; color: gray;'>Deploy: {deploy_time}</div>",
    unsafe_allow_html=True,
)
