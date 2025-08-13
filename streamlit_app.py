import streamlit as st
import random
from pathlib import Path
import subprocess
from ts_core import load_table, infer_date_and_target, forecast_linear_safe, DataError, detect_interval


def get_deploy_info():
    """Return short git commit hash and commit datetime."""
    try:
        version = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).decode().strip()
        deploy_time = subprocess.check_output(
            ["git", "show", "-s", "--format=%ci", "HEAD"]
        ).decode().strip()
    except Exception:
        version = "unknown"
        deploy_time = "unknown"
    return version, deploy_time

st.set_page_config(page_title="Simple Time-Series Predictor", page_icon="⏱️", layout="wide")
st.title("⏱️ Simple Time-Series Predictor (Baseline)")
st.write("Upload CSV/XLSX, choose columns, and get a small baseline forecast with strong safety checks.")

DATA_DIR = Path(__file__).parent / "test_files"

with st.sidebar:
    if "show_examples" not in st.session_state:
        st.session_state.show_examples = False
    if st.button("Download examples"):
        st.session_state.show_examples = not st.session_state.show_examples
    if st.session_state.show_examples:
        example_files = sorted(DATA_DIR.glob("*"))
        for example in example_files:
            st.download_button(
                label=example.name,
                data=example.read_bytes(),
                file_name=example.name,
                key=f"example-{example.name}"
            )
    uploaded = st.file_uploader("CSV or Excel (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])
    horizon = st.number_input("Forecast horizon (steps)", min_value=1, max_value=1000, value=12)

if uploaded is None:
    sample_files = list(DATA_DIR.glob("*.csv"))
    if not sample_files:
        st.info("Upload a file to begin.")
        st.stop()
    choice = random.choice(sample_files)
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

st.subheader("Select columns")
date_col = st.selectbox("Date/time column", df.columns.tolist(),
                        index=(df.columns.get_loc(auto_date) if auto_date in df.columns else 0))
candidates = [c for c in df.columns if c != date_col]
target_col = st.selectbox("Target (numeric)", candidates,
            index=(candidates.index(auto_target) if auto_target in candidates else 0))

st.caption(f"Detected interval: {detect_interval(df[date_col])}")

try:
    out = forecast_linear_safe(df, date_col, target_col, int(horizon))
    st.subheader("Forecast")
    st.line_chart(out.set_index("date")[["y", "yhat"]])
    st.download_button("Download predictions as CSV",
                       out.to_csv(index=False).encode("utf-8"),
                       file_name="predictions.csv", mime="text/csv")
    with st.expander("Details"):
        st.dataframe(out.tail(min(50, len(out))))
except DataError as e:
    st.error(str(e))
except Exception as e:
    st.error(f"Unexpected error: {e}")

st.caption("Baseline uses scikit-learn LinearRegression with a safe fallback to last value if modeling fails.")


version, deploy_time = get_deploy_info()
st.markdown(
    f"<div style='position: fixed; bottom: 0; right: 0; font-size:0.75rem; color: gray;'>"
    f"Version: {version}<br/>Last deploy: {deploy_time}</div>",
    unsafe_allow_html=True,
)
