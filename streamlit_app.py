import streamlit as st
import pandas as pd
from ts_core import load_table, infer_date_and_target, forecast_linear_safe, DataError

st.set_page_config(page_title="Simple Time-Series Predictor", page_icon="⏱️", layout="wide")
st.title("⏱️ Simple Time-Series Predictor (Baseline)")
st.write("Upload CSV/XLSX, choose columns, and get a small baseline forecast with strong safety checks.")

with st.sidebar:
    uploaded = st.file_uploader("CSV or Excel (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])
    horizon = st.number_input("Forecast horizon (steps)", min_value=1, max_value=1000, value=12)

if uploaded is None:
    st.info("Upload a file to begin.")
    st.stop()

try:
    df = load_table(uploaded)
except DataError as e:
    st.error(str(e)); st.stop()

st.subheader("Preview"); st.dataframe(df.head(20))
auto_date, auto_target = infer_date_and_target(df)

st.subheader("Select columns")
date_col = st.selectbox("Date/time column", df.columns.tolist(),
                        index=(df.columns.get_loc(auto_date) if auto_date in df.columns else 0))
candidates = [c for c in df.columns if c != date_col]
target_col = st.selectbox("Target (numeric)", candidates,
                          index=(candidates.index(auto_target) if auto_target in candidates else 0))

if st.button("Run baseline forecast"):
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

st.caption("Baseline uses scikit-learn LinearRegression with a safe fallback to last value if modeling fails. For .xls, please upload .xlsx/.csv.")
