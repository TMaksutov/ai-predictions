from typing import List
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Simple TS Benchmark", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
METRIC_NAME = "Regression RMSE"


# -----------------------------
# Utilities
# -----------------------------
def list_csv_files(directory: Path) -> List[Path]:
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])


def count_rows_in_csv(file_path: Path) -> int:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return 0


def load_series_from_csv(file_path: Path) -> pd.DataFrame:
    """
    Assumes first column is timestamp and last column is target.
    Renames to (ds, y), coerces types, sorts by ds.
    """
    df = pd.read_csv(file_path)
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least two columns: time and target")

    first_col = df.columns[0]
    last_col = df.columns[-1]
    sub = df[[first_col, last_col]].copy()
    sub = sub.rename(columns={first_col: "ds", last_col: "y"})
    sub["ds"] = pd.to_datetime(sub["ds"], errors="coerce")
    sub["y"] = pd.to_numeric(sub["y"], errors="coerce")
    sub = sub.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
    return sub


def forecast_with_model(series_df: pd.DataFrame, test_fraction: float = 0.2):
    """
    Your model should return at least:
      - forecast_df with columns ['ds','yhat'] (predictions for every ds in series_df, including test)
      - test_df with columns ['ds','y'] (ground truth for the test split)
    """
    from models.regression_model import forecast_regression
    # keeping the original behavior but we won't use the returned nRMSE
    _nrmse, forecast_df, test_df = forecast_regression(series_df, test_fraction=test_fraction)
    return forecast_df, test_df


def compute_rmse(forecast_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    merged = pd.merge(test_df[["ds", "y"]], forecast_df[["ds", "yhat"]], on="ds", how="inner")
    if merged.empty:
        raise ValueError("No overlapping timestamps between forecast and test sets to compute RMSE.")
    return float(((merged["y"] - merged["yhat"]) ** 2).mean() ** 0.5)


# -----------------------------
# UI
# -----------------------------
st.markdown("### Daily Regression Benchmark (single)")

csv_files = list_csv_files(DATA_DIR)
if not csv_files:
    st.info(f"No CSV files found in `{DATA_DIR}`. Add some time series CSVs (time in first column, target in last).")
    st.stop()

index_names = [p.name for p in csv_files]
rows_counts = [count_rows_in_csv(p) for p in csv_files]

summary_df = pd.DataFrame(
    {"# Rows": rows_counts, METRIC_NAME: [None] * len(index_names)},
    index=index_names,
)

left, right = st.columns([1, 2], gap="large")

with left:
    st.caption("Choose a dataset to run the benchmark:")
    selected_name = st.radio("Datasets", options=index_names, index=0, label_visibility="collapsed")

with right:
    # Run immediately on selection
    selected_path = DATA_DIR / selected_name
    try:
        series = load_series_from_csv(selected_path)
        if len(series) < 30:
            st.error("Insufficient data (need at least 30 rows) in the selected file.")
        else:
            forecast_df, test_df = forecast_with_model(series, test_fraction=0.2)
            rmse = compute_rmse(forecast_df, test_df)

            # Update and show summary table with this one benchmark result
            out_df = summary_df.copy()
            out_df.loc[selected_name, METRIC_NAME] = rmse
            st.dataframe(out_df, use_container_width=True)

            # Plot actual vs forecast
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(series["ds"], series["y"], label="Actual", linewidth=2, alpha=0.85)
            ax.plot(forecast_df["ds"], forecast_df["yhat"], linestyle="--", linewidth=2, label="Forecast")
            if len(test_df) > 0:
                ax.axvline(test_df["ds"].iloc[0], linestyle=":", linewidth=1, label="Train/Test split")
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.set_title(f"{selected_name} — Regression ({METRIC_NAME} = {rmse:.4f})")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Failed to run forecast on '{selected_name}': {e}")
