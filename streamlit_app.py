import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from prophet import Prophet
import tsdl


st.set_page_config(page_title="Monash Prophet Benchmark", layout="wide")
st.title("Prophet Benchmark — Monash (first 10 datasets)")
st.caption("RMSE on last 20% holdout. Select a dataset to view its forecast plot.")


def _first_10_dataset_names() -> List[str]:
    names = tsdl.datasets()
    # Normalize to a list of strings if the library returns richer structures
    try:
        if isinstance(names, pd.DataFrame):
            if "name" in names.columns:
                names = names["name"].astype(str).tolist()
            elif "dataset" in names.columns:
                names = names["dataset"].astype(str).tolist()
            else:
                # Fallback: try first column
                names = names.iloc[:, 0].astype(str).tolist()
        elif isinstance(names, (list, tuple)):
            if names and not isinstance(names[0], str):
                try:
                    names = [n.get("name", str(n)) for n in names]
                except Exception:
                    names = [str(n) for n in names]
        else:
            names = [str(n) for n in list(names)]
    except Exception:
        names = [str(n) for n in list(names)]
    return list(names)[:10]


@st.cache_data(show_spinner=False)
def _load_dataset(name: str) -> Tuple[pd.DataFrame, dict]:
    data, metadata = tsdl.load(name)
    # Some tsdl versions return (data, metadata) where data may be a dict-like of series
    # Normalize to a single DataFrame
    if isinstance(data, dict):
        # Try to concatenate dict of series/frames
        try:
            frames = []
            for key, val in data.items():
                if isinstance(val, pd.Series):
                    f = val.reset_index()
                    f.columns = ["ds", "y"]
                else:
                    f = pd.DataFrame(val)
                    if f.shape[1] >= 2:
                        f = f.iloc[:, :2]
                        f.columns = ["ds", "y"]
                f["unique_id"] = str(key)
                frames.append(f)
            data = pd.concat(frames, ignore_index=True)
        except Exception:
            data = pd.DataFrame(data)
    if isinstance(data, pd.Series):
        data = data.reset_index()
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    return data.copy(), dict(metadata or {})


def _prepare_single_series(df: pd.DataFrame) -> pd.DataFrame:
    # If multiple series are present, keep the first one
    if "unique_id" in df.columns:
        first_id = str(df["unique_id"].astype(str).iloc[0])
        df = df[df["unique_id"].astype(str) == first_id].copy()

    # If already in Prophet format
    if set(["ds", "y"]).issubset(df.columns):
        sub = df[["ds", "y"]].copy()
    else:
        # Try to infer simple two-column format (date + value)
        if df.shape[1] >= 2:
            sub = df.iloc[:, :2].copy()
            sub.columns = ["ds", "y"]
        else:
            raise ValueError("Dataset does not contain enough columns to infer a time series.")

    # Coerce types and clean
    sub["ds"] = pd.to_datetime(sub["ds"], errors="coerce")
    sub["y"] = pd.to_numeric(sub["y"], errors="coerce")
    sub = sub.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)

    # Deduplicate by timestamp
    sub = sub.groupby("ds", as_index=False).agg({"y": "mean"})

    return sub


def _compute_prophet_forecast_and_rmse(series_df: pd.DataFrame, test_fraction: float = 0.2):
    n = len(series_df)
    if n < 10:
        raise ValueError("Series too short for 20% holdout evaluation.")

    test_size = max(1, int(math.ceil(n * test_fraction)))
    train_df = series_df.iloc[:-test_size].copy()
    test_df = series_df.iloc[-test_size:].copy()

    model = Prophet()
    model.fit(train_df)

    # Predict exactly on observed timestamps to avoid frequency assumptions
    future = pd.DataFrame({"ds": series_df["ds"]})
    forecast = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    yhat_test = forecast["yhat"].iloc[-test_size:].to_numpy()
    y_true = test_df["y"].to_numpy()
    rmse = float(np.sqrt(np.mean((yhat_test - y_true) ** 2)))

    return rmse, forecast, test_df


@st.cache_data(show_spinner=False)
def _compute_benchmark(dataset_names: Tuple[str, ...]) -> pd.DataFrame:
    results = []
    for name in dataset_names:
        try:
            raw_df, _ = _load_dataset(name)
            series_df = _prepare_single_series(raw_df)
            rmse, _, _ = _compute_prophet_forecast_and_rmse(series_df, test_fraction=0.2)
            results.append({"Dataset": name, "RMSE": rmse})
        except Exception as e:
            results.append({"Dataset": name, "RMSE": np.nan, "Error": str(e)})
    return pd.DataFrame(results)


# Load dataset names (first 10)
dataset_names = _first_10_dataset_names()
if not dataset_names:
    st.error("No datasets available from tsdl.")
    st.stop()

# Benchmark table
st.subheader("Benchmark (Prophet RMSE on last 20%)")
with st.spinner("Computing benchmark for first 10 datasets..."):
    bench_df = _compute_benchmark(tuple(dataset_names))

st.dataframe(bench_df[[c for c in ["Dataset", "RMSE"] if c in bench_df.columns]], use_container_width=True)

# Selection and plot
st.subheader("Forecast plot for selected dataset (last 20% only)")
selected = st.selectbox("Select dataset", dataset_names, index=0)

if selected:
    with st.spinner(f"Loading and forecasting: {selected}"):
        raw_df, _ = _load_dataset(selected)
        series_df = _prepare_single_series(raw_df)
        rmse, forecast, test_df = _compute_prophet_forecast_and_rmse(series_df, test_fraction=0.2)

    st.caption(f"RMSE: {rmse:.4f}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series_df["ds"], series_df["y"], color="#333333", label="Actual", linewidth=2)

    # Only show forecast line for the test segment
    test_pred = forecast.tail(len(test_df))
    ax.plot(test_pred["ds"], test_pred["yhat"], color="#1f77b4", linestyle="--", linewidth=2, label="Prophet forecast (test)")

    # Split marker
    ax.axvline(test_df["ds"].iloc[0], color="#888888", linestyle=":", label="Train/Test split")

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    st.pyplot(fig)
