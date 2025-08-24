"""
Common data processing utilities shared across the application.
"""

import pandas as pd
from pathlib import Path


def load_default_dataset(default_path: Path):
    """Load the default dataset if available."""
    if default_path.exists():
        from data_io import read_table_any
        return read_table_any(default_path)
    return pd.DataFrame(), {"error": f"Default data file not found: {default_path}"}


def prepare_series_from_dataframe(raw_df: pd.DataFrame, file_info: dict):
    """
    Convert raw CSV dataframe to standardized time series format.
    Returns (series_df, load_meta) where series has 'ds' and 'y' columns.
    """
    if raw_df.empty or file_info.get("error") or raw_df.shape[1] < 2:
        return pd.DataFrame(), {}

    first, last = raw_df.columns[0], raw_df.columns[-1]
    series = raw_df.copy()
    # Insert standardized columns and preserve intermediate features
    series.insert(0, "ds", pd.to_datetime(series.pop(first), errors="coerce"))
    series["y"] = pd.to_numeric(series.pop(last), errors="coerce")
    series = series.sort_values("ds").reset_index(drop=True)

    load_meta = {
        "original_time_col": first,
        "original_target_col": last,
        "trailing_missing_count": 0,
        "last_known_ds": series["ds"].max() if not series.empty else None,
        "future_rows_raw": pd.DataFrame(),
    }

    return series, load_meta


def get_future_rows(series: pd.DataFrame, feature_cols: list):
    """Extract future rows (rows with missing target values) from series."""
    y_notna = series["y"].notna()
    if not y_notna.any():
        return pd.DataFrame(), -1

    last_observed_idx = int(y_notna[y_notna].index.max())
    if last_observed_idx >= len(series) - 1:
        return pd.DataFrame(), last_observed_idx

    future_rows = series.iloc[last_observed_idx + 1:][["ds"] + feature_cols].copy()
    return future_rows, last_observed_idx
