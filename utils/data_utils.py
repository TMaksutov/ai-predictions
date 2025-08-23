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
    if raw_df.empty or file_info.get("error"):
        return pd.DataFrame(), {}

    orig_cols = list(raw_df.columns)
    if len(orig_cols) < 2:
        return pd.DataFrame(), {}

    first_col = orig_cols[0]
    last_col = orig_cols[-1]

    # Convert to datetime and numeric
    ds_series = pd.to_datetime(raw_df[first_col], errors="coerce")
    y_series = pd.to_numeric(raw_df[last_col], errors="coerce")

    # Create standardized series
    series = pd.DataFrame({
        "ds": ds_series,
        "y": y_series,
    })

    # Add any intermediate feature columns
    for c in orig_cols[1:-1]:
        if c not in ("ds", "y"):
            series[c] = raw_df[c]

    series = series.sort_values("ds").reset_index(drop=True)

    load_meta = {
        "original_time_col": first_col,
        "original_target_col": last_col,
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
