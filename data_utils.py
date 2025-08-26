"""
Common data processing utilities shared across the application.
"""

from typing import List, Optional

import pandas as pd
from pathlib import Path


def load_default_dataset(default_path: Path):
    """Load the default dataset if available."""
    if default_path.exists():
        from data_io import load_data_with_checklist
        return load_data_with_checklist(default_path)
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


# Shared date utilities
SUPPORTED_DATE_FORMATS: List[str] = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%Y.%m.%d",
    "%d.%m.%Y",
]


def detect_datetime_format(sample: List[str], max_samples: int = 200) -> Optional[str]:
    """Return the first format that parses all non-empty samples without NaT."""
    vals: List[str] = [str(v).strip() for v in sample if str(v).strip()]
    if not vals:
        return None
    vals = vals[:max_samples]
    for fmt in SUPPORTED_DATE_FORMATS:
        try:
            parsed = pd.to_datetime(vals, format=fmt, errors="coerce")
        except Exception:
            continue
        if parsed.notna().all():
            return fmt
    return None


def _strip_whitespace_df(df: pd.DataFrame) -> pd.DataFrame:
    try:
        out = df.copy()
        for col in out.columns:
            if out[col].dtype == object:
                out[col] = out[col].astype(str).map(lambda x: x.strip())
        return out
    except Exception:
        return df


def _standardize_missing_tokens_df(df: pd.DataFrame) -> pd.DataFrame:
    tokens = {"", "na", "n/a", "nan", "null", "none", "-", "?", "—"}
    out = df.copy()
    try:
        for col in out.columns:
            if out[col].dtype == object:
                out[col] = out[col].astype(str)
                out[col] = out[col].map(lambda x: pd.NA if x.strip().lower() in tokens else x)
        return out
    except Exception:
        return df


def _normalize_dates_to_day(series: pd.Series, date_format: Optional[str] = None) -> pd.Series:
    try:
        if date_format:
            parsed = pd.to_datetime(series, errors="coerce", format=date_format)
        else:
            parsed = None
            # First try pandas mixed parser
            try:
                parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            except Exception:
                parsed = None
            # If still missing values or parsing failed, try day-first heuristic
            if parsed is None or parsed.isna().any():
                try:
                    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
                except Exception:
                    parsed = None
            # Final plain fallback
            if parsed is None:
                parsed = pd.to_datetime(series, errors="coerce")
        return parsed.dt.normalize()
    except Exception:
        return pd.to_datetime([pd.NA] * len(series), errors="coerce")


