from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class DataError(ValueError):
    """Raised for user-facing, recoverable data errors."""

def load_table(uploaded_file) -> pd.DataFrame:
    """
    Load CSV or Excel from a file-like object (Streamlit uploader).
    Supports .csv and .xlsx. For legacy .xls, instruct user to convert to .xlsx.
    """
    if uploaded_file is None:
        raise DataError("No file provided.")
    name = (getattr(uploaded_file, "name", "") or "").lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        elif name.endswith(".xls"):
            raise DataError("Legacy .xls not supported. Please upload .xlsx or .csv.")
        else:
            # Try CSV as fallback if extension missing
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        raise DataError(f"Failed to read file: {e}")
    if df is None or df.empty:
        raise DataError("File is empty.")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def infer_date_and_target(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    date_col, target_col = None, None
    for c in df.columns:
        if pd.to_datetime(df[c], errors="coerce").notna().sum() >= max(3, int(0.2 * len(df))):
            date_col = c
            break
    for c in df.columns:
        if c != date_col and pd.api.types.is_numeric_dtype(df[c]):
            target_col = c
            break
    return date_col, target_col

def _prepare(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    s = df.copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s[target_col] = pd.to_numeric(s[target_col], errors="coerce")
    s = s[[date_col, target_col]].dropna().sort_values(date_col)
    s = s[~s[date_col].duplicated(keep="last")]
    if len(s) < 3:
        raise DataError("Not enough clean rows (need ≥3).")
    return s

def _infer_offset(dates: pd.Series):
    try:
        freq = pd.infer_freq(dates)
        if freq:
            return pd.tseries.frequencies.to_offset(freq)
    except Exception:
        pass
    diffs = dates.diff().dropna()
    if len(diffs) and hasattr(diffs.iloc[0], "days"):
        days = int(max(1, diffs.median().days))
        return pd.DateOffset(days=days)
    return pd.DateOffset(days=1)

def forecast_linear_safe(df: pd.DataFrame, date_col: str, target_col: str, horizon: int) -> pd.DataFrame:
    """
    Linear regression on integer time with robust fallback to "last value" if anything fails.
    Returns columns: date, y, yhat, kind (historical/forecast).
    """
    s = _prepare(df, date_col, target_col)
    n = len(s)
    X = np.arange(n).reshape(-1, 1)
    y = s[target_col].to_numpy(dtype=float)

    def _naive():
        off = _infer_offset(s[date_col])
        future = [s[date_col].iloc[-1] + off * (i + 1) for i in range(horizon)]
        return pd.DataFrame({
            "date": pd.concat([s[date_col], pd.Series(future)], ignore_index=True),
            "y": pd.concat([s[target_col], pd.Series([np.nan]*horizon)], ignore_index=True),
            "yhat": list(s[target_col].astype(float)) + [float(s[target_col].iloc[-1])] * horizon,
            "kind": ["historical"]*n + ["forecast"]*horizon
        })

    try:
        if horizon < 1 or horizon > 10000 or not np.isfinite(y).all():
            raise DataError("Invalid horizon or values.")
        model = LinearRegression().fit(X, y)
        fut = np.arange(n, n + horizon).reshape(-1, 1)
        yhat = model.predict(fut).astype(float)
        hist_min, hist_max = float(np.nanmin(y)), float(np.nanmax(y))
        span = max(1.0, hist_max - hist_min)
        yhat = np.clip(yhat, hist_min - 5*span, hist_max + 5*span)
        off = _infer_offset(s[date_col])
        future = [s[date_col].iloc[-1] + off * (i + 1) for i in range(horizon)]
        return pd.DataFrame({
            "date": pd.concat([s[date_col], pd.Series(future)], ignore_index=True),
            "y": pd.concat([s[target_col], pd.Series([np.nan]*horizon)], ignore_index=True),
            "yhat": list(s[target_col].astype(float)) + list(yhat),
            "kind": ["historical"]*n + ["forecast"]*horizon
        })
    except Exception:
        return _naive()

