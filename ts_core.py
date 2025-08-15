from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class DataError(ValueError):
    """Raised for user-facing, recoverable data errors."""

def load_table(uploaded_file) -> pd.DataFrame:
    """Load user data from CSV/Excel safely."""
    if uploaded_file is None:
        raise DataError("No file provided.")
    
    name = (getattr(uploaded_file, "name", "") or "").lower()
    
    try:
        uploaded_file.seek(0)
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file, engine="openpyxl" if name.endswith(".xlsx") else "xlrd")
        else:
            df = pd.read_csv(uploaded_file, sep=None, engine="python", on_bad_lines="skip")
        
        # Clean column names
        df.columns = [str(c).strip().strip('"\'') for c in df.columns]
        
        # Handle files without headers
        if list(df.columns) == list(range(df.shape[1])):
            df.columns = [f"col{i}" for i in range(df.shape[1])]
            
    except Exception as e:
        raise DataError(f"Failed to read file: {e}")
    
    if df.empty:
        raise DataError("File is empty.")
    
    return df

def infer_date_and_target(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Auto-detect date and target columns."""
    date_col = None
    target_col = None
    
    # Find date column
    for col in df.columns:
        if pd.to_datetime(df[col], errors="coerce").notna().sum() >= max(3, int(0.2 * len(df))):
            date_col = col
            break
    
    # Find numeric target column
    for col in df.columns:
        if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
            target_col = col
            break
    
    return date_col, target_col

def _prepare_data(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    """Prepare data for forecasting."""
    df_clean = df.copy()
    df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors="coerce")
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors="coerce")
    
    # Remove duplicates and sort
    df_clean = df_clean[[date_col, target_col]].dropna().sort_values(date_col)
    df_clean = df_clean[~df_clean[date_col].duplicated(keep="last")]
    
    return df_clean

def detect_interval(dates: pd.Series) -> str:
    """Detect time series interval using pandas."""
    dates_clean = pd.to_datetime(dates, errors="coerce").dropna()
    
    if len(dates_clean) < 2:
        return "unknown"
    
    # Use pandas built-in frequency inference
    try:
        freq = pd.infer_freq(dates_clean)
        if freq:
            return freq
    except:
        pass
    
    # Fallback to median difference
    diffs = dates_clean.diff().dropna()
    if len(diffs) > 0:
        median_diff = diffs.median()
        if hasattr(median_diff, "days"):
            days = int(max(1, median_diff.days))
            return f"{days}D"
    
    return "1D"

def forecast_linear_safe(df: pd.DataFrame, date_col: str, target_col: str, horizon: int) -> pd.DataFrame:
    """Simple linear regression forecast with fallback."""
    if horizon < 1 or horizon > 1000:
        raise DataError("Invalid horizon. Must be between 1 and 1000.")
    
    df_clean = _prepare_data(df, date_col, target_col)
    
    if len(df_clean) < 3:
        raise DataError("Not enough data for forecasting.")
    
    # Prepare features
    X = np.arange(len(df_clean)).reshape(-1, 1)
    y = df_clean[target_col].values
    
    try:
        # Fit linear model
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate future dates
        last_date = df_clean[date_col].iloc[-1]
        interval = detect_interval(df_clean[date_col])
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=interval)[1:]
        
        # Predict
        future_X = np.arange(len(df_clean), len(df_clean) + horizon).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        # Clip predictions to reasonable range
        y_min, y_max = y.min(), y.max()
        y_range = max(1.0, y_max - y_min)
        predictions = np.clip(predictions, y_min - 2*y_range, y_max + 2*y_range)
        
        # Create result dataframe
        result = pd.DataFrame({
            "date": pd.concat([df_clean[date_col], pd.Series(future_dates)]),
            "y": pd.concat([df_clean[target_col], pd.Series([np.nan] * horizon)]),
            "yhat": pd.concat([df_clean[target_col], pd.Series(predictions)]),
            "kind": ["historical"] * len(df_clean) + ["forecast"] * horizon
        })
        
        return result
        
    except Exception:
        # Fallback to naive forecast (last value)
        last_date = df_clean[date_col].iloc[-1]
        last_value = df_clean[target_col].iloc[-1]
        interval = detect_interval(df_clean[date_col])
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=interval)[1:]
        
        result = pd.DataFrame({
            "date": pd.concat([df_clean[date_col], pd.Series(future_dates)]),
            "y": pd.concat([df_clean[target_col], pd.Series([np.nan] * horizon)]),
            "yhat": pd.concat([df_clean[target_col], pd.Series([last_value] * horizon)]),
            "kind": ["historical"] * len(df_clean) + ["forecast"] * horizon
        })
        
        return result

