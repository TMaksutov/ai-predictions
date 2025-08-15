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
            # Try to detect if there's a header by checking if first row looks like data
            df_no_header = pd.read_csv(uploaded_file, sep=None, engine="python", on_bad_lines="skip", header=None)
            
            # Check if first row looks like data (contains dates or numbers)
            first_row = df_no_header.iloc[0]
            looks_like_data = False
            
            for val in first_row:
                try:
                    pd.to_datetime(val, errors='raise')
                    looks_like_data = True
                    break
                except:
                    try:
                        float(val)
                        looks_like_data = True
                        break
                    except:
                        continue
            
            if looks_like_data:
                # No header, use default column names
                df = df_no_header
                df.columns = [f"col{i}" for i in range(df.shape[1])]
            else:
                # Has header, read normally
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=None, engine="python", on_bad_lines="skip")
        
        # Clean column names
        df.columns = [str(c).strip().strip('"\'') for c in df.columns]
            
    except Exception as e:
        raise DataError(f"Failed to read file: {e}")
    
    if df.empty:
        raise DataError("File is empty.")
    
    return df

def infer_date_and_target(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Auto-detect date and target columns."""
    date_col = None
    target_col = None
    
    # Find date column - must have at least 3 valid dates and be mostly valid
    for col in df.columns:
        try:
            parsed_dates = pd.to_datetime(df[col], errors="coerce")
            valid_dates = parsed_dates.notna().sum()
            if valid_dates >= max(3, int(0.2 * len(df))) and valid_dates >= 3:
                # Additional check: ensure the column name suggests it's a date
                col_lower = col.lower()
                if any(date_word in col_lower for date_word in ['date', 'time', 'timestamp', 'when', 'day']):
                    date_col = col
                    break
        except:
            continue
    
    # Find numeric target column - prefer columns that look like values
    target_col = None
    for col in df.columns:
        if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
            # Prefer columns that don't look like IDs
            col_lower = col.lower()
            if not any(id_word in col_lower for id_word in ['id', 'index', 'key', 'pk']):
                target_col = col
                break
    
    # If no preferred target found, take the first numeric column
    if target_col is None:
        for col in df.columns:
            if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
                target_col = col
                break
    
    # If no date column found, don't return the first numeric column as date
    if date_col is None:
        return None, target_col
    
    return date_col, target_col

def _prepare_data(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    """Prepare data for forecasting."""
    # Check if required columns exist
    if date_col not in df.columns:
        raise DataError(f"Date column '{date_col}' not found in data")
    if target_col not in df.columns:
        raise DataError(f"Target column '{target_col}' not found in data")
    
    df_clean = df.copy()
    df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors="coerce")
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors="coerce")
    
    # Remove duplicates and sort
    df_clean = df_clean[[date_col, target_col]].dropna().sort_values(date_col)
    df_clean = df_clean[~df_clean[date_col].duplicated(keep="last")]
    
    # Check if we have any valid data after cleaning
    if len(df_clean) == 0:
        raise DataError("No valid data remaining after cleaning")
    
    return df_clean

def detect_interval(dates: pd.Series) -> str:
    """Detect time series interval using pandas."""
    dates_clean = pd.to_datetime(dates, errors="coerce").dropna()
    
    if len(dates_clean) < 2:
        return "1D"  # Default to daily
    
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
        # Handle different time units
        if hasattr(median_diff, "total_seconds"):
            total_seconds = median_diff.total_seconds()
            if total_seconds < 60:  # Less than 1 minute
                return "s"  # seconds (lowercase)
            elif total_seconds < 3600:  # Less than 1 hour
                minutes = int(total_seconds / 60)
                return f"{minutes}min"  # minutes
            elif total_seconds < 86400:  # Less than 1 day
                hours = int(total_seconds / 3600)
                return f"{hours}h"  # hours
            else:
                days = int(total_seconds / 86400)
                return f"{days}D"  # days
    
    return "1D"

def forecast_linear_safe(df: pd.DataFrame, date_col: str, target_col: str, horizon: int) -> pd.DataFrame:
    """Simple linear regression forecast with fallback."""
    if horizon < 1 or horizon > 1000:
        raise DataError("Invalid horizon. Must be between 1 and 1000.")
    
    # Check if dataframe is empty
    if df.empty:
        raise DataError("DataFrame is empty")
    
    df_clean = _prepare_data(df, date_col, target_col)
    
    # Handle small datasets with fallback
    if len(df_clean) < 3:
        # Use naive forecast for very small datasets
        last_date = df_clean[date_col].iloc[-1]
        last_value = df_clean[target_col].iloc[-1]
        interval = detect_interval(df_clean[date_col])
        
        try:
            future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=interval)[1:]
        except ValueError:
            # Fallback to daily frequency if interval is invalid
            future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq="1D")[1:]
        
        result = pd.DataFrame({
            "date": pd.concat([df_clean[date_col], pd.Series(future_dates)]),
            "y": pd.concat([df_clean[target_col], pd.Series([np.nan] * horizon)]),
            "yhat": pd.concat([df_clean[target_col], pd.Series([last_value] * horizon)]),
            "kind": ["historical"] * len(df_clean) + ["forecast"] * horizon
        })
        
        return result
    
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

