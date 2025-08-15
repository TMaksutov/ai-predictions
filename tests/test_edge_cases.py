import os, sys, io
import pandas as pd, numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ts_core import forecast_linear_safe, load_table, detect_interval, DataError, infer_date_and_target

def test_empty_dataframe():
    """Test handling of empty dataframes."""
    df = pd.DataFrame(columns=["date", "value"])
    with pytest.raises(DataError):
        forecast_linear_safe(df, "date", "value", horizon=1)

def test_missing_columns():
    """Test handling of missing required columns."""
    df = pd.DataFrame({"date": ["2023-01-01"], "other": [1]})
    with pytest.raises(DataError):
        forecast_linear_safe(df, "date", "value", horizon=1)

def test_all_nan_values():
    """Test handling of all NaN values."""
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "value": [np.nan, np.nan, np.nan]
    })
    with pytest.raises(DataError):
        forecast_linear_safe(df, "date", "value", horizon=1)

def test_single_value():
    """Test forecasting with single data point."""
    df = pd.DataFrame({"date": ["2023-01-01"], "value": [10]})
    out = forecast_linear_safe(df, "date", "value", horizon=2)
    assert len(out) == 3  # 1 historical + 2 forecast
    assert out["kind"].iloc[-2:].tolist() == ["forecast", "forecast"]
    assert out["yhat"].iloc[-2:].tolist() == [10, 10]  # Should repeat last value

def test_duplicate_dates():
    """Test handling of duplicate dates."""
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-01", "2023-01-02"],
        "value": [10, 15, 20]
    })
    out = forecast_linear_safe(df, "date", "value", horizon=1)
    # Should keep last value for duplicate date
    assert len(out) == 3  # 2 historical (after deduplication) + 1 forecast
    assert out["kind"].iloc[-1] == "forecast"
    assert out["yhat"].iloc[-1] == 20  # Should use last historical value

def test_mixed_data_types():
    """Test handling of mixed data types in date column."""
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "invalid_date", "2023-01-04"],
        "value": [10, 20, 30, 40]
    })
    out = forecast_linear_safe(df, "date", "value", horizon=1)
    # Should handle invalid dates gracefully
    assert len(out) == 4  # 3 valid historical + 1 forecast

def test_very_large_horizon():
    """Test handling of large horizon values."""
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "value": [10, 20, 30]
    })
    with pytest.raises(DataError):
        forecast_linear_safe(df, "date", "value", horizon=1001)

def test_negative_horizon():
    """Test handling of negative horizon values."""
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "value": [10, 20, 30]
    })
    with pytest.raises(DataError):
        forecast_linear_safe(df, "date", "value", horizon=0)

def test_infer_date_and_target():
    """Test automatic date and target column detection."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=5),
        "sales": [100, 200, 300, 400, 500],
        "category": ["A", "B", "A", "B", "A"]
    })
    date_col, target_col = infer_date_and_target(df)
    assert date_col == "timestamp"
    assert target_col == "sales"

def test_infer_date_and_target_no_date():
    """Test detection when no date column exists."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "sales": [100, 200, 300, 400, 500]
    })
    date_col, target_col = infer_date_and_target(df)
    assert date_col is None
    assert target_col == "sales"

def test_detect_interval_seconds():
    """Test interval detection for sub-minute data."""
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=3, freq="30s"),
        "value": [1, 2, 3]
    })
    interval = detect_interval(df["date"])
    assert "30s" in interval or "s" in interval

def test_detect_interval_minutes():
    """Test interval detection for minute-level data."""
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=3, freq="15min"),
        "value": [1, 2, 3]
    })
    interval = detect_interval(df["date"])
    assert "15min" in interval or "min" in interval

def test_detect_interval_weekly():
    """Test interval detection for weekly data."""
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=4, freq="W"),
        "value": [1, 2, 3, 4]
    })
    interval = detect_interval(df["date"])
    assert "W" in interval

def test_detect_interval_monthly():
    """Test interval detection for monthly data."""
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=4, freq="ME"),
        "value": [1, 2, 3, 4]
    })
    interval = detect_interval(df["date"])
    assert "M" in interval

def test_load_table_excel_error():
    """Test handling of Excel file read errors."""
    # Create a file that looks like Excel but isn't
    fake_excel = b"PK\x03\x04\x14\x00\x00\x00\x08\x00"  # Fake ZIP header
    f = io.BytesIO(fake_excel)
    f.name = "test.xlsx"
    
    with pytest.raises(DataError):
        load_table(f)

def test_load_table_csv_with_quotes():
    """Test loading CSV with quoted values."""
    csv_data = '"date","value"\n"2023-01-01","10"\n"2023-01-02","20"'
    f = io.BytesIO(csv_data.encode("utf-8"))
    f.name = "test.csv"
    
    df = load_table(f)
    assert len(df) == 2
    assert list(df.columns) == ["date", "value"]

def test_load_table_csv_with_commas_in_values():
    """Test loading CSV with commas inside quoted values."""
    csv_data = '"date","description","value"\n"2023-01-01","Item, with comma","10"'
    f = io.BytesIO(csv_data.encode("utf-8"))
    f.name = "test.csv"
    
    df = load_table(f)
    assert len(df) == 1
    assert df["description"].iloc[0] == "Item, with comma"