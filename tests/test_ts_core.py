import os, sys, io
import pandas as pd, numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ts_core import forecast_linear_safe, load_table, detect_interval, DataError

def _make_df(n=20, start="2023-01-01"):
    dates = pd.date_range(start=start, periods=n, freq="D")
    y = np.arange(n) + 10
    return pd.DataFrame({"date": dates, "value": y})

def test_linear_trend_increases():
    df = _make_df()
    out = forecast_linear_safe(df, "date", "value", horizon=5)
    assert set(out.columns) == {"date","y","yhat","kind"}
    assert (out["kind"].tail(5) == "forecast").all()
    assert out["yhat"].iloc[-1] > out["yhat"].iloc[-5]

def test_fallback_for_small_sample():
    df = _make_df(n=2)
    out = forecast_linear_safe(df, "date", "value", horizon=3)
    last = df["value"].iloc[-1]
    assert np.allclose(out["yhat"].tail(3).to_numpy(), last)

def test_load_table_csv():
    csv = _make_df().to_csv(index=False).encode("utf-8")
    f = io.BytesIO(csv); f.name = "demo.csv"
    df = load_table(f)
    assert not df.empty

def test_load_table_no_header():
    csv = _make_df().to_csv(index=False, header=False).encode("utf-8")
    f = io.BytesIO(csv); f.name = "demo.csv"
    df = load_table(f)
    assert list(df.columns) == ["col0", "col1"]

def test_load_table_semicolon():
    df0 = _make_df()
    csv = df0.to_csv(index=False, sep=';').encode('utf-8')
    f = io.BytesIO(csv); f.name = 'demo.txt'
    df = load_table(f)
    df['date'] = pd.to_datetime(df['date'])
    pd.testing.assert_frame_equal(df, df0)

def test_subday_frequency_offset():
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=2, freq="h"),
        "value": [1, 2],
    })
    out = forecast_linear_safe(df, "date", "value", horizon=2)
    diffs = out["date"].diff().dropna()
    assert (diffs == pd.Timedelta(hours=1)).all()

def test_detect_interval():
    df = _make_df()
    interval = detect_interval(df['date'])
    assert 'D' in interval

def test_invalid_horizon():
    df = _make_df()
    with pytest.raises(DataError):
        forecast_linear_safe(df, "date", "value", horizon=0)
    
    with pytest.raises(DataError):
        forecast_linear_safe(df, "date", "value", horizon=1001)

def test_data_error_handling():
    with pytest.raises(DataError):
        load_table(None)
    
    empty_csv = "".encode("utf-8")
    f = io.BytesIO(empty_csv); f.name = "empty.csv"
    with pytest.raises(DataError):
        load_table(f)

