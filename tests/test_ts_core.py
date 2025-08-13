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

def test_load_table_semicolon_and_xls():
    df0 = _make_df()
    csv = df0.to_csv(index=False, sep=';').encode('utf-8')
    f = io.BytesIO(csv); f.name = 'demo.txt'
    df = load_table(f)
    df['date'] = pd.to_datetime(df['date'])
    pd.testing.assert_frame_equal(df, df0)
    pytest.importorskip('xlwt')
    pytest.importorskip('xlrd')
    f2 = io.BytesIO()
    with pd.ExcelWriter(f2, engine='xlwt') as writer:
        df0.to_excel(writer, index=False)
    f2.seek(0); f2.name = 'demo.xls'
    df_xls = load_table(f2)
    df_xls['date'] = pd.to_datetime(df_xls['date'])
    pd.testing.assert_frame_equal(df_xls, df0)

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

