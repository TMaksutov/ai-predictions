import os, sys, io
import pandas as pd, numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ts_core import forecast_linear_safe, load_table, DataError

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

