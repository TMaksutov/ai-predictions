from typing import Tuple
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import ExponentialSmoothing


def _naive_baseline(series_df: pd.DataFrame, test_fraction: float) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    n = len(series_df)
    test_size = max(1, int(n * test_fraction)) if n > 1 else 1
    train_df = series_df.iloc[:-test_size].copy() if n > test_size else series_df.iloc[:0].copy()
    test_df = series_df.iloc[-test_size:].copy()
    last_value = float(train_df['y'].iloc[-1]) if len(train_df) > 0 else float(series_df['y'].iloc[0])
    yhat = np.full(len(test_df), last_value, dtype=float)
    forecast_df = pd.DataFrame({
        'ds': test_df['ds'].to_numpy(),
        'yhat': yhat,
        'yhat_lower': yhat,
        'yhat_upper': yhat,
    })
    y_true = test_df['y'].to_numpy() if len(test_df) > 0 else np.array([last_value], dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - forecast_df['yhat'].to_numpy()) ** 2)))
    y_range = np.max(y_true) - np.min(y_true) if len(y_true) > 0 else 0.0
    nrmse = rmse / y_range if y_range > 0 else rmse
    return nrmse, forecast_df, test_df


def forecast_and_nrmse(series_df: pd.DataFrame, test_fraction: float = 0.2) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Minimal Darts ExponentialSmoothing forecast. Returns (nrmse, forecast_df, test_df).
    """
    try:
        n = len(series_df)
        if n < 20:
            return _naive_baseline(series_df, test_fraction)
        test_size = max(1, int(n * test_fraction))
        train_df = series_df.iloc[:-test_size].copy().reset_index(drop=True)
        test_df = series_df.iloc[-test_size:].copy().reset_index(drop=True)

        series = TimeSeries.from_dataframe(series_df, time_col='ds', value_cols='y')
        train_series = series[:-test_size]
        test_series = series[-test_size:]

        model = ExponentialSmoothing()
        model.fit(train_series)
        pred_series = model.predict(len(test_series))

        yhat = pred_series.values().flatten()
        if len(yhat) != len(test_df):
            return _naive_baseline(series_df, test_fraction)
        ds_index = pd.DatetimeIndex(pred_series.time_index)

        forecast_df = pd.DataFrame({
            'ds': ds_index.to_pydatetime(),
            'yhat': yhat,
            'yhat_lower': yhat,
            'yhat_upper': yhat,
        })

        y_true = test_df['y'].to_numpy()
        rmse = float(np.sqrt(np.mean((y_true - forecast_df['yhat'].to_numpy()) ** 2)))
        y_range = np.max(y_true) - np.min(y_true)
        nrmse = rmse / y_range if y_range > 0 else rmse
        return nrmse, forecast_df, test_df
    except Exception:
        return _naive_baseline(series_df, test_fraction)