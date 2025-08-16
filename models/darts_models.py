from typing import Tuple
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import ExponentialSmoothing


def forecast_and_nrmse(series_df: pd.DataFrame, test_fraction: float = 0.2) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Minimal Darts ExponentialSmoothing forecast. Returns (nrmse, forecast_df, test_df).
    """
    test_size = int(len(series_df) * test_fraction)
    train_df = series_df.iloc[:-test_size].copy().reset_index(drop=True)
    test_df = series_df.iloc[-test_size:].copy().reset_index(drop=True)

    series = TimeSeries.from_dataframe(series_df, time_col='ds', value_cols='y')
    train_series = series[:-test_size]
    test_series = series[-test_size:]

    model = ExponentialSmoothing()
    model.fit(train_series)
    pred_series = model.predict(len(test_series))

    yhat = pred_series.values().flatten()
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