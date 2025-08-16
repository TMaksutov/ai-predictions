from typing import Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def _infer_seasonal_period(series_df: pd.DataFrame) -> int:
    freq = pd.infer_freq(series_df['ds'])
    if freq == 'H':
        return 24
    if freq in ('D', 'B'):
        return 7
    if freq in ('W',):
        return 52
    if freq in ('M', 'MS'):
        return 12
    return 0


def forecast_and_nrmse(series_df: pd.DataFrame, test_fraction: float = 0.2) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Minimal Holt-Winters forecast with additive components if seasonal period is inferred.
    Returns (nrmse, forecast_df, test_df).
    """
    test_size = int(len(series_df) * test_fraction)
    train_df = series_df.iloc[:-test_size].copy()
    test_df = series_df.iloc[-test_size:].copy()

    m = _infer_seasonal_period(series_df)
    seasonal = 'add' if m and m > 1 else None
    seasonal_periods = m if m and m > 1 else None

    model = ExponentialSmoothing(
        train_df['y'],
        trend='add',
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
    )
    fit = model.fit(optimized=True)
    yhat = fit.forecast(test_size)

    forecast_df = pd.DataFrame({
        'ds': test_df['ds'].to_numpy(),
        'yhat': yhat.to_numpy(),
        'yhat_lower': yhat.to_numpy(),
        'yhat_upper': yhat.to_numpy(),
    })

    y_true = test_df['y'].to_numpy()
    rmse = float(np.sqrt(np.mean((y_true - forecast_df['yhat'].to_numpy()) ** 2)))
    y_range = np.max(y_true) - np.min(y_true)
    nrmse = rmse / y_range if y_range > 0 else rmse
    return nrmse, forecast_df, test_df