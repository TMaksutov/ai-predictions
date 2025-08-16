from typing import Tuple
import numpy as np
import pandas as pd
from autots import AutoTS


def forecast_and_nrmse(series_df: pd.DataFrame, test_fraction: float = 0.2) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Minimal AutoTS setup restricted to top 10 models. Returns (nrmse, forecast_df, test_df).
    """
    n = len(series_df)
    if n < 30:
        raise ValueError("Insufficient data for AutoTS forecasting")

    test_size = max(1, int(n * test_fraction))
    train_df = series_df.iloc[:-test_size].copy().reset_index(drop=True)
    test_df = series_df.iloc[-test_size:].copy().reset_index(drop=True)

    # Restrict to 10 commonly used AutoTS models
    top10_models = [
        'AverageValueNaive',
        'LastValueNaive',
        'SeasonalNaive',
        'GLM',
        'ETS',
        'ARIMA',
        'Theta',
        'DatepartRegression',
        'WindowRegression',
        'UnivariateMotif',
    ]

    model = AutoTS(
        forecast_length=test_size,
        frequency='infer',
        ensemble=None,
        model_list=top10_models,
        num_validations=0,
        random_seed=42,
    )
    fitted = model.fit(train_df, date_col='ds', value_col='y', id_col=None)
    prediction = fitted.predict()
    fcst_df = prediction.forecast
    # Extract the first column as forecast values
    col = fcst_df.columns[0]
    yhat = fcst_df[col].to_numpy()

    if len(yhat) != len(test_df):
        raise RuntimeError("Forecast length does not match test set length")

    forecast_df = pd.DataFrame({
        'ds': test_df['ds'].to_numpy(),
        'yhat': yhat,
        'yhat_lower': yhat,
        'yhat_upper': yhat,
    })

    y_true = test_df['y'].to_numpy()
    rmse = float(np.sqrt(np.mean((y_true - forecast_df['yhat'].to_numpy()) ** 2)))
    y_range = np.max(y_true) - np.min(y_true)
    nrmse = rmse / y_range if y_range > 0 else rmse
    return nrmse, forecast_df, test_df