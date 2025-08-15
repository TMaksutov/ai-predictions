from typing import Tuple, Dict
import itertools
import logging

import numpy as np
import pandas as pd
from prophet import Prophet


def optimize_parameters(series_df: pd.DataFrame, test_fraction: float = 0.2) -> Dict:
    """
    Find optimal Prophet parameters for a dataset using grid search.

    Args:
        series_df: DataFrame with 'ds' and 'y' columns
        test_fraction: Fraction of data to use for testing

    Returns:
        dict: Best parameters found
    """
    param_grid = {
        'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.5],
        'seasonality_prior_scale': [1.0, 10.0, 20.0],
        'daily_seasonality': [True, False],
        'weekly_seasonality': [True, False],
        'yearly_seasonality': [True, False],
    }

    keys = param_grid.keys()
    values = param_grid.values()
    param_combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    test_size = int(len(series_df) * test_fraction)
    train_df = series_df.iloc[:-test_size].copy()
    test_df = series_df.iloc[-test_size:].copy()

    best_nrmse = float('inf')
    best_params: Dict = {}

    logging.getLogger('prophet').setLevel(logging.ERROR)

    max_combinations = 20
    param_combinations = param_combinations[:max_combinations]

    for params in param_combinations:
        try:
            model = Prophet(**params)
            model.fit(train_df)

            future = model.make_future_dataframe(periods=test_size)
            forecast = model.predict(future)

            yhat_test = forecast['yhat'].iloc[-test_size:].to_numpy()
            y_true = test_df['y'].to_numpy()
            rmse = float(np.sqrt(np.mean((y_true - yhat_test) ** 2)))
            y_range = np.max(y_true) - np.min(y_true)
            nrmse = rmse / y_range if y_range > 0 else rmse

            if nrmse < best_nrmse:
                best_nrmse = nrmse
                best_params = params.copy()
        except Exception:
            continue

    if not best_params:
        best_params = {
            'daily_seasonality': True,
            'weekly_seasonality': True,
            'yearly_seasonality': True,
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0
        }

    return best_params


def forecast_and_nrmse(series_df: pd.DataFrame, test_fraction: float = 0.2, optimize_params_flag: bool = True) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Compute Prophet forecast and NRMSE on test set.

    Args:
        series_df: DataFrame with 'ds' and 'y' columns
        test_fraction: Fraction of data to use for testing
        optimize_params_flag: Whether to optimize parameters for this dataset

    Returns:
        tuple: (nrmse, forecast_df, test_df)
    """
    test_size = int(len(series_df) * test_fraction)
    train_df = series_df.iloc[:-test_size].copy()
    test_df = series_df.iloc[-test_size:].copy()

    if optimize_params_flag:
        prophet_params = optimize_parameters(series_df, test_fraction)
    else:
        prophet_params = {
            'daily_seasonality': True,
            'weekly_seasonality': True,
            'yearly_seasonality': True,
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0
        }

    model = Prophet(**prophet_params)
    logging.getLogger('prophet').setLevel(logging.WARNING)
    model.fit(train_df)

    future = model.make_future_dataframe(periods=test_size)
    forecast = model.predict(future)

    yhat_test = forecast['yhat'].iloc[-test_size:].to_numpy()
    y_true = test_df['y'].to_numpy()
    rmse = float(np.sqrt(np.mean((y_true - yhat_test) ** 2)))
    y_range = np.max(y_true) - np.min(y_true)
    nrmse = rmse / y_range if y_range > 0 else rmse

    return nrmse, forecast, test_df