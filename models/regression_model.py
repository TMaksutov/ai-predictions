from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _ensure_daily_frequency(series_df: pd.DataFrame) -> None:
    """
    Ensure the series has a daily frequency with no gaps.
    Raises ValueError otherwise.
    """
    if series_df.empty:
        raise ValueError("Empty series provided")

    # Sort and check inferred frequency
    series_df = series_df.sort_values("ds").reset_index(drop=True)
    inferred = pd.infer_freq(series_df["ds"])
    if inferred != "D":
        raise ValueError(
            "Dataset must be daily frequency ('D') with regular 1-day intervals. "
            "Please resample/fill gaps before uploading."
        )


def _build_features(series_df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Build regression features:
    - Lags: 1 and 7 days
    - Day-of-week one-hot (drop_first to avoid collinearity)
    Returns (features_df, feature_columns)
    """
    df = series_df.copy()
    df = df.sort_values("ds").reset_index(drop=True)

    df["lag1"] = df["y"].shift(1)
    df["lag7"] = df["y"].shift(7)
    df["dow"] = df["ds"].dt.dayofweek.astype(int)

    dow_dummies = pd.get_dummies(df["dow"], prefix="dow", drop_first=True)
    features = pd.concat([df[["ds", "y", "lag1", "lag7"]], dow_dummies], axis=1)

    # Drop rows without required lagged values
    features = features.dropna().reset_index(drop=True)

    feature_cols = ["lag1", "lag7"] + list(dow_dummies.columns)
    return features, feature_cols


def forecast_regression(
    series_df: pd.DataFrame, test_fraction: float = 0.2
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Forecast using a simple multiple linear regression with lag and calendar features.

    Splits the series into train/test by time, fits on train, predicts the last
    test_fraction window. Returns (nrmse, forecast_df, test_df), where nrmse is
    RMSE normalized by the range of y_true on the test set.
    """
    if series_df.shape[0] < 30:
        raise ValueError("Insufficient data (need at least 30 rows)")

    # Ensure daily frequency
    _ensure_daily_frequency(series_df)

    n = len(series_df)
    test_size = max(1, int(n * test_fraction))
    test_start_ds = series_df.sort_values("ds").iloc[-test_size]["ds"]

    # Build features for the whole series then split by date
    features, feature_cols = _build_features(series_df)

    train_mask = features["ds"] < test_start_ds
    test_mask = features["ds"] >= test_start_ds

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Not enough rows after feature construction for train/test split")

    X_train = features.loc[train_mask, feature_cols].to_numpy()
    y_train = features.loc[train_mask, "y"].to_numpy()

    X_test = features.loc[test_mask, feature_cols].to_numpy()
    y_test = features.loc[test_mask, "y"].to_numpy()

    model = LinearRegression()
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)

    # Build forecast dataframe aligned to test window
    forecast_df = pd.DataFrame({
        "ds": features.loc[test_mask, "ds"].to_numpy(),
        "yhat": yhat,
    })

    # Compute nRMSE over the aligned test rows
    rmse = float(np.sqrt(np.mean((y_test - yhat) ** 2)))
    y_range = float(np.max(y_test) - np.min(y_test))
    nrmse = rmse / y_range if y_range > 0 else rmse

    # Original test frame from the raw series (for plotting split line)
    test_df = series_df[series_df["ds"] >= test_start_ds].reset_index(drop=True)

    return nrmse, forecast_df, test_df