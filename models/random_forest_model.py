from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def _build_features(history_df: pd.DataFrame) -> pd.DataFrame:
    feat_df = history_df.copy()
    for lag in [1, 2, 3, 7, 14]:
        feat_df[f'lag_{lag}'] = feat_df['y'].shift(lag)
    feat_df['rolling_mean_7'] = feat_df['y'].shift(1).rolling(7).mean()
    feat_df['rolling_std_7'] = feat_df['y'].shift(1).rolling(7).std()
    feat_df = feat_df.dropna().reset_index(drop=True)
    return feat_df


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
    Minimal Random Forest with lag features and recursive forecasting.
    Returns (nrmse, forecast_df, test_df).
    """
    try:
        n = len(series_df)
        if n < 20:
            return _naive_baseline(series_df, test_fraction)
        test_size = max(1, int(n * test_fraction))
        train_df = series_df.iloc[:-test_size].copy().reset_index(drop=True)
        test_df = series_df.iloc[-test_size:].copy().reset_index(drop=True)

        train_feat = _build_features(train_df)
        if train_feat.shape[0] < 10:
            return _naive_baseline(series_df, test_fraction)
        X_train = train_feat.drop(columns=['ds', 'y'])
        y_train = train_feat['y']

        if X_train.nunique().max() <= 1:
            return _naive_baseline(series_df, test_fraction)

        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        history = train_df.copy()
        preds = []
        for i in range(len(test_df)):
            extended = pd.concat([history, test_df[['ds']].iloc[:i].assign(y=preds)], ignore_index=True)
            feat_built = _build_features(extended)
            if len(feat_built) == 0:
                preds.append(float(y_train.iloc[-1]))
                continue
            feat_row = feat_built.iloc[[-1]]
            X_row = feat_row.drop(columns=['ds', 'y'])
            yhat = float(model.predict(X_row)[0])
            preds.append(yhat)

        forecast_df = pd.DataFrame({
            'ds': test_df['ds'].to_numpy(),
            'yhat': np.array(preds),
            'yhat_lower': np.array(preds),
            'yhat_upper': np.array(preds),
        })

        y_true = test_df['y'].to_numpy()
        rmse = float(np.sqrt(np.mean((y_true - forecast_df['yhat'].to_numpy()) ** 2)))
        y_range = np.max(y_true) - np.min(y_true)
        nrmse = rmse / y_range if y_range > 0 else rmse
        return nrmse, forecast_df, test_df
    except Exception:
        return _naive_baseline(series_df, test_fraction)