from typing import Tuple
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def _build_features(history_df: pd.DataFrame) -> pd.DataFrame:
    feat_df = history_df.copy()
    for lag in [1, 2, 3, 7, 14]:
        feat_df[f'lag_{lag}'] = feat_df['y'].shift(lag)
    feat_df['rolling_mean_7'] = feat_df['y'].shift(1).rolling(7).mean()
    feat_df['rolling_std_7'] = feat_df['y'].shift(1).rolling(7).std()
    feat_df = feat_df.dropna().reset_index(drop=True)
    return feat_df


def forecast_and_nrmse(series_df: pd.DataFrame, test_fraction: float = 0.2) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Minimal XGBoost with lag features and recursive forecasting.
    Returns (nrmse, forecast_df, test_df).
    """
    test_size = int(len(series_df) * test_fraction)
    train_df = series_df.iloc[:-test_size].copy().reset_index(drop=True)
    test_df = series_df.iloc[-test_size:].copy().reset_index(drop=True)

    train_feat = _build_features(train_df)
    X_train = train_feat.drop(columns=['ds', 'y'])
    y_train = train_feat['y']

    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    history = train_df.copy()
    preds = []
    for i in range(len(test_df)):
        extended = pd.concat([history, test_df[['ds']].iloc[:i].assign(y=preds)], ignore_index=True)
        feat_row = _build_features(extended).iloc[[-1]]
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