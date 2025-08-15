from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesModels:
    """Collection of stable time series prediction models."""
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.metrics = {}
    
    def prepare_features(self, df: pd.DataFrame, target_col: str, max_lags: int = 12) -> pd.DataFrame:
        """Create lag features for time series modeling."""
        df_features = df.copy()
        
        # Create lag features
        for lag in range(1, min(max_lags + 1, len(df) // 2)):
            df_features[f'lag_{lag}'] = df_features[target_col].shift(lag)
        
        # Create rolling statistics
        for window in [3, 7, 14]:
            if len(df) >= window:
                df_features[f'rolling_mean_{window}'] = df_features[target_col].rolling(window=window).mean()
                df_features[f'rolling_std_{window}'] = df_features[target_col].rolling(window=window).std()
        
        # Create trend features
        df_features['trend'] = np.arange(len(df_features))
        df_features['trend_squared'] = df_features['trend'] ** 2
        
        # Create seasonal features (if enough data)
        if len(df) >= 12:
            df_features['month'] = pd.to_datetime(df_features.index).month
            df_features['quarter'] = pd.to_datetime(df_features.index).quarter
        
        return df_features
    
    def naive_forecast(self, df: pd.DataFrame, target_col: str, horizon: int) -> np.ndarray:
        """Naive forecast using last value."""
        last_value = df[target_col].iloc[-1]
        return np.full(horizon, last_value)
    
    def naive_seasonal_forecast(self, df: pd.DataFrame, target_col: str, horizon: int, seasonality: int = 12) -> np.ndarray:
        """Seasonal naive forecast using values from previous season."""
        if len(df) < seasonality:
            return self.naive_forecast(df, target_col, horizon)
        
        predictions = []
        for i in range(horizon):
            season_idx = len(df) - seasonality + (i % seasonality)
            predictions.append(df[target_col].iloc[season_idx])
        
        return np.array(predictions)
    
    def linear_trend_forecast(self, df: pd.DataFrame, target_col: str, horizon: int) -> np.ndarray:
        """Linear trend forecast using simple linear regression."""
        try:
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[target_col].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            future_X = np.arange(len(df), len(df) + horizon).reshape(-1, 1)
            predictions = model.predict(future_X)
            
            # Clip predictions to reasonable range
            y_min, y_max = y.min(), y.max()
            y_range = max(1.0, y_max - y_min)
            predictions = np.clip(predictions, y_min - 2*y_range, y_max + 2*y_range)
            
            return predictions
        except:
            return self.naive_forecast(df, target_col, horizon)
    
    def exponential_smoothing_forecast(self, df: pd.DataFrame, target_col: str, horizon: int, alpha: float = 0.3) -> np.ndarray:
        """Simple exponential smoothing forecast."""
        try:
            values = df[target_col].values
            if len(values) < 2:
                return self.naive_forecast(df, target_col, horizon)
            
            # Calculate exponential smoothing
            smoothed = [values[0]]
            for i in range(1, len(values)):
                smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[i-1])
            
            # Forecast using the last smoothed value
            last_smoothed = smoothed[-1]
            predictions = np.full(horizon, last_smoothed)
            
            return predictions
        except:
            return self.naive_forecast(df, target_col, horizon)
    
    def moving_average_forecast(self, df: pd.DataFrame, target_col: str, horizon: int, window: int = 5) -> np.ndarray:
        """Moving average forecast."""
        try:
            if len(df) < window:
                return self.naive_forecast(df, target_col, horizon)
            
            # Use the last window values to calculate moving average
            last_values = df[target_col].tail(window).values
            moving_avg = np.mean(last_values)
            
            predictions = np.full(horizon, moving_avg)
            return predictions
        except:
            return self.naive_forecast(df, target_col, horizon)
    
    def polynomial_trend_forecast(self, df: pd.DataFrame, target_col: str, horizon: int, degree: int = 2) -> np.ndarray:
        """Polynomial trend forecast."""
        try:
            if len(df) < degree + 2:
                return self.linear_trend_forecast(df, target_col, horizon)
            
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[target_col].values
            
            # Fit polynomial
            coeffs = np.polyfit(X.flatten(), y, degree)
            poly = np.poly1d(coeffs)
            
            # Predict
            future_X = np.arange(len(df), len(df) + horizon)
            predictions = poly(future_X)
            
            # Clip predictions to reasonable range
            y_min, y_max = y.min(), y.max()
            y_range = max(1.0, y_max - y_min)
            predictions = np.clip(predictions, y_min - 2*y_range, y_max + 2*y_range)
            
            return predictions
        except:
            return self.linear_trend_forecast(df, target_col, horizon)
    
    def fit_all_models(self, df: pd.DataFrame, target_col: str, test_size: float = 0.2) -> Dict[str, np.ndarray]:
        """Fit all models and return predictions."""
        # Split data into train and test
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Store test data for evaluation
        self.test_data = test_df
        self.train_data = train_df
        
        # Generate predictions for all models
        horizon = len(test_df)
        
        models = {
            'Naive': self.naive_forecast(train_df, target_col, horizon),
            'Seasonal Naive': self.naive_seasonal_forecast(train_df, target_col, horizon),
            'Linear Trend': self.linear_trend_forecast(train_df, target_col, horizon),
            'Exponential Smoothing': self.exponential_smoothing_forecast(train_df, target_col, horizon),
            'Moving Average': self.moving_average_forecast(train_df, target_col, horizon),
            'Polynomial Trend': self.polynomial_trend_forecast(train_df, target_col, horizon)
        }
        
        self.predictions = models
        return models
    
    def evaluate_models(self, target_col: str) -> pd.DataFrame:
        """Evaluate all models using test data."""
        if not hasattr(self, 'test_data') or not hasattr(self, 'predictions'):
            raise ValueError("Models must be fitted first using fit_all_models")
        
        actual = self.test_data[target_col].values
        results = []
        
        for model_name, predictions in self.predictions.items():
            # Calculate metrics
            mae = mean_absolute_error(actual, predictions)
            mse = mean_squared_error(actual, predictions)
            rmse = np.sqrt(mse)
            
            # Calculate MAPE safely (avoid division by zero)
            mape = mean_absolute_percentage_error(actual, predictions) * 100
            
            results.append({
                'Model': model_name,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'MAPE (%)': mape
            })
        
        # Sort by RMSE (lower is better)
        results_df = pd.DataFrame(results).sort_values('RMSE')
        self.metrics = results_df
        return results_df
    
    def get_forecast_dataframe(self, df: pd.DataFrame, target_col: str, horizon: int) -> pd.DataFrame:
        """Get complete dataframe with historical data and all model forecasts."""
        # Fit models first
        self.fit_all_models(df, target_col)
        
        # Get historical data
        historical_dates = df.index
        historical_values = df[target_col].values
        
        # Generate future dates
        last_date = historical_dates[-1]
        if isinstance(last_date, pd.Timestamp):
            # Infer frequency from data
            freq = pd.infer_freq(historical_dates)
            if freq is None:
                # Fallback to daily frequency
                freq = 'D'
            future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
        else:
            # If not datetime, use integer indices
            future_dates = np.arange(len(df), len(df) + horizon)
        
        # Create result dataframe
        result_data = {
            'date': pd.concat([pd.Series(historical_dates), pd.Series(future_dates)]),
            'actual': pd.concat([pd.Series(historical_values), pd.Series([np.nan] * horizon)]),
            'kind': ['historical'] * len(df) + ['forecast'] * horizon
        }
        
        # Add predictions from each model
        for model_name, predictions in self.predictions.items():
            # Pad predictions to match horizon
            if len(predictions) < horizon:
                predictions = np.pad(predictions, (0, horizon - len(predictions)), mode='edge')
            elif len(predictions) > horizon:
                predictions = predictions[:horizon]
            
            result_data[f'{model_name}_forecast'] = pd.concat([
                pd.Series([np.nan] * len(df)),
                pd.Series(predictions)
            ])
        
        return pd.DataFrame(result_data)
    
    def get_best_model(self) -> str:
        """Get the name of the best performing model based on RMSE."""
        if self.metrics.empty:
            raise ValueError("Models must be evaluated first using evaluate_models")
        return self.metrics.iloc[0]['Model']