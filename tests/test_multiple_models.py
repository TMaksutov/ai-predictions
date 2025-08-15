import os, sys, io
import pandas as pd, numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ts_models import TimeSeriesModels
from ts_core import forecast_multiple_models, DataError

def _make_test_df(n=50, start="2023-01-01"):
    """Create test dataframe with trend and seasonality."""
    dates = pd.date_range(start=start, periods=n, freq="D")
    # Create data with trend and seasonality
    trend = np.arange(n) * 0.1
    seasonality = 10 * np.sin(2 * np.pi * np.arange(n) / 7)  # Weekly seasonality
    noise = np.random.normal(0, 0.5, n)
    values = 100 + trend + seasonality + noise
    
    return pd.DataFrame({"date": dates, "value": values})

def test_time_series_models_initialization():
    """Test TimeSeriesModels class initialization."""
    models = TimeSeriesModels()
    assert models.models == {}
    assert models.predictions == {}
    assert models.metrics == {}

def test_naive_forecast():
    """Test naive forecast functionality."""
    models = TimeSeriesModels()
    df = _make_test_df(n=20)
    
    predictions = models.naive_forecast(df, "value", 5)
    assert len(predictions) == 5
    assert np.allclose(predictions, df["value"].iloc[-1])

def test_linear_trend_forecast():
    """Test linear trend forecast."""
    models = TimeSeriesModels()
    df = _make_test_df(n=20)
    
    predictions = models.linear_trend_forecast(df, "value", 5)
    assert len(predictions) == 5
    # Should show some trend
    assert not np.allclose(predictions, predictions[0])

def test_exponential_smoothing_forecast():
    """Test exponential smoothing forecast."""
    models = TimeSeriesModels()
    df = _make_test_df(n=20)
    
    predictions = models.exponential_smoothing_forecast(df, "value", 5)
    assert len(predictions) == 5
    # Exponential smoothing produces constant forecasts
    assert np.allclose(predictions, predictions[0])

def test_moving_average_forecast():
    """Test moving average forecast."""
    models = TimeSeriesModels()
    df = _make_test_df(n=20)
    
    predictions = models.moving_average_forecast(df, "value", 5)
    assert len(predictions) == 5
    # Moving average produces constant forecasts
    assert np.allclose(predictions, predictions[0])

def test_polynomial_trend_forecast():
    """Test polynomial trend forecast."""
    models = TimeSeriesModels()
    df = _make_test_df(n=20)
    
    predictions = models.polynomial_trend_forecast(df, "value", 5)
    assert len(predictions) == 5
    assert not np.allclose(predictions, predictions[0])

def test_fit_all_models():
    """Test fitting all models."""
    models = TimeSeriesModels()
    df = _make_test_df(n=50)
    
    predictions = models.fit_all_models(df, "value", test_size=0.2)
    
    expected_models = ['Naive', 'Seasonal Naive', 'Linear Trend', 
                      'Exponential Smoothing', 'Moving Average', 'Polynomial Trend']
    
    assert set(predictions.keys()) == set(expected_models)
    assert hasattr(models, 'test_data')
    assert hasattr(models, 'train_data')
    assert len(models.test_data) == int(50 * 0.2)
    assert len(models.train_data) == int(50 * 0.8)

def test_evaluate_models():
    """Test model evaluation."""
    models = TimeSeriesModels()
    df = _make_test_df(n=50)
    
    # Fit models first
    models.fit_all_models(df, "value", test_size=0.2)
    
    # Evaluate models
    metrics = models.evaluate_models("value")
    
    assert not metrics.empty
    assert 'Model' in metrics.columns
    assert 'MAE' in metrics.columns
    assert 'MSE' in metrics.columns
    assert 'RMSE' in metrics.columns
    assert 'MAPE (%)' in metrics.columns
    
    # Should have metrics for all models
    assert len(metrics) == 6

def test_get_forecast_dataframe():
    """Test getting forecast dataframe."""
    models = TimeSeriesModels()
    df = _make_test_df(n=50)
    
    # Get forecast dataframe
    forecast_df = models.get_forecast_dataframe(df, "value", 10)
    
    assert not forecast_df.empty
    assert 'date' in forecast_df.columns
    assert 'actual' in forecast_df.columns
    assert 'kind' in forecast_df.columns
    
    # Should have forecast columns for all models
    forecast_cols = [col for col in forecast_df.columns if col.endswith('_forecast')]
    assert len(forecast_cols) == 6
    
    # Check historical vs forecast split
    historical = forecast_df[forecast_df['kind'] == 'historical']
    forecast = forecast_df[forecast_df['kind'] == 'forecast']
    
    assert len(historical) == 50
    assert len(forecast) == 10

def test_get_best_model():
    """Test getting best model."""
    models = TimeSeriesModels()
    df = _make_test_df(n=50)
    
    # Fit and evaluate models
    models.fit_all_models(df, "value", test_size=0.2)
    models.evaluate_models("value")
    
    best_model = models.get_best_model()
    assert best_model in models.predictions.keys()

def test_forecast_multiple_models_integration():
    """Test the integrated forecast_multiple_models function."""
    df = _make_test_df(n=50)
    
    result, metrics = forecast_multiple_models(df, "date", "value", 10)
    
    assert not result.empty
    assert not metrics.empty
    assert 'date' in result.columns
    assert 'actual' in result.columns
    
    # Check that we have forecast columns
    forecast_cols = [col for col in result.columns if col.endswith('_forecast')]
    assert len(forecast_cols) > 0

def test_forecast_multiple_models_small_dataset():
    """Test multiple models with small dataset."""
    df = _make_test_df(n=5)
    
    with pytest.raises(DataError, match="Dataset too small"):
        forecast_multiple_models(df, "date", "value", 5)

def test_forecast_multiple_models_invalid_horizon():
    """Test multiple models with invalid horizon."""
    df = _make_test_df(n=50)
    
    with pytest.raises(DataError, match="Invalid horizon"):
        forecast_multiple_models(df, "date", "value", 0)
    
    with pytest.raises(DataError, match="Invalid horizon"):
        forecast_multiple_models(df, "date", "value", 1001)

def test_model_stability():
    """Test that models are stable and don't crash."""
    models = TimeSeriesModels()
    
    # Test with various data patterns
    test_cases = [
        _make_test_df(n=20),  # Small dataset
        _make_test_df(n=100),  # Larger dataset
        _make_test_df(n=50, start="2020-01-01"),  # Different start date
    ]
    
    for df in test_cases:
        try:
            predictions = models.fit_all_models(df, "value", test_size=0.2)
            metrics = models.evaluate_models("value")
            
            # All models should produce predictions
            assert len(predictions) == 6
            assert not metrics.empty
            
        except Exception as e:
            pytest.fail(f"Model failed for dataset with {len(df)} rows: {e}")

def test_feature_preparation():
    """Test feature preparation functionality."""
    models = TimeSeriesModels()
    df = _make_test_df(n=50)
    
    features_df = models.prepare_features(df, "value")
    
    # Should have additional features
    assert len(features_df.columns) > len(df.columns)
    
    # Check for lag features
    lag_cols = [col for col in features_df.columns if col.startswith('lag_')]
    assert len(lag_cols) > 0
    
    # Check for rolling features
    rolling_cols = [col for col in features_df.columns if col.startswith('rolling_')]
    assert len(rolling_cols) > 0