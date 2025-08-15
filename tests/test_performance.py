import os, sys
import pandas as pd, numpy as np
import pytest
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ts_core import forecast_linear_safe, load_table, detect_interval

def test_large_dataset_performance():
    """Test performance with large datasets."""
    # Create a large dataset
    n_points = 1000
    dates = pd.date_range("2020-01-01", periods=n_points, freq="D")
    values = np.random.randn(n_points).cumsum() + 100
    
    df = pd.DataFrame({"date": dates, "value": values})
    
    start_time = time.time()
    out = forecast_linear_safe(df, "date", "value", horizon=100)
    end_time = time.time()
    
    # Should complete in reasonable time (less than 5 seconds)
    assert end_time - start_time < 5.0
    assert len(out) == n_points + 100
    assert out["kind"].tail(100).tolist() == ["forecast"] * 100

def test_very_long_horizon():
    """Test forecasting with very long horizon."""
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=10, freq="D"),
        "value": np.arange(10) + 100
    })
    
    start_time = time.time()
    out = forecast_linear_safe(df, "date", "value", horizon=1000)
    end_time = time.time()
    
    # Should complete in reasonable time
    assert end_time - start_time < 10.0
    assert len(out) == 10 + 1000
    assert out["kind"].tail(1000).tolist() == ["forecast"] * 1000

def test_memory_efficiency():
    """Test that large operations don't consume excessive memory."""
    # Create large dataset
    n_points = 5000
    dates = pd.date_range("2020-01-01", periods=n_points, freq="h")
    values = np.random.randn(n_points).cumsum() + 100
    
    df = pd.DataFrame({"date": dates, "value": values})
    
    # Just test that it completes without error
    out = forecast_linear_safe(df, "date", "value", horizon=500)
    
    # Verify output is correct
    assert len(out) == n_points + 500
    assert out["kind"].tail(500).tolist() == ["forecast"] * 500

def test_concurrent_forecasting():
    """Test that multiple forecasts can run concurrently."""
    import concurrent.futures
    
    def run_forecast(seed):
        np.random.seed(seed)
        n_points = 100
        dates = pd.date_range("2023-01-01", periods=n_points, freq="D")
        values = np.random.randn(n_points).cumsum() + 100
        
        df = pd.DataFrame({"date": dates, "value": values})
        out = forecast_linear_safe(df, "date", "value", horizon=50)
        return len(out)
    
    # Run 5 forecasts concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_forecast, i) for i in range(5)]
        results = [future.result() for future in futures]
    
    # All should complete successfully
    assert all(result == 150 for result in results)  # 100 historical + 50 forecast

def test_numerical_stability():
    """Test numerical stability with extreme values."""
    # Test with very large numbers
    df_large = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=10, freq="D"),
        "value": [1e15, 2e15, 3e15, 4e15, 5e15, 6e15, 7e15, 8e15, 9e15, 1e16]
    })
    
    out_large = forecast_linear_safe(df_large, "date", "value", horizon=5)
    assert not out_large["yhat"].isna().any()
    assert len(out_large) == 15
    
    # Test with very small numbers
    df_small = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=10, freq="D"),
        "value": [1e-15, 2e-15, 3e-15, 4e-15, 5e-15, 6e-15, 7e-15, 8e-15, 9e-15, 1e-14]
    })
    
    out_small = forecast_linear_safe(df_small, "date", "value", horizon=5)
    assert not out_small["yhat"].isna().any()
    assert len(out_small) == 15

def test_interval_detection_performance():
    """Test interval detection performance with various frequencies."""
    frequencies = ["s", "min", "h", "D", "W", "ME", "QE", "YE"]
    
    for freq in frequencies:
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=100, freq=freq),
            "value": np.random.randn(100)
        })
        
        start_time = time.time()
        interval = detect_interval(df["date"])
        end_time = time.time()
        
        # Should complete quickly
        assert end_time - start_time < 0.1
        assert interval != "unknown"

def test_forecast_consistency():
    """Test that forecasts are consistent across multiple runs."""
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=20, freq="D"),
        "value": np.arange(20) + 100
    })
    
    # Run forecast multiple times
    results = []
    for _ in range(5):
        out = forecast_linear_safe(df, "date", "value", horizon=10)
        results.append(out["yhat"].tail(10).tolist())
    
    # All results should be identical (deterministic)
    for i in range(1, len(results)):
        np.testing.assert_array_almost_equal(results[0], results[i])

def test_edge_case_performance():
    """Test performance with edge case data patterns."""
    # Test with constant values
    df_constant = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=100, freq="D"),
        "value": [42] * 100
    })
    
    start_time = time.time()
    out_constant = forecast_linear_safe(df_constant, "date", "value", horizon=50)
    end_time = time.time()
    
    assert end_time - start_time < 1.0
    assert len(out_constant) == 150
    
    # Test with alternating values
    df_alternating = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=100, freq="D"),
        "value": [1, -1] * 50
    })
    
    start_time = time.time()
    out_alternating = forecast_linear_safe(df_alternating, "date", "value", horizon=50)
    end_time = time.time()
    
    assert end_time - start_time < 1.0
    assert len(out_alternating) == 150