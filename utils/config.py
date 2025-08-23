"""
Configuration constants for the time series forecasting application.
"""

# Feature engineering parameters
LAG_PERIODS = [1, 7]  # Days to lag (removed 14, 28 to avoid multicollinearity)
MOVING_AVERAGE_WINDOWS = [7]  # Days for moving averages (removed 14, 28)

# Model evaluation parameters
DEFAULT_TEST_FRACTION = 0.2
MIN_TRAINING_ROWS = 10

# File processing parameters
MAX_FILE_SIZE_MB = 10
SUPPORTED_EXTENSIONS = [".csv"]

# Plotting parameters
DEFAULT_FIGURE_SIZE = (20, 6)
DEFAULT_PLOT_COLORS = {
    "actual": "#222",
    "prediction": "#d62728",
    "forecast": "orange"
}
