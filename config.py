"""
Configuration constants for the time series forecasting application.
"""

# Feature engineering parameters
# Short and long seasonal context
LAG_PERIODS = [1, 7, 30, 90, 365]
MOVING_AVERAGE_WINDOWS = [7, 30, 90]

# Fourier seasonality
FOURIER_PERIODS = [7, 30, 365]  # Candidate fixed periods (days)
FOURIER_HARMONICS = 3  # Harmonics per period (sin/cos pairs per k=1..K)

# Auto-detection of seasonality
AUTO_DETECT_FOURIER = True
AUTO_MAX_PERIOD = 400  # Search up to this period (days)
AUTO_TOP_N = 3         # Add top-N detected periods
AUTO_MIN_CYCLES = 3    # Require at least this many cycles in history

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


