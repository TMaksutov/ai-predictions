import os, sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ts_core import detect_interval


def test_detect_interval_daily_series():
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    assert detect_interval(dates) == 'D (days)'
