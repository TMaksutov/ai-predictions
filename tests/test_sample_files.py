from pathlib import Path
import os, sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ts_core import load_table, forecast_linear_safe

DATA_DIR = Path(__file__).resolve().parents[1] / "test_files"
FILES = sorted(DATA_DIR.glob("*.csv"))

@pytest.mark.parametrize("path", FILES, ids=[p.name for p in FILES])
def test_forecast_on_sample_files(path):
    with path.open("rb") as f:
        df = load_table(f)
    assert not df.empty
    date_col, target_col = df.columns[:2]
    out = forecast_linear_safe(df, date_col, target_col, horizon=2)
    assert "yhat" in out.columns
    assert len(out) == len(df) + 2
