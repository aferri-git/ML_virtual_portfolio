## Test functions for load_data.py
    # Both loaders must return a dictionary with the same keys and aligned data,
    # ready for trial_pipeline
    
# Imports

import pytest
from vportfolio.load_data import load_universe, load_benchmark
from vportfolio.params import PROCESSED, DATA_KEYS, RAW_FILE

# Data is only local, so skip if not available
pytestmark = pytest.mark.skipif(not PROCESSED.exists() or not RAW_FILE.exists(),
                                reason="local data not available")

# shared checks for any loaded data dictionary
def check_loaded(data):
    # the keys trial_pipeline expects
    assert set(data.keys()) == set(DATA_KEYS)
    # open, returns and eligible have the same tickers and dates as close
    for key in ["open", "returns", "eligible"]:
        assert data[key].columns.equals(data["close"].columns), key
        assert data[key].index.equals(data["close"].index), key
    # every ticker has a trading cost (subset: extra tickers in cost_bp are fine)
    assert set(data["close"].columns) <= set(data["cost_bp"].index)
    # every rebalance date is a real trading day
    assert data["rebal_dates"].isin(data["close"].index).all()

# the full ETF universe
def test_load_universe():
    data = load_universe()
    check_loaded(data)

# the 60/40 benchmark: SPY and AGG only, from the raw file
def test_load_benchmark():
    data = load_benchmark()
    check_loaded(data)
    assert list(data["close"].columns) == ["SPY", "AGG"]
