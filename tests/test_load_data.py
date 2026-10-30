## Test functions for laod_data.py
    # ensure data are loaded correctly and have the expected properties
    
# Imports

import pytest
from vportfolio.load_data import load_universe, load_benchmark
from vportfolio.params import PROCESSED, DATA_KEYS, RAW_FILE

# Data is only local, so skip if not available
pytestmark = pytest.mark.skipif(not PROCESSED.exists() or not RAW_FILE.exists(),
                                reason="local data not available")

# test the loading of ETF data
def test_load_universe():
    # run the function
    universe_data = load_universe(folder = PROCESSED)
    # test the names
    assert set(universe_data.keys()) == set(DATA_KEYS)
    # the dataframes in the dictionary have the correct column names and dates as index
    for key in ["open", "returns", "eligible"]:
        assert universe_data[key].columns.equals(universe_data["close"].columns), key
        assert universe_data[key].index.equals(universe_data["close"].index), key
    # the cost data covers every ticker
    assert set(universe_data["close"].columns) <= set(universe_data["cost_bp"].index) # lesser or equal to make sure nothing is missing
    # rebalance dates exist in real trading days
    assert universe_data["rebal_dates"].isin(universe_data["close"].index).all()

# test the loading of benchmark data
def test_load_benchmark():
    # run the function
    benchmark_data = load_benchmark(folder = PROCESSED)
    # add a test to see if the tickers correspond to the columns
    assert list(benchmark_data["close"].columns) == ["SPY", "AGG"]
    # test the names
    assert set(benchmark_data.keys()) == set(DATA_KEYS)
    # the dataframes in the dictionary have the correct column names and dates as index
    for key in ["open", "returns", "eligible"]:
        assert benchmark_data[key].columns.equals(benchmark_data["close"].columns), key
        assert benchmark_data[key].index.equals(benchmark_data["close"].index), key
    # the cost data covers every ticker
    assert set(benchmark_data["close"].columns) <= set(benchmark_data["cost_bp"].index) # lesser or equal to make sure nothing is missing
    # rebalance dates exist in real trading days
    assert benchmark_data["rebal_dates"].isin(benchmark_data["close"].index).all()
