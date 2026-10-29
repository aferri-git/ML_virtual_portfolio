# Test functions for data.py

# Check that the data has the properties confirmed during the exploration in order to not break the pipeline
    # No gaps after inception
    # No UUP manual fix (see decision log)
    # 204 month_end rebalance dates, 2005-2021
    # lookback window free of NaNs at every rebalance date
# Make tests now because a later step depends on the oputput of the pre-processing
    
    
## 5 Tests ##

# Imports
import pandas as pd
import pytest
from vportfolio.params import PROCESSED, RESEARCH_END, LOOKBACK, RESEARCH_START, RESEARCH_END

# Data is only local, so skip if not available
pytestmark = pytest.mark.skipif(not PROCESSED.exists(), reason="processed data not available")

# We use a decorator to make sure that the data is loaded only once per module, and not for every test function
@pytest.fixture(scope="module")

# loads the processed data from parquet files and returns a dictionary with the dataframes
# will share the data with all tests accross the file thanks to the above decorator
def data():
    return {
        "close":    pd.read_parquet(PROCESSED / "close.parquet"),
        "returns":  pd.read_parquet(PROCESSED / "returns.parquet"),
        "eligible": pd.read_parquet(PROCESSED / "eligible.parquet"),
        "rebal":    pd.read_parquet(PROCESSED / "rebal_dates.parquet")["date"],
    }

# make sure every ETF has no gaps after its inception date
def test_no_gaps_after_inception(data):
    close = data["close"]
    for t in close.columns:
        s = close[t].loc[close[t].first_valid_index():]
        assert s.notna().all(), f"{t} has gaps after inception"

# replaces UUP close on 20-06-2007 with the value of 19-06-2007, as done in the pre-processing step (check logs)
def test_uup_bad_close_fixed(data):
    c = data["close"]["UUP"]
    assert c.loc["2007-06-20"] == c.loc["2007-06-19"]
    
# makes sure that the rebalancing dates are 204 month_end dates between 2005 and 2021
def test_rebalance_dates(data):
    rebal = data["rebal"]
    assert len(rebal) == 204                        
    assert rebal.min() >= pd.Timestamp(RESEARCH_START)
    assert rebal.max() <= pd.Timestamp(RESEARCH_END)

# makes sure that the lookback window of 504 days has no NaNs at every rebalancing date
def test_lookback_windows_have_no_nans(data):
    ret, elig = data["returns"], data["eligible"]
    for d in data["rebal"]:
        assets = elig.columns[elig.loc[d]] # finds which ETFs are eligible for each date of 204
        window = ret.loc[:d, assets].tail(LOOKBACK) # builds new window based on previous step
        assert window.notna().all().all(), f"NaNs in lookback window at {d}" # will check if each ETF has values for each day and if all ETFs pass. will return a date if there is an issue