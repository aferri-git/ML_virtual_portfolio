## Test functions for pipelines.py
    # Tests requirements are within the assert statements of each function

import pandas as pd
import numpy as np
from vportfolio import allocators, pipelines

def test_trial_pipeline(tmp_path):
    # create a temp path for the test to save the parquet file and log
    tmp = tmp_path
    # set variables for a sixty_forty run
    dates = pd.bdate_range("2020-01-01", periods=10)
    name = "sixty_forty"
    role = "benchmark"
    close = pd.DataFrame({"SPY": 100 * 1.01 ** np.arange(10),
                        "AGG": 100.0}, index=dates)
    open_ = close.copy()
    data = {"close": close,
            "open": open_,
            "returns": close.pct_change(fill_method=None),
            "eligible": close.notna(),
            "rebal_dates": pd.Series([dates[2], dates[5]]),
            "cost_bp": pd.Series(2.0, index=close.columns),
            "rf_daily": pd.Series(0.0, index=dates)}
    # make a test run with 6040 allocator
    new_row = pipelines.trial_pipeline("test_6040", allocators.sixty_forty, "benchmark", data, cap = None, lookback = 1, folder = tmp, trials_log = tmp/"log.csv")
    log = pd.read_csv(tmp / "log.csv")
    assert len(log) == 1 # check that the log has 1 new row
    saved = log.iloc[0]
    # iterate over every line in both new_row and the saved log to make sure they are the same 
    for key in new_row:
        if isinstance(new_row[key], str):
            assert saved[key] == new_row[key], key # if is text, check line by line all key match. add key at the end in case there is an error
        else:
            assert np.isclose(saved[key], new_row[key]), key # otherwise it's a number so check if they are close
