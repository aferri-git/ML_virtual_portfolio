## Test functions for pipelines.py
    # Tests requirements are within the assert statements of each function

import pandas as pd
import numpy as np
from pathlib import Path
from vportfolio import allocators, pipelines

# helper: a small fake data dictionary with the keys trial_pipeline expects
# SPY rises 1% a day, AGG is flat
def make_fake_data():
    dates = pd.bdate_range("2020-01-01", periods=10)
    close = pd.DataFrame({"SPY": 100 * 1.01 ** np.arange(10), "AGG": 100.0}, index=dates)
    return {"close": close,
            "open": close.copy(),
            "returns": close.pct_change(fill_method=None),
            "eligible": close.notna(),
            "rebal_dates": pd.Series([dates[2], dates[5]]),
            "cost_bp": pd.Series(2.0, index=close.columns),
            "rf_daily": pd.Series(0.0, index=dates)}

def test_trial_pipeline(tmp_path):
    data = make_fake_data()
    log_path = tmp_path / "log.csv"
    new_row = pipelines.trial_pipeline("test_6040", allocators.sixty_forty, "benchmark", data,
                                       cap=None, lookback=1, folder=tmp_path, trials_log=log_path)
    log = pd.read_csv(log_path)

    # 1. always true: one row logged, identical to the row returned, and its returns file exists
    assert len(log) == 1
    saved = log.iloc[0]
    for key in new_row:
        if isinstance(new_row[key], str):
            assert saved[key] == new_row[key], key # text: must match exactly
        else:
            assert np.isclose(saved[key], new_row[key]), key # numbers: must match up to rounding
    assert Path(new_row["returns_file"]).exists()

    # 2. known answer: name, allocator and settings recorded as run; SPY rises, AGG flat → positive return
    assert new_row["name"] == "test_6040"
    assert new_row["allocator"] == "sixty_forty"
    assert new_row["settings"] == "cap=None, lookback=1"
    assert new_row["ann_return"] > 0

    # 3. inputs matter: a new name adds a row; re-running a name replaces it; the allocator changes the result
    other_row = pipelines.trial_pipeline("test_ew", allocators.equal_weights, "benchmark", data,
                                         cap=None, lookback=1, folder=tmp_path, trials_log=log_path)
    assert len(pd.read_csv(log_path)) == 2 # new name → added
    pipelines.trial_pipeline("test_6040", allocators.sixty_forty, "benchmark", data,
                             cap=None, lookback=1, folder=tmp_path, trials_log=log_path)
    assert len(pd.read_csv(log_path)) == 2 # same name → replaced, not duplicated
    assert not np.isclose(other_row["ann_return"], new_row["ann_return"]) # 50/50 differs from 60/40