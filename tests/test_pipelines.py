## Test functions for pipelines.py
    # Tests requirements are within the assert statements of each function

import pandas as pd
import numpy as np
from pathlib import Path
from vportfolio import allocators, pipelines
from tests.helpers import make_fake_data, make_fake_raw
from vportfolio.params import UNIVERSE, BENCHMARK, RISK_FREE

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

def test_data_pipeline():
    processed = pipelines.data_pipeline(make_fake_raw())
    # 1. always true: the expected tables, aligned on the same dates and tickers
    assert set(processed.keys()) == {"close", "open", "returns", "benchmark_returns",
                                     "eligible", "rebal_dates", "rf_daily", "cost_bp"}
    for key in ["open", "returns", "eligible"]:
        assert processed[key].index.equals(processed["close"].index), key
        assert processed[key].columns.equals(processed["close"].columns), key
    assert processed["rf_daily"].index.equals(processed["close"].index)
    assert processed["rf_daily"].notna().all()
    assert set(processed["close"].columns) <= set(processed["cost_bp"].index)
    # 2. known answer: data from Jan 2019 to mid-March 2022 gives 38 complete months, ending in February
    assert len(processed["rebal_dates"]) == 38
    assert processed["rebal_dates"].iloc[-1] == pd.Timestamp("2022-02-28")
    assert not processed["eligible"].iloc[0].any()       # nothing eligible on day one
    assert processed["eligible"].iloc[-1].all()          # everything eligible after 504 days
    # 3. inputs matter: data reaching the end of March adds March as a rebalance date
    assert len(pipelines.data_pipeline(make_fake_raw("2022-03-31"))["rebal_dates"]) == 39
