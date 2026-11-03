## Test functions from trials.py

# Imports
import numpy as np
import pandas as pd
from pathlib import Path
from vportfolio.trials import save_returns, load_navs, make_new_row, update_trials_log
from vportfolio.allocators import equal_weights

# save_returns
def test_save_returns(tmp_path): # tmp_path: a fresh temporary folder for this test
    folder = tmp_path / "test_folder" # does not exist yet
    dates = pd.date_range("2020-01-01", periods=3)
    nav = pd.Series([1.0, 1.1, 1.21], index=dates) # +10% each day
    path = save_returns("test", nav, folder)
    saved = pd.read_parquet(path)
    # 1. always true: the folder is created, the file is where the function says, one column named after the strategy
    assert folder.exists()
    assert path.exists()
    assert list(saved.columns) == ["test"]
    # 2. known answer: NAV 1 → 1.1 → 1.21 gives two daily returns of 10%
    assert np.isclose(saved["test"], 0.1).all()
    # 3. inputs matter: a new NAV under the same name overwrites the file (+100%, then +50%)
    path = save_returns("test", pd.Series([1.0, 2.0, 3.0], index=dates), folder)
    assert np.isclose(pd.read_parquet(path)["test"], [1.0, 0.5]).all()

# load_navs
def test_load_navs(tmp_path):
    dates = pd.date_range("2020-01-01", periods=3)
    save_returns("a", pd.Series([1.0, 1.1, 1.21], index=dates), tmp_path)   # +10% a day
    save_returns("b", pd.Series([1.0, 2.0, 3.0], index=dates), tmp_path)    # +100%, then +50%
    navs = load_navs(["a", "b"], folder=tmp_path)
    # 1. always true: one column per name, no NaNs
    assert list(navs.columns) == ["a", "b"]
    assert navs.notna().all().all()
    # 2. known answer: compounding the returns gives 1.1, then 1.21
    assert np.allclose(navs["a"], [1.1, 1.21])
    # 3. inputs matter: different saved returns give different NAVs
    assert np.allclose(navs["b"], [2.0, 3.0])
    
# make_new_row
def test_make_new_row():
    performance = pd.Series({"ann_return": 0.01, "ann_vol": 0.1, "sharpe": 0.5, "max_drawdown": -0.2})
    turnover = pd.Series([0.1, 0.3])
    returns_file = Path("results/backtest/test_returns.parquet") # a Path, as save_returns returns; it need not exist
    row = make_new_row("test", equal_weights, "testing", performance, turnover, returns_file)
    # 1. always true: the log's columns, the path stored as text, today's date
    key_list = ["date", "name", "allocator", "settings", "role", "ann_return", "ann_vol",
                "sharpe", "avg_turnover", "max_drawdown", "returns_file"] # fixed so a renamed key gets noticed
    assert set(row.keys()) == set(key_list)
    assert row["returns_file"] == "results/backtest/test_returns.parquet"
    assert row["date"] == pd.Timestamp.today().strftime("%Y-%m-%d")
    # 2. known answer: values copied from the inputs, turnover averaged, allocator stored by name
    assert row["name"] == "test" and row["role"] == "testing"
    assert row["allocator"] == "equal_weights"
    assert row["ann_return"] == 0.01 and row["ann_vol"] == 0.1
    assert row["sharpe"] == 0.5 and row["max_drawdown"] == -0.2
    assert np.isclose(row["avg_turnover"], 0.2)
    assert row["settings"] == "cap=0.2, lookback=504" # defaults from params
    # 3. inputs matter: the settings record what the run actually used
    row_6040 = make_new_row("test", equal_weights, "testing", performance, turnover, returns_file, cap=None, lookback=1)
    assert row_6040["settings"] == "cap=None, lookback=1"

# update_trials_log
def test_update_trials_log(tmp_path):
    log_path = tmp_path / "log.csv" # does not exist yet
    log = update_trials_log(pd.DataFrame([{"name": "A", "sharpe": 0.5}]), path=log_path)
    # 1. always true: the file is created, and the returned log is what was saved
    assert log_path.exists()
    assert log.equals(pd.read_csv(log_path))
    # 2. known answer: the first write gives one row, named A
    assert len(log) == 1 and set(log["name"]) == {"A"}
    # 3. inputs matter: a new name adds a row; the same name replaces its row and leaves the others alone
    update_trials_log(pd.DataFrame([{"name": "B", "sharpe": 0.2}]), path=log_path)
    log = pd.read_csv(log_path)
    assert len(log) == 2 and set(log["name"]) == {"A", "B"} # B added
    update_trials_log(pd.DataFrame([{"name": "A", "sharpe": 0.9}]), path=log_path)
    log = pd.read_csv(log_path)
    assert len(log) == 2 # A replaced, not duplicated
    assert log.loc[log["name"] == "A", "sharpe"].item() == 0.9 # A has the new value
    assert log.loc[log["name"] == "B", "sharpe"].item() == 0.2 # B untouched