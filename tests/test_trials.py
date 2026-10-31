## Test functions from trials.py

# Imports
import numpy as np
import pandas as pd
from pathlib import Path
from vportfolio.trials import save_returns, make_new_row, update_trials_log
from vportfolio.allocators import equal_weights

# test for: folder exists, file is where it says, column name is correct and values are as expected
def test_path_exists(tmp_path): # tmp_path wll assign the test a temporary directory to save the file
    folder = tmp_path / "test_folder"
    dates = pd.date_range('2020-01-01', periods=3)
    nav = pd.Series([1.0, 1.1, 1.21], index=dates) # 10% return each day
    path = save_returns("test", nav, folder)
    saved = pd.read_parquet(path)
    
    assert folder.exists() # check if the folder was created
    assert path.exists() # check if the file was created
    assert list(saved.columns) == ["test"] # check if the column name is correct
    assert np.isclose(saved["test"], 0.1).all() # values saved
    
    nav = pd.Series([1.0, 2, 3], index=dates)
    path = save_returns("test", nav, folder)
    saved = pd.read_parquet(path)
    
    assert np.isclose(saved["test"], [1.0, 0.5]).all() # check for the exact expected returns

def test_make_new_row():
    name = "test"
    allocator = equal_weights
    role = "testing"
    performance = pd.Series({"ann_return": 0.01, "ann_vol": 0.1, "sharpe": 0.5, "max_drawdown": -0.2})
    turnover = pd.Series([0.1,0.3])
    returns_file = Path("results/backtests/test.parquet") # the test will only take the name so no need for it to be real
    row = make_new_row(name, allocator, role, performance, turnover, returns_file)
    key_list = ["date", "name", "allocator", "settings", "role", "ann_return", "ann_vol", "sharpe", "avg_turnover", "max_drawdown", "returns_file"]
    # we make the list as is of today in case the function will change in the future and we forget
    assert set(row.keys()) == set(key_list)
    assert row["allocator"] == "equal_weights" # that the allocator function is saved as a string
    assert row["settings"] == "cap=0.2, lookback=504" # check that the right parameters are ingested
    assert row["ann_return"] == 0.01
    assert row["ann_vol"] == 0.1
    assert row["sharpe"] == 0.5
    assert row["max_drawdown"] == -0.2
    assert np.isclose(row["avg_turnover"], 0.2)
    assert isinstance(row["returns_file"], str)
    assert row["date"] == pd.Timestamp.today().strftime("%Y-%m-%d")

def test_update_trials_log(tmp_path):
    log_path = tmp_path / "log.csv"
    
    # test a file is created with 1 row with correct name
    test_rows1=pd.DataFrame([{"name": "A", "sharpe": 0.5}])
    update_trials_log(test_rows1, path = log_path)
    log = pd.read_csv(log_path)
    assert len(log) == 1
    assert set(log["name"]) == {"A"}
    
    # test a new row is added with correct name
    test_rows2=pd.DataFrame([{"name": "A", "sharpe": 0.5},
                            {"name":"B", "sharpe":0.2}])
    update_trials_log(test_rows2, path = log_path)
    log = pd.read_csv(log_path)
    assert len(log) == 2
    assert set(log["name"]) == {"A", "B"}
    
    # test an old row is edited
    test_rows3=pd.DataFrame([{"name": "A", "sharpe": 0.9},
                            {"name":"B", "sharpe":0.2}])
    update_trials_log(test_rows3, path = log_path)
    log = pd.read_csv(log_path)
    assert len(log) == 2 # to make sure A was not just duplicated
    assert log.loc[log["name"] == "A", "sharpe"].item() == 0.9