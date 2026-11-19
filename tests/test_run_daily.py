# Test functions for run_daily.py

import pytest
from run_daily import run
from tests.helpers import calm_fake_raw

PORTFOLIOS = {"ew": {"allocator": "equal_weights", "cap": 0.2,  "lookback": 504},
              "sf": {"allocator": "sixty_forty",   "cap": None, "lookback": 1}}

def clean_download(end, start):
    return calm_fake_raw()

def broken_download(end, start):
    bad = calm_fake_raw()
    bad.loc[bad.index[-1], ("Close", "UUP")] *= 3 # last close tripled
    return bad

def test_run(tmp_path):
    db, raw_folder = tmp_path / "store.duckdb", tmp_path / "raw"
    settings = dict(start = "2021-06-01", raw_folder = raw_folder, data_start = "2019-01-01", portfolios = PORTFOLIOS)
    summary = run("2022-03-16", db, download = clean_download, **settings)

    # 1. always true: the store and the day's download file are created; the summary has the three parts
    assert db.exists() and (raw_folder / "yahoo_prices_2022-03-16.parquet").exists()
    assert set(summary.keys()) == {"run_date", "days", "navs"}

    # 2. known answer: every day from the live start is processed, with a NAV for each portfolio
    assert summary["days"][0] == "2021-06-01" and summary["days"][-1] == "2022-03-15"
    assert set(summary["navs"].keys()) == {"ew", "sf"}

    # 3. inputs matter: running again on the same date reuses the download and adds nothing
    again = run("2022-03-16", db, download = broken_download, **settings)
    assert again["days"] == [] and again["navs"] == summary["navs"]

    # 3. inputs matter: a bad download stops the run before the store is ever opened
    other_db = tmp_path / "other.duckdb"
    with pytest.raises(ValueError):
        run("2022-03-17", other_db, download = broken_download, **settings)
    assert not other_db.exists()