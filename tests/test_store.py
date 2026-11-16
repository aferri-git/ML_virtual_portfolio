# test functions from store.py

import duckdb
import pandas as pd
import pytest
from datetime import date
from vportfolio.store import create_tables, append_rows, last_date, last_nav, last_holdings, open_store
from tests.helpers import make_nav_rows, make_store

# create_tables
def test_create_tables():
    con = make_store()
    # 1. always true: calling it again does not fail
    create_tables(con)
    # 2. known answer: the five tables exist
    tables = set(con.execute("SHOW TABLES").df()["name"])
    assert tables == {"nav", "holdings", "targets", "trades", "asset_returns"}
    # 3. inputs matter: rows already stored survive another call
    append_rows(con, "nav", make_nav_rows("2026-12-01", {"nco": 1.0}))
    create_tables(con)
    assert con.execute("SELECT count(*) FROM nav").fetchone()[0] == 1

# append_rows
def test_append_rows():
    con = make_store()
    append_rows(con, "nav", make_nav_rows("2026-12-01", {"nco": 1.0, "ew": 1.0}))
    # 1. always true: the table grows by the number of rows added
    assert con.execute("SELECT count(*) FROM nav").fetchone()[0] == 2
    # 2. known answer: values land in the right columns, even with the columns in another order
    shuffled = make_nav_rows("2026-12-02", {"nco": 1.05})[["nav", "daily_return", "portfolio", "date"]]
    append_rows(con, "nav", shuffled)
    stored = con.execute("SELECT portfolio, nav FROM nav WHERE date = '2026-12-02'").fetchone()
    assert stored == ("nco", 1.05)
    # 3. inputs matter: a row already there is refused, and nothing from that batch is added
    batch = make_nav_rows("2026-12-02", {"nco": 9.99, "ew": 1.0}) # nco repeated, ew new
    with pytest.raises(duckdb.ConstraintException):
        append_rows(con, "nav", batch)
    assert con.execute("SELECT count(*) FROM nav").fetchone()[0] == 3
    assert con.execute("SELECT nav FROM nav WHERE date = '2026-12-02' AND portfolio = 'nco'").fetchone()[0] == 1.05

# last_date, last_nav, last_holdings
def test_read_functions():
    con = make_store()
    # 1. always true: an empty store answers without failing
    assert last_date(con) is None
    assert last_nav(con).empty and last_holdings(con).empty
    # 2. known answer: with two days stored, the answers come from the later one
    append_rows(con, "nav", make_nav_rows("2026-12-01", {"nco": 1.0, "ew": 1.0}))
    append_rows(con, "nav", make_nav_rows("2026-12-02", {"nco": 1.01, "ew": 0.99}))
    assert last_date(con) == date(2026, 12, 2)
    navs = last_nav(con).set_index("portfolio")["nav"]
    assert navs["nco"] == 1.01 and navs["ew"] == 0.99 and len(navs) == 2
    for day, weight in [("2026-12-01", 0.5), ("2026-12-02", 0.6)]:
        append_rows(con, "holdings", pd.DataFrame({"date": day, "portfolio": "nco",
                                                   "ticker": ["SPY", "AGG"], "weight": [weight, 1 - weight]}))
    held = last_holdings(con).set_index("ticker")["weight"]
    assert held["SPY"] == 0.6 and len(held) == 2
    # 3. inputs matter: adding a later day moves the answer
    append_rows(con, "nav", make_nav_rows("2026-12-03", {"nco": 1.02, "ew": 0.98}))
    assert last_date(con) == date(2026, 12, 3)

# open_store
def test_open_store(tmp_path):
    path = tmp_path / "live" / "test.duckdb" # the folder does not exist yet
    con = open_store(path)
    # 1. always true: the file is created, with its folder
    assert path.exists()
    # 2. known answer: the tables are ready
    assert len(con.execute("SHOW TABLES").df()) == 5
    # 3. inputs matter: what is stored is still there after closing and opening again
    append_rows(con, "nav", make_nav_rows("2026-12-01", {"nco": 1.0}))
    con.close()
    again = open_store(path)
    assert last_date(again) == date(2026, 12, 1)
    again.close()
