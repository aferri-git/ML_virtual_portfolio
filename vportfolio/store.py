# Functions related to storing and reading data tables into the DuckDB database

# Imports
import duckdb
import pandas as pd
import tempfile
from datetime import date
import pytest
from pathlib import Path
from vportfolio.params import LIVE_DB

# Create the empty tables in the database if they do not exist
def create_tables(con):
    # create the nav table
    con.execute("""
        CREATE TABLE IF NOT EXISTS nav (
            date DATE,
            portfolio VARCHAR,
            nav DOUBLE,
            daily_return DOUBLE,
            PRIMARY KEY (date, portfolio)
        )
    """)
    # create the holdings table
    con.execute("""
        CREATE TABLE IF NOT EXISTS holdings (
            date DATE,
            portfolio VARCHAR,
            ticker VARCHAR,
            weight DOUBLE,
            PRIMARY KEY (date, portfolio, ticker)
        )
    """)
    # create the targets table
    con.execute("""
        CREATE TABLE IF NOT EXISTS targets (
            decision_date DATE,
            portfolio VARCHAR,
            ticker VARCHAR,
            weight DOUBLE,
            PRIMARY KEY (decision_date, portfolio, ticker)
        )
    """)
    # create the trades table
    con.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_date DATE,
            portfolio VARCHAR,
            ticker VARCHAR,
            weight_before DOUBLE,
            weight_after DOUBLE,
            cost DOUBLE,
            PRIMARY KEY (trade_date, portfolio, ticker)
        )
    """)
    # create the asset returns table
    con.execute("""
        CREATE TABLE IF NOT EXISTS asset_returns (
            date DATE,
            ticker VARCHAR,
            daily_return DOUBLE,
            PRIMARY KEY (date, ticker)
        )
    """)

# add the rows of a dataframe to one of the live tables
# will raise an error if a key is in the table already
def append_rows(con, table, rows):
    con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM rows")

# now we connect to the database file, creating the folder and the tables if needed
def open_store(path = LIVE_DB):
    path.parent.mkdir(parents = True, exist_ok = True)
    con = duckdb.connect(str(path))
    create_tables(con)
    return con

# checks atest date recorded in the nav table, NA if nothing is recorded yet
def last_date(con):
    return con.execute("SELECT max(date) FROM nav").fetchone()[0] # fetchnote takes that one row out, as a tuple like

# NAV of each portfolio on the latest recorded date
# returns a dataframe with columns portfolio and nav
def last_nav(con):
    return con.execute("""
        SELECT portfolio, nav
        FROM nav
        WHERE date = (SELECT max(date) FROM nav)
    """).df()

# Weights held by each portfolio at the latest recorded close
# returns a dataframe with columns portfolio, ticker and weight
def last_holdings(con):
    return con.execute("""
        SELECT portfolio, ticker, weight
        FROM holdings
        WHERE date = (SELECT max(date) FROM holdings)
    """).df()
