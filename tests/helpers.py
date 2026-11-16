import numpy as np
import pandas as pd
import duckdb
from vportfolio.params import UNIVERSE, BENCHMARK, RISK_FREE
from vportfolio.store import create_tables

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

# helper: fake raw data shaped like the Yahoo download (all five fields for every ticker)
def make_fake_raw(end="2022-03-15"):
    days = pd.bdate_range("2019-01-01", end)
    tickers = UNIVERSE + BENCHMARK + RISK_FREE
    rng = np.random.default_rng(0)
    close = pd.DataFrame(rng.uniform(10, 100, (len(days), len(tickers))), index=days, columns=tickers)
    open_ = pd.DataFrame(rng.uniform(10, 100, (len(days), len(tickers))), index=days, columns=tickers)
    volume = pd.DataFrame(1e6, index=days, columns=tickers)
    high = np.maximum(close, open_) * 1.01
    low = np.minimum(close, open_) * 0.99
    return pd.concat({"Close": close, "Open": open_, "High": high, "Low": low, "Volume": volume}, axis=1)

# fake raw where the last day is a quiet one: every close 1% above the day before
def calm_fake_raw():
    raw = make_fake_raw("2022-03-15")
    last, previous = raw.index[-1], raw.index[-2]
    for ticker in raw["Close"].columns:
        raw.loc[last, ("Close", ticker)] = raw.loc[previous, ("Close", ticker)] * 1.01
    return raw

# helper: nav rows for one day, from a dictionary portfolio -> nav
def make_nav_rows(day, navs):
    return pd.DataFrame({"date": day, "portfolio": list(navs.keys()),
                         "nav": list(navs.values()), "daily_return": 0.0})
# helper: an empty database in memory, with the tables created
def make_store():
    con = duckdb.connect()
    create_tables(con)
    return con
