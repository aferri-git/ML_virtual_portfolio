# functions related to loading of data

# Imports
import pandas as pd
from vportfolio.params import RAW, PROCESSED, DATA_KEYS, RAW_FILE

# load data used accross all models
def _load_shared(folder = PROCESSED):
    rebal_dates = pd.read_parquet(folder / "rebal_dates.parquet")["date"]
    cost_bp = pd.read_parquet(folder / "cost_bp.parquet")["cost_bp"]
    rf_daily = pd.read_parquet(folder / "rf_daily.parquet")["rf"]
    shared_data= {"rebal_dates" : rebal_dates, "cost_bp" : cost_bp, "rf_daily" : rf_daily}
    return shared_data

# load the universe of ETFs
def load_universe(folder = PROCESSED):
    close = pd.read_parquet(folder / "close.parquet")
    open_ = pd.read_parquet(folder / "open.parquet")
    returns = pd.read_parquet(folder / "returns.parquet")
    eligible = pd.read_parquet(folder / "eligible.parquet")
    universe_data = {"close" : close, "open" : open_, "returns" : returns, "eligible": eligible}
    shared_data = _load_shared(folder)
    universe_data = {**universe_data, **shared_data}
    return universe_data

# load data for benchmark tickers only
def load_benchmark(tickers = ["SPY", "AGG"], raw_file = RAW_FILE, folder = PROCESSED):
    raw_data = pd.read_parquet(raw_file)
    close = raw_data["Close"][tickers]
    open_ = raw_data["Open"][tickers]
    returns = close.pct_change(fill_method=None)
    eligible = close.notna()
    benchmark_data = {"close" : close, "open" : open_, "returns" : returns, "eligible": eligible}
    shared_data = _load_shared(folder)
    benchmark_data = {**benchmark_data, **shared_data}
    return benchmark_data