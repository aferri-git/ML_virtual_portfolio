# Data processing functions

# Imports
import pandas as pd
from vportfolio.params import UNIVERSE, BENCHMARK, RISK_FREE, DATA_FIXES, LOOKBACK, RESEARCH_START, RESEARCH_END, COST_OVERRIDES

# split raw data into partitions needed to run the engine
# returns a dictionary with the raw data
def split_prices(raw):
    # split the data and saves them through .copy so you can fix one without changing raw_data
    return {"universe_close" : raw["Close"][UNIVERSE].copy(),
            "universe_open" : raw["Open"][UNIVERSE].copy(),
            "benchmark_close" : raw["Close"][BENCHMARK + ["SPY"]].copy(),
            "irx" : raw["Close"][RISK_FREE[0]].copy()}
    
# apply the manual data fixes listed in params.DATA_FIXES
# returns a copy of close with the fixes applied
def apply_fixes(close, fixes=DATA_FIXES):
    close = close.copy()
    for ticker, bad_date, copy_from in fixes:
        if pd.Timestamp(bad_date) in close.index and pd.Timestamp(copy_from) in close.index:
            close.loc[bad_date, ticker] = close.loc[copy_from, ticker]
    return close

# translate price data into daily returns
# returns a dataframe with returns calculated from closing prices
def make_returns(close):
    return close.pct_change(fill_method=None)

# apply the eligibility criteria of at least 2 years of daily returns
# returns a dataframe with eligible returns
def make_eligible(returns, lookback=LOOKBACK):
    return returns.notna().cumsum() >= lookback

# Check last trading date of each month and regularize
# returns a Series of rebalance dates (plain 0, 1, 2... index, like rebal_dates.parquet)
def make_rebal_dates(index, start, end=None):
    dates = index[index >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    monthly = dates.to_series().groupby([dates.year, dates.month]).max().reset_index(drop=True)
    if monthly.empty:
        raise ValueError(f"no trading days between {start} and {end}")
    # drop the last month if the data stops before its last business day (incomplete month)
    last = monthly.iloc[-1]
    if last < last + pd.offsets.BMonthEnd(0):
        monthly = monthly.iloc[:-1]
    return monthly

# Convert the 13-week T-bill yield into a daily risk-free return on the ETF trading days
# returns a Series named "rf"
def make_rf_daily(irx, index):
    rf = (irx.ffill() # fill bond-market holidays with the previous yield
          / 100 # from % to decimal
          / 252) # annual to daily
    return rf.reindex(index).ffill().rename("rf") # align with the ETF trading days, fill any gaps

# estimate trading costs as per pre-registration
# returns a Series of per-ETF costs in bp, fixed from research-period dollar volume
def estimate_costs(raw_data, tickers, overrides = COST_OVERRIDES):
    volume = (raw_data["Close"][tickers] * raw_data["Volume"][tickers]).loc[RESEARCH_START:RESEARCH_END].median()
    cost_bp = pd.cut(volume, bins=[0, 100e6, 1e9, float("inf")], labels=[10, 5, 2]).astype(int)
    cost_bp[overrides] = 2
    return cost_bp