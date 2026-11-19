# Data processing functions

# Imports
import pandas as pd
import yfinance as yf
from vportfolio.params import UNIVERSE, BENCHMARK, RISK_FREE, DATA_FIXES, LOOKBACK, RESEARCH_START, RESEARCH_END, COST_OVERRIDES, MAX_DAILY_MOVE, LIVE_RAW, DATA_START

# split raw data into partitions needed to run the engine
## returns a dictionary with the raw data
def split_prices(raw):
    # split the data and saves them through .copy so you can fix one without changing raw_data
    return {"universe_close" : raw["Close"][UNIVERSE].copy(),
            "universe_open" : raw["Open"][UNIVERSE].copy(),
            "benchmark_close" : raw["Close"][BENCHMARK + ["SPY"]].copy(),
            "irx" : raw["Close"][RISK_FREE[0]].copy()}

# apply the manual data fixes listed in params.DATA_FIXES
## returns a copy of close with the fixes applied
def apply_fixes(close, fixes=DATA_FIXES):
    close = close.copy()
    for ticker, bad_date, copy_from in fixes:
        if ticker in close.columns and pd.Timestamp(bad_date) in close.index and pd.Timestamp(copy_from) in close.index:
            close.loc[bad_date, ticker] = close.loc[copy_from, ticker]
    return close

# translate price data into daily returns
## returns a dataframe with returns calculated from closing prices
def make_returns(close):
    return close.pct_change(fill_method=None)

# apply the eligibility criteria of at least 2 years of daily returns
## returns a dataframe with eligible returns
def make_eligible(returns, lookback=LOOKBACK):
    return returns.notna().cumsum() >= lookback

# Check last trading date of each month and regularize
## returns a Series of rebalance dates (plain 0, 1, 2... index, like rebal_dates.parquet)
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
## returns a Series named "rf"
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

# Use yfinance libray to download them into a dataframe
    ## threads set to false makes sure tickers are downloaded in sequence and not in parallel (less likely to be blocked by yahoo finance)
    ## progress set to false hides the progress bar
    ## returns a dataframe with the prices of the listed tickers
def download_raw(end, start = DATA_START, universe = UNIVERSE, benchmark = BENCHMARK, rf = RISK_FREE):
    return yf.download(universe + benchmark + rf, start = start, end = end, auto_adjust = True, group_by = "column", threads = False, progress = False)

# Validate the downloaded data
## will return a list of sentences, one per problem. Empty if all is fine
def validate_raw(raw, tickers = UNIVERSE + BENCHMARK, rf = RISK_FREE, start = DATA_START, max_move = MAX_DAILY_MOVE,
                 fields = ("Open", "High", "Low", "Close", "Volume")):
    # create a list of problems
    problems = []
    # Is every expected ticker there, for every expected field? If not, say which are missing
    expected_cols = pd.MultiIndex.from_product([fields, tickers + rf])
    missing_cols = expected_cols.difference(raw.columns)
    if not missing_cols.empty:
        problems.append(f"missing columns: {list(missing_cols)}")
        return problems  # the checks below would crash without the columns
    # Is the index in date order, with no date twice?
    if not raw.index.is_monotonic_increasing:
        problems.append("dates are not in order")
    if not raw.index.is_unique:
        problems.append("some dates appear twice")
    # Does the history start where it should (the first date is not much later than start)?
    first_date = raw.index[0]
    if first_date > pd.Timestamp(start) + pd.Timedelta(days=7):
        problems.append(f"history starts late: first date {first_date.date()}, expected {start}")
    # Is the last row complete: every universe and benchmark ticker has a close and an open?
    for field in ["Close", "Open"]:
        last_row = raw[field][tickers].iloc[-1]
        empty = last_row[last_row.isna()].index.tolist()
        if empty:
            problems.append(f"last row has no {field} for: {empty}")
    # Are all prices above zero?
    for field in ["Open", "High", "Low", "Close"]:
        bad = (raw[field][tickers] <= 0).any()
        bad_tickers = bad[bad].index.tolist()
        if bad_tickers:
            problems.append(f"{field} at or below zero for: {bad_tickers}")
    # Is the last day's move believable: no ticker moved more than some limit in one day?
    close = raw["Close"][tickers]
    last_move = (close.iloc[-1] / close.iloc[-2] - 1).abs()
    too_big = last_move[last_move > max_move].index.tolist()
    if too_big:
        problems.append(f"last day move above {max_move:.0%} for: {too_big}")
    return problems

# Save the validated raw_data into a parquet file
def live_path(run_date, folder = LIVE_RAW):
    return folder / f"yahoo_prices_{run_date}.parquet"

def save_raw(raw_data, run_date, folder = LIVE_RAW):
    folder.mkdir(parents = True, exist_ok = True)
    path = live_path(run_date, folder)
    if not path.exists():
        raw_data.to_parquet(path)
    return path

# Build the data set the 60/40 runs on, from a raw download
# Returns a dictionary with close, open, returns and eligible
def benchmark_data(raw, tickers = ["SPY", "AGG"]):
    close = raw["Close"][tickers]
    open_ = raw["Open"][tickers]
    returns = close.pct_change(fill_method=None)
    eligible = close.notna()
    return {"close": close, "open": open_, "returns": returns, "eligible": eligible}