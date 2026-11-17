# Live trading engine
    ## 1. Know where the last run stopped
    ## 2. Know what kind of day it is (i.e. last day of the month)
    ## 3. Valuing the portfolio as it moves on normal days
    ## 4. Running the allocators live on decision days
    ## 5. Trading after decision days
    # 6. Recording results

# Imports
import pandas as pd
import numpy as np
from vportfolio.params import LIVE_START, CAP, LOOKBACK
from vportfolio import engine

# A function to compare the last date in store with the new downloaded data
## returns the dates still to process, in order
def new_days (dates, last, start = LIVE_START):
    new_dates = dates[dates >= pd.Timestamp(start)]
    if last is not None:
        new_dates = new_dates[new_dates > pd.Timestamp(last)]
    return new_dates

# Decide if the date in question is a decision day, a trade day or a normal day
## Returns a series of labels for the given dates
def day_kind (todo, rebal_dates, dates):
    pos = dates.get_indexer(rebal_dates) # step 1: positions of the decision days
    next_pos = pos + 1 # step 2: one further
    next_pos = next_pos[next_pos < len(dates)] # adjust for runs on decision days
    trade_days = dates[next_pos] # step 3: the dates
    kinds = pd.Series("normal", index = todo)
    kinds[todo.isin(trade_days)] = "trade"
    kinds[todo.isin(rebal_dates)] = "decision"
    return kinds

# Actions required for normal days: hold, and let the weights drift with the day's returns
## Returns the portfolio's return for the day and the drifted weights at today's close
def normal_day(holdings, returns):
    held_returns = returns[holdings.index] # keep returns of held tickets
    if held_returns.isna().any(): # make sure no missing return impacts calculations
        raise ValueError(f"missing returns for held tickers: {list(held_returns[held_returns.isna()].index)}")
    grown = holdings * (held_returns +1)
    day_return = grown.sum() - 1
    new_holdings = grown / grown.sum()
    return day_return, new_holdings

# A wrapper running the allocator live on decision days
## Returns a series with ticker and target weight
def decide_targets(allocator, processed, day, cap = CAP, lookback = LOOKBACK):
    return engine.new_target(allocator, processed["returns"], processed["eligible"], day, cap, lookback, processed["returns"].columns)

# Build the data set the 60/40 runs on, from a raw download
## Returns a dictionary with close, open, returns and eligible
def benchmark_data(raw, tickers = ["SPY", "AGG"]):
    close = raw["Close"][tickers]
    open_ = raw["Open"][tickers]
    returns = close.pct_change(fill_method=None)
    eligible = close.notna()
    return {"close": close, "open": open_, "returns": returns, "eligible": eligible}

# Simulate trading on a trading day
## Returns a series with portfolio's daily return, new holdings and trades
def trade_day(holdings, targets, overnight, intraday, cost_bp):
    # 1. overnight: the holdings move to today's open (nothing held on the first day)
    if holdings.empty:
        overnight_return = 0.0
        weights_at_open = pd.Series(0.0, index = targets.index)
    else:
        overnight_return, weights_at_open = normal_day(holdings, overnight)
    # line up both on every ticker that is either held or targeted, 0 where absent
    tickers = targets.index.union(weights_at_open.index)
    before = weights_at_open.reindex(tickers, fill_value = 0.0)
    after = targets.reindex(tickers, fill_value = 0.0)
    # 2. trade at the open: same cost rule as the backtest
    costs = engine.trading_costs(after, before, cost_bp.reindex(tickers))
    # 3. during the day: the new weights move to today's close (only tickers actually held)
    intraday_return, new_holdings = normal_day(after[after > 0], intraday)
    # 4. the day's return, chaining the three moves
    day_return = (1 + overnight_return) * (1 - costs["cost"]) * (1 + intraday_return) - 1
    # 5. the trades: every ticker whose weight changed
    changed = (after - before).abs() > 1e-12
    trades = pd.DataFrame({"ticker": tickers[changed],
                           "weight_before": before[changed].values,
                           "weight_after": after[changed].values,
                           "cost": ((after - before).abs() * cost_bp.reindex(tickers) / 10000)[changed].values})
    return day_return, new_holdings, trades
