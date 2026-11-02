# Engine Steps
    # 1 decide target weights at month end close based on difference with allocator
    # 2 estimate trading costs based on decision from step 1
    # 3 estimate daily NAV from trade-day open to the day before next trade
    # 4 estimate value and weights at the next trade day's open

# Imports
import pandas as pd
import numpy as np
from vportfolio.params import LOOKBACK, CAP

# Helper Functions

## Returns a lookback window of returns for a given date
def lookback_window(returns, eligible, date, n=LOOKBACK):
    assets = eligible.columns[eligible.loc[date]] # which ETFs are available on a given date by TRUE/FALSE
    window = returns.loc[:date, assets].tail(n) # lookback period of 504 raws (2 years) from the given date for the eligible assets
    assert window.notna().all().all() # double checks and stops if lookback period contains NaN (extra safety after data tests)
    return window

## apply the maximum weight cap of 20% maximum per ETF
def apply_cap(w, cap=CAP):
    w = w.copy()
    for _ in range(100): # repeat the capping step until nothing is over the cap but never more than 100 times
        # for loop is necssary as redistributing can generate weights above 20% so needs to continue until all are in line
        over = w > cap + 1e-12 # finds weights above cap of 20% + rounding error of small decimals
        if not over.any(): 
            break # exits the loop when no weights are over the cap
        excess = (w[over] - cap).sum() # calculates the excess of over - cap as a total
        w[over] = cap # cuts those weights to CAP
        under = ~over & (w < cap) # finds weights below cap
        w[under] += excess * w[under] / w[under].sum() # adds to the current values of the under capped ETFs weight a share of the excess
        # proportional to their weights
    return w

# Main Functions

## step 1; decide
### applies allocator methodology to final target weight
### will return the final weights per ETF, and 0 for the unavailable ones
def new_target(allocator, returns, eligible, day, cap, lookback, tickers): # allocator passed as an argument
    w = allocator(lookback_window(returns, eligible, day, lookback)) # build lookback period for given day and give it to the allocator
    if cap is not None: 
        w = apply_cap(w, cap) # apply cap function unless is not set
    return w.reindex(tickers, fill_value=0.0)

## step 2; trade
### simulate trading costs based on weights change
### the results, cost and turnover, will feed into every trade
def trading_costs(target, weights, cost_bp):
    traded = (target - weights).abs()
    turnover = traded.sum()
    cost = (traded * cost_bp/10000).sum()    
    return pd.Series({"cost": cost, "turnover": turnover})

## step 3: hold
###calculate portfolio NAV from trad_day to next_trade_day
def holding_nav(value, target, close_p, open_p, trade_day, next_trade_day):
    cp = close_p.loc[trade_day: next_trade_day].iloc[:-1] # take closing prices of holding period, excluding last one
    op = open_p.loc[trade_day] # take open prices at trading date
    etf_growth = cp/op # calculate growht
    new_nav = value * (target * etf_growth).fillna(0).sum(axis=1) # calculates change in protfolio $ value and assigns 0 tho NaNs
    return new_nav

##step 4: drift
### calculates portfolio value at next trade day's open and the new weights
def drift(value, target, open_price, trade_day, next_trade_day):
    ETF_growth = (open_price.loc[next_trade_day]/open_price.loc[trade_day]).fillna(0)
    relative_weight = target * ETF_growth
    new_nav = value*(relative_weight.sum())
    new_weights = relative_weight / (relative_weight).sum()
    return pd.Series({"NAV": new_nav, "Weights": new_weights})

## We can now chain these functions into a single one, the backtesting engine, which will repeat the steps monthly
### It will return the series of daily NAVs and the monthly turnover
def backtest_engine (allocator, close_price, open_price, returns, eligible, rebal_dates, cost_bp, cap=CAP, lookback = LOOKBACK): 
    value = 1.0 # set starting value a 1$
    weights = pd.Series(0.0, index=close_price.columns) # set starting weights to 0
    monthly_NAV = [] # empty list to store NAVs
    turnover = {}
    for i, decision_day in enumerate(rebal_dates[:-1]):
        dates = close_price.index
        trade_day = dates[dates.get_loc(decision_day) + 1]
        next_trade_day = dates[dates.get_loc(rebal_dates.iloc[i + 1]) + 1]
        # step 1
        target = new_target(allocator, returns, eligible, decision_day, cap, lookback, close_price.columns)
        # step 2
        costs_turnover = trading_costs(target, weights, cost_bp)
        turnover[trade_day] = costs_turnover["turnover"]
        value = value * (1 - costs_turnover["cost"])
        # step 3
        daily_nav = holding_nav(value, target, close_price, open_price, trade_day, next_trade_day)
        monthly_NAV.append(daily_nav)
        # step 4
        nav_weights = drift(value, target, open_price, trade_day, next_trade_day)
        value = nav_weights["NAV"]
        weights = nav_weights["Weights"]       
    return pd.Series({"NAV": pd.concat(monthly_NAV), "Turnover": pd.Series(turnover)})