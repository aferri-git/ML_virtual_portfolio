## Test functions for engine.py
    # Tests requirements are within the assert statements of each function

# Imports
import pandas as pd
import numpy as np
from vportfolio.engine import lookback_window, apply_cap, new_target, trading_costs, holding_nav, drift, backtest_engine
from vportfolio.params import LOOKBACK

# lookback_window
def test_lookback_window():
    dates = pd.bdate_range("2020-1-1", periods=600) # a bit more than 504 days
    fake_returns = pd.DataFrame(0.001, index=dates, columns=["A", "B", "C", "D", "E"]) # constant returns
    fake_returns.loc[:dates[200], "E"] = np.nan # E starts trading later
    fake_eligible = fake_returns.notna().cumsum() >= LOOKBACK # same rule as preprocessing
    check_day = dates[-1] # last date: A-D eligible, E not yet
    window = lookback_window(fake_returns, fake_eligible, check_day)
    
    assert len(window) == LOOKBACK # exactly 504 rows
    assert window.index[-1] == check_day # ends on the requested date: no look-ahead
    assert "E" not in window.columns # E not eligible yet, so excluded

# apply_cap
def test_apply_cap():
    w_equal = pd.Series({"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2}) # nothing above the cap
    w_over = pd.Series({"A": 0.5, "B": 0.2, "C": 0.1, "D": 0.1, "E": 0.05, "F": 0.05}) # A above the cap
    w_over_copy = w_over.copy() # to check the original is not modified
    capped_equal = apply_cap(w_equal)
    capped_over = apply_cap(w_over)

    assert (capped_equal == w_equal).all() # 1. unchanged when nothing is over the cap
    assert (capped_over <= 0.20 + 1e-12).all() # 2. no weight above the cap
    assert abs(capped_over.sum() - 1) < 1e-12 # 2. weights still sum to 1
    assert abs(capped_over["E"] / capped_over["F"] - 1) < 1e-12 # 3. E and F started equal, end equal
    assert abs(capped_over["C"] / capped_over["E"] - 2) < 1e-12 # 3. C was twice E, still twice E
    assert (w_over == w_over_copy).all() # 4. original weights not modified

# new_target 
def test_new_target():
    def half_half(window): # dummy allocator: 50/50 in A and B, ignores the data
        return pd.Series({"A": 0.5, "B": 0.5})

    dates = pd.bdate_range("2020-1-1", periods=600) # a bit more than 504 days
    fake_returns = pd.DataFrame(0.001, index=dates, columns=["A", "B", "C", "D", "E"]) # constant returns
    fake_returns.loc[:dates[200], "E"] = np.nan # E starts trading later
    fake_eligible = fake_returns.notna().cumsum() >= LOOKBACK # same rule as preprocessing
    check_day = dates[-1] # last date: A-D eligible, E not yet
    target = new_target(half_half, fake_returns, fake_eligible, check_day, None, LOOKBACK, fake_returns.columns)
    assert list(target.index) == list(fake_returns.columns) # one entry for every ticker
    assert (target[["C", "D", "E"]] == 0).all() # tickers not chosen get 0
    assert abs(target.sum() - 1) < 1e-12 # weights sum to 1


# trading_costs
def test_trading_costs():
    target = pd.Series({"A": 0.6, "B": 0.4})
    weights1 = pd.Series({"A": 0.0, "B": 0.0}) # at start
    weights2 = pd.Series({"A": 0.6, "B": 0.4}) # already at target
    weights3 = pd.Series({"A": 0.62, "B": 0.38}) # drifted
    cost_bp1 = pd.Series({"A": 10, "B": 10}) # 10 bp each
    cost_bp2 = pd.Series({"A": 2, "B": 10}) # different costs

    test1 = trading_costs(target, weights1, cost_bp1) # 100% traded at 0.1%
    test2 = trading_costs(target, weights2, cost_bp1) # 0% traded
    test3 = trading_costs(target,weights3, cost_bp1) # traded 4%
    test4 = trading_costs(target,weights3, cost_bp2) # traded 4% at diferent costs

    assert np.isclose(test1["cost"], 1*0.001,atol=1e-12) 
    assert np.isclose(test2["cost"], 0*0.001,atol=1e-12)
    assert np.isclose(test3["cost"], 0.04*0.001, atol=1e-12)
    assert np.isclose(test4["cost"], 0.02*0.0002 + 0.02*0.001, atol=1e-12)

    assert np.isclose(test1["turnover"], 1, atol=1e-12)
    assert np.isclose(test2["turnover"], 0, atol=1e-12)
    assert np.isclose(test3["turnover"], 0.04, atol=1e-12)
    assert np.isclose(test4["turnover"], 0.04, atol=1e-12)

# holding_nav
def test_holding_nav():
    dates = pd.bdate_range("2020-1-1", periods=6)

    # test 1 : NAV flat
    close_flat= pd.DataFrame({"A": [100] * 6, "B": [100] * 6}, index = dates) # all open and close set to 100
    open_flat = close_flat.copy()
    weights_equal = pd.Series({"A": 0.5, "B": 0.5}) # equal weights
    trade_day = dates[0]
    next_trade_day = dates[5]
    test1= holding_nav(1.0,weights_equal, close_flat, open_flat, trade_day, next_trade_day)
    assert np.isclose(test1, 1.0, atol=1e-12).all() 

    # test 2: NAV of 1 ETF, other stable
    close_up = close_flat.copy()
    close_up.loc[dates[2]:, "A"] = 200
    open_up = open_flat.copy()
    open_up.loc[dates[3]:, "A"] = 200
    weights_1 = pd.Series({"A": 0.5, "B": 0.5}) # equal weights
    test2= holding_nav(1.0,weights_1, close_up, open_up, trade_day, next_trade_day)
    assert np.isclose(test2.iloc[2:], 1.5, atol=1e-12).all()

    # test 3: ETF with NANs
    close_na = close_flat.copy()
    close_na["C"] = np.nan # add ETF with NANs
    open_na = close_na.copy()
    weights_na = pd.Series({"A": 0.5, "B": 0.5, "C": 0.0}) # give C 0 weight
    test3 = holding_nav(1.0, weights_na, close_na, open_na, trade_day, next_trade_day)
    assert test3.notna().all() # no NaN in the NAV
    assert np.isclose(test3, 1.0, atol=1e-12).all() # A and B flat: NAV unchanged\
    
# drift
def test_drift():
    dates = pd.bdate_range("2020-1-1", periods=6)
    weights_equal = pd.Series({"A": 0.5, "B": 0.5}) # equal weights
    close_flat= pd.DataFrame({"A": [100] * 6, "B": [100] * 6}, index = dates) # all open and close set to 100
    open_flat = close_flat.copy()
    trade_day = dates[0]
    next_trade_day = dates[5]
    
    test1 = drift(1.0, weights_equal, open_flat, trade_day, next_trade_day)
    assert np.isclose(test1["NAV"], 1.0, atol=1e-12).all() 
    assert np.isclose(test1["Weights"], [0.5, 0.5], atol=1e-12).all() 

    open_up = open_flat.copy()
    open_up.loc[dates[3]:, "A"] = 200
    test2 = drift(1.0, weights_equal, open_up, trade_day, next_trade_day)
    assert np.isclose(test2["NAV"], 1.5, atol=1e-12)
    assert np.isclose(test2["Weights"]["A"], 2/3, atol=1e-12)

    close_na = close_flat.copy()
    close_na["C"] = np.nan # add ETF with NANs
    open_na = close_na.copy()
    weights_na = pd.Series({"A": 0.5, "B": 0.5, "C": 0.0}) # give C 0 weight
    test3 = drift(1.0, weights_na, open_na, trade_day, next_trade_day)
    assert test3["Weights"].notna().all()
    assert np.isclose(test3["Weights"], [0.5, 0.5, 0], atol=1e-12).all()

# backtest_engine
def test_backtest_engine():
# fake market data
    dates_long = pd.bdate_range("2020-1-1", periods=756) # about 3 years of business days
    t = np.arange(len(dates_long))
    close_fake = pd.DataFrame({"AAA": 100 * 1.001 ** t, "BBB": [50.0] * len(dates_long)}, index=dates_long) # AAA +0.1% a day, BBB flat
    open_fake = close_fake.shift(1) # open = previous day's close
    open_fake.iloc[0] = close_fake.iloc[0] # first day: open = close
    returns_fake = close_fake.pct_change(fill_method=None)
    eligible_fake = close_fake.notna() # everything eligible
    month_days = returns_fake.loc["2020-03-01":"2022-10-31"].index # start a few months in
    rebal_fake = month_days.to_series().groupby([month_days.year, month_days.month]).max()
    cost_zero = pd.Series({"AAA": 0, "BBB": 0}) # no costs
    cost_ten = pd.Series({"AAA": 10, "BBB": 10}) # 10 bp each

    # dummy allocators
    def all_in_aaa(window): # always 100% AAA
        return pd.Series({"AAA": 1.0})

    def half_half(window): # always 50/50
        return pd.Series({"AAA": 0.5, "BBB": 0.5})

    # test 1: buy and hold, no costs
    test1 = backtest_engine(all_in_aaa, close_fake, open_fake, returns_fake, eligible_fake, rebal_fake, cost_zero, cap=None, lookback=1)
    first_trade_day = test1["Turnover"].index[0]
    last_nav_day = test1["NAV"].index[-1]
    expected = close_fake.loc[last_nav_day, "AAA"] / open_fake.loc[first_trade_day, "AAA"]
    assert np.isclose(test1["NAV"].iloc[-1], expected, atol=1e-12) # matches holding AAA directly

    # test 2: first trade from cash is 100%
    assert np.isclose(test1["Turnover"].iloc[0], 1.0, atol=1e-12)

    # test 3: costs charged once at 10 bp
    test3 = backtest_engine(all_in_aaa, close_fake, open_fake, returns_fake, eligible_fake, rebal_fake, cost_ten, cap=None, lookback=1)
    assert np.isclose(test3["NAV"].iloc[-1], test1["NAV"].iloc[-1] * (1 - 0.001), atol=1e-12)

    # test 4: 50/50 drifts, so monthly rebalancing trades a little
    test4 = backtest_engine(half_half, close_fake, open_fake, returns_fake, eligible_fake, rebal_fake, cost_zero, cap=None, lookback=1)
    assert (test4["Turnover"].iloc[1:] > 0).all() # every rebalance trades something
    assert (test4["Turnover"].iloc[1:] < 0.05).all() # but only a little

    # test 5: 100% AAA never drifts, so nothing to trade after the first trade
    assert (test1["Turnover"].iloc[1:] < 1e-12).all()
