## Test functions for engine.py
    # Tests requirements are within the assert statements of each function

# Imports
import pandas as pd
import numpy as np
from vportfolio.engine import lookback_window, apply_cap, new_target, trading_costs, holding_nav, drift, backtest_engine
from vportfolio.params import LOOKBACK

# shared fake data: 600 business days, constant returns, E starts trading later
def make_fake_returns():
    dates = pd.bdate_range("2020-1-1", periods=600) # a bit more than 504 days
    fake_returns = pd.DataFrame(0.001, index=dates, columns=["A", "B", "C", "D", "E"])
    fake_returns.loc[:dates[200], "E"] = np.nan # E starts trading later
    fake_eligible = fake_returns.notna().cumsum() >= LOOKBACK # same rule as preprocessing
    return fake_returns, fake_eligible

# lookback_window
def test_lookback_window():
    fake_returns, fake_eligible = make_fake_returns()
    check_day = fake_returns.index[-1] # last date: A-D eligible, E not yet
    window = lookback_window(fake_returns, fake_eligible, check_day)
    # 1. always true: ends on the requested date (no look-ahead) and has no NaNs
    assert window.index[-1] == check_day
    assert window.notna().all().all()
    # 2. known answer: exactly 504 rows, and E (not yet eligible) is excluded
    assert len(window) == LOOKBACK
    assert list(window.columns) == ["A", "B", "C", "D"]
    # 3. inputs matter: the length follows n
    assert len(lookback_window(fake_returns, fake_eligible, check_day, n=100)) == 100

# apply_cap
def test_apply_cap():
    w_equal = pd.Series({"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2}) # nothing above the cap
    w_over = pd.Series({"A": 0.5, "B": 0.2, "C": 0.1, "D": 0.1, "E": 0.05, "F": 0.05}) # A above the cap
    w_over_copy = w_over.copy() # to check the original is not modified
    capped_over = apply_cap(w_over)
    # 1. always true: no weight above the cap, weights still sum to 1, original not modified
    assert (capped_over <= 0.20 + 1e-12).all()
    assert np.isclose(capped_over.sum(), 1, atol=1e-12)
    assert w_over.equals(w_over_copy)
    # 2. known answer: unchanged when nothing is over the cap; excess shared in proportion
    assert apply_cap(w_equal).equals(w_equal)
    assert np.isclose(capped_over["E"] / capped_over["F"], 1, atol=1e-12) # E and F started equal, end equal
    assert np.isclose(capped_over["C"] / capped_over["E"], 2, atol=1e-12) # C was twice E, still twice E
    # 3. inputs matter: the cap value sets the limit
    assert np.isclose(apply_cap(w_over, cap=0.3)["A"], 0.3, atol=1e-12)

# new_target
def test_new_target():
    def half_half(window): # dummy allocator: 50/50 in A and B, ignores the data
        return pd.Series({"A": 0.5, "B": 0.5})
    fake_returns, fake_eligible = make_fake_returns()
    check_day = fake_returns.index[-1]
    target = new_target(half_half, fake_returns, fake_eligible, check_day, None, LOOKBACK, fake_returns.columns)
    # 1. always true: one entry for every ticker, weights sum to 1
    assert list(target.index) == list(fake_returns.columns)
    assert np.isclose(target.sum(), 1, atol=1e-12)
    # 2. known answer: tickers the allocator did not choose get 0
    assert (target[["C", "D", "E"]] == 0).all()
    # 3. inputs matter: the cap is applied when given
    def heavy_a(window): # dummy allocator: 50% A, 10% each in five others
        return pd.Series({"A": 0.5, "B": 0.1, "C": 0.1, "D": 0.1, "E": 0.1, "F": 0.1})
    dates = pd.bdate_range("2020-1-1", periods=600)
    six_returns = pd.DataFrame(0.001, index=dates, columns=list("ABCDEF"))
    six_eligible = six_returns.notna()
    capped = new_target(heavy_a, six_returns, six_eligible, dates[-1], 0.2, LOOKBACK, six_returns.columns)
    assert np.isclose(capped["A"], 0.2, atol=1e-12) # A held at the cap
    assert np.isclose(capped["B"], 0.16, atol=1e-12) # its excess shared equally by the other five

# trading_costs
def test_trading_costs():
    target = pd.Series({"A": 0.6, "B": 0.4})
    weights_start = pd.Series({"A": 0.0, "B": 0.0}) # at start (cash)
    weights_same = pd.Series({"A": 0.6, "B": 0.4}) # already at target
    weights_drift = pd.Series({"A": 0.62, "B": 0.38}) # drifted
    cost_flat = pd.Series({"A": 10, "B": 10}) # 10 bp each
    cost_mixed = pd.Series({"A": 2, "B": 10}) # different costs
    test1 = trading_costs(target, weights_start, cost_flat) # 100% traded at 0.1%
    test2 = trading_costs(target, weights_same, cost_flat) # nothing traded
    test3 = trading_costs(target, weights_drift, cost_flat) # 4% traded at 0.1%
    # 1. always true: cost and turnover are never negative
    for t in [test1, test2, test3]:
        assert t["cost"] >= 0 and t["turnover"] >= 0
    # 2. known answer: cost = traded × bp, turnover = total traded
    assert np.isclose(test1["cost"], 1 * 0.001, atol=1e-12)
    assert np.isclose(test2["cost"], 0, atol=1e-12)
    assert np.isclose(test3["cost"], 0.04 * 0.001, atol=1e-12)
    assert np.isclose(test1["turnover"], 1, atol=1e-12)
    assert np.isclose(test2["turnover"], 0, atol=1e-12)
    assert np.isclose(test3["turnover"], 0.04, atol=1e-12)
    # 3. inputs matter: per-ticker costs change the cost, not the turnover
    test4 = trading_costs(target, weights_drift, cost_mixed)
    assert np.isclose(test4["cost"], 0.02 * 0.0002 + 0.02 * 0.001, atol=1e-12)
    assert np.isclose(test4["turnover"], 0.04, atol=1e-12)

# holding_nav
def test_holding_nav():
    dates = pd.bdate_range("2020-1-1", periods=6)
    trade_day, next_trade_day = dates[0], dates[5]
    weights_equal = pd.Series({"A": 0.5, "B": 0.5})
    close_flat = pd.DataFrame({"A": [100] * 6, "B": [100] * 6}, index=dates) # all prices 100
    open_flat = close_flat.copy()
    # 1. always true: no NaNs, even with an ETF that has no prices and zero weight
    close_na = close_flat.copy()
    close_na["C"] = np.nan
    weights_na = pd.Series({"A": 0.5, "B": 0.5, "C": 0.0})
    test_na = holding_nav(1.0, weights_na, close_na, close_na.copy(), trade_day, next_trade_day)
    assert test_na.notna().all()
    # 2. known answer: flat prices keep the NAV at 1; A doubling lifts it to 1.5
    assert np.isclose(holding_nav(1.0, weights_equal, close_flat, open_flat, trade_day, next_trade_day), 1.0, atol=1e-12).all()
    close_up = close_flat.copy()
    close_up.loc[dates[2]:, "A"] = 200
    test_up = holding_nav(1.0, weights_equal, close_up, open_flat, trade_day, next_trade_day)
    assert np.isclose(test_up.iloc[2:], 1.5, atol=1e-12).all()
    # 3. inputs matter: the starting value scales the NAV
    test_two = holding_nav(2.0, weights_equal, close_up, open_flat, trade_day, next_trade_day)
    assert np.allclose(test_two, 2 * test_up)
    
# drift
def test_drift():
    dates = pd.bdate_range("2020-1-1", periods=6)
    trade_day, next_trade_day = dates[0], dates[5]
    weights_equal = pd.Series({"A": 0.5, "B": 0.5})
    open_flat = pd.DataFrame({"A": [100] * 6, "B": [100] * 6}, index=dates)
    open_up = open_flat.copy()
    open_up.loc[dates[3]:, "A"] = 200
    test_up = drift(1.0, weights_equal, open_up, trade_day, next_trade_day)
    # 1. always true: the new weights sum to 1
    assert np.isclose(test_up["Weights"].sum(), 1, atol=1e-12)
    # 2. known answer: A doubles → NAV 1.5, A's weight 2/3; flat prices change nothing
    assert np.isclose(test_up["NAV"], 1.5, atol=1e-12)
    assert np.isclose(test_up["Weights"]["A"], 2/3, atol=1e-12)
    test_flat = drift(1.0, weights_equal, open_flat, trade_day, next_trade_day)
    assert np.isclose(test_flat["NAV"], 1.0, atol=1e-12)
    assert np.isclose(test_flat["Weights"], [0.5, 0.5], atol=1e-12).all()
    # 3. inputs matter: an ETF with no prices and zero weight stays at 0, with no NaNs
    open_na = open_flat.copy()
    open_na["C"] = np.nan
    weights_na = pd.Series({"A": 0.5, "B": 0.5, "C": 0.0})
    test_na = drift(1.0, weights_na, open_na, trade_day, next_trade_day)
    assert test_na["Weights"].notna().all()
    assert np.isclose(test_na["Weights"], [0.5, 0.5, 0], atol=1e-12).all()

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

    run_aaa = backtest_engine(all_in_aaa, close_fake, open_fake, returns_fake, eligible_fake, rebal_fake, cost_zero, cap=None, lookback=1)
    run_half = backtest_engine(half_half, close_fake, open_fake, returns_fake, eligible_fake, rebal_fake, cost_zero, cap=None, lookback=1)

    # 1. always true: the first trade from cash is 100% turnover; the NAV has no NaNs
    assert np.isclose(run_aaa["Turnover"].iloc[0], 1.0, atol=1e-12)
    assert run_aaa["NAV"].notna().all()
    # 2. known answer: 100% AAA matches holding AAA directly, and never trades again
    first_trade_day = run_aaa["Turnover"].index[0]
    last_nav_day = run_aaa["NAV"].index[-1]
    expected = close_fake.loc[last_nav_day, "AAA"] / open_fake.loc[first_trade_day, "AAA"]
    assert np.isclose(run_aaa["NAV"].iloc[-1], expected, atol=1e-12)
    assert (run_aaa["Turnover"].iloc[1:] < 1e-12).all()
    # 3. inputs matter: costs are charged (once, at 10 bp), and the allocator changes the trading
    run_costs = backtest_engine(all_in_aaa, close_fake, open_fake, returns_fake, eligible_fake, rebal_fake, cost_ten, cap=None, lookback=1)
    assert np.isclose(run_costs["NAV"].iloc[-1], run_aaa["NAV"].iloc[-1] * (1 - 0.001), atol=1e-12)
    assert (run_half["Turnover"].iloc[1:] > 0).all() # 50/50 drifts, so every rebalance trades something
    assert (run_half["Turnover"].iloc[1:] < 0.05).all() # but only a little