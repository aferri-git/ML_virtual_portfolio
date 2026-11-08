# Test functions for data.py

# imports
import pandas as pd
import numpy as np
from vportfolio.data import split_prices, apply_fixes, make_returns, make_eligible, make_rebal_dates, make_rf_daily, estimate_costs
from vportfolio.params import BENCHMARK, BENCHMARK, RISK_FREE, RISK_FREE, UNIVERSE

# split_prices
def test_split_prices():
    # fake raw data shaped like the Yahoo download: ("Close", ticker) and ("Open", ticker) columns
    days = pd.bdate_range("2024-01-01", periods=3)
    all_tickers = UNIVERSE + BENCHMARK + RISK_FREE
    rng = np.random.default_rng(0)
    close_fake = pd.DataFrame(rng.uniform(10, 100, (3, len(all_tickers))), index=days, columns=all_tickers)
    open_fake = pd.DataFrame(rng.uniform(10, 100, (3, len(all_tickers))), index=days, columns=all_tickers)
    raw_fake = pd.concat({"Close": close_fake, "Open": open_fake}, axis=1)
    parts = split_prices(raw_fake)
    # . always true: the four parts, each on the same dates as the raw data
    assert set(parts.keys()) == {"universe_close", "universe_open", "benchmark_close", "irx"}
    assert all(p.index.equals(days) for p in parts.values())
    # 2. known answer: the right tickers and the same values as the raw data
    assert list(parts["universe_close"].columns) == UNIVERSE
    assert list(parts["universe_open"].columns) == UNIVERSE
    assert list(parts["benchmark_close"].columns) == BENCHMARK + ["SPY"]
    assert parts["universe_close"].equals(close_fake[UNIVERSE])
    assert parts["universe_open"].equals(open_fake[UNIVERSE])
    assert parts["irx"].equals(close_fake[RISK_FREE[0]])
    # 3. inputs matter: the parts are copies, so changing one leaves the raw data untouched
    parts["universe_close"].iloc[0, 0] = -1.0
    assert raw_fake["Close"][UNIVERSE].iloc[0, 0] != -1.0
    
# apply_fixes
def test_apply_fixes():
    days = pd.bdate_range("2007-06-18", periods=4) # 18, 19, 20, 21 June 2007
    close = pd.DataFrame({"UUP": [25.0, 25.1, 99.0, 25.2], # 20 June: the bad price
                        "SPY": [150.0, 151.0, 152.0, 153.0]}, index=days)
    original = close.copy()
    fixed = apply_fixes(close)
    # 1. always true: same shape, and the input is not modified
    assert fixed.shape == close.shape and close.equals(original)
    # 2. known answer: the bad UUP price takes the previous day's close
    assert fixed.loc["2007-06-20", "UUP"] == 25.1
    # 3. inputs matter: only that one cell changes; data without that date passes through unchanged
    assert (fixed != close).sum().sum() == 1
    recent = close.copy()
    recent.index = pd.bdate_range("2026-01-05", periods=4)
    assert apply_fixes(recent).equals(recent)

# make_returns
def test_make_returns():
    days = pd.bdate_range("2024-01-01", periods=4)
    close = pd.DataFrame({"A": [100.0, 110.0, 99.0, 99.0],
                        "B": [None, None, 50.0, 55.0]}, index=days)  # B starts trading later
    returns = make_returns(close)
    # 1. always true: same dates and tickers as the prices; the first day has no return
    assert returns.index.equals(close.index) and returns.columns.equals(close.columns)
    assert returns.iloc[0].isna().all()
    # 2. known answer: 100 → 110 is +10%, 110 → 99 is −10%, unchanged is 0
    assert np.allclose(returns["A"].iloc[1:], [0.10, -0.10, 0.0])
    # 3. inputs matter: no return before an ETF exists (no filling); its first return is on its second day
    assert returns["B"].iloc[:3].isna().all()
    assert np.isclose(returns["B"].iloc[3], 0.10)
    
# make_eligible
def test_make_eligible():
    days = pd.bdate_range("2024-01-01", periods=6)
    returns = pd.DataFrame({"A": [None, 0.01, 0.01, 0.01, 0.01, 0.01],
                            "B": [None, None, None, None, 0.01, 0.01]}, index=days)  # B starts later
    eligible = make_eligible(returns, lookback=3)
    # 1. always true: True/False only; once eligible, an ETF stays eligible
    assert eligible.dtypes.eq(bool).all()
    assert (eligible.astype(int).diff().fillna(0) >= 0).all().all()
    # 2. known answer: A reaches 3 returns on day 4; B only has 2, so it is never eligible
    assert list(eligible["A"]) == [False, False, False, True, True, True]
    assert not eligible["B"].any()
    # 3. inputs matter: a shorter lookback makes A eligible a day sooner
    assert make_eligible(returns, lookback=2)["A"].iloc[2] and not eligible["A"].iloc[2]
    
# make_rebal_dates
def test_make_rebal_dates():
    days = pd.bdate_range("2024-01-01", "2024-03-15") # data stops mid-March
    dates = make_rebal_dates(days, "2024-01-01")
    # 1. always true: sorted, and every date is a real day in the data
    assert dates.is_monotonic_increasing and dates.isin(days).all()
    # 2. known answer: January and February month-ends; March incomplete, so dropped
    assert list(dates) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]
    # 3. inputs matter: data reaching March's last business day keeps March
    full = pd.bdate_range("2024-01-01", "2024-03-29")
    assert make_rebal_dates(full, "2024-01-01").iloc[-1] == pd.Timestamp("2024-03-29")
    
# make_rf_daily
def test_make_rf_daily():
    days = pd.bdate_range("2024-01-01", periods=5)
    irx = pd.Series([5.0, None, 5.0, 4.0, 4.0], index=days) # a missing day, then a lower yield
    rf = make_rf_daily(irx, days)
    # 1. always true: no gaps on trading days
    assert rf.notna().all()
    # 2. known answer: 5% a year → 0.05 / 252 a day; the missing day takes the previous value
    assert np.isclose(rf.iloc[0], 0.05 / 252) and np.isclose(rf.iloc[1], 0.05 / 252)
    # 3. inputs matter: a lower yield gives a lower rate
    assert rf.iloc[3] < rf.iloc[0]
    
# estimate_costs
def test_estimate_costs():
    # fake raw data shaped like the Yahoo download: ("Close", ticker) and ("Volume", ticker) columns
    days = pd.bdate_range("2005-01-03", "2023-12-29")
    close = pd.DataFrame(1.0, index=days, columns=["A", "B", "C", "SHY"])   # price 1 → dollar volume = volume
    volume = pd.DataFrame({"A": 50e6, "B": 500e6, "C": 2e9, "SHY": 50e6}, index=days)
    raw_fake = pd.concat({"Close": close, "Volume": volume}, axis=1)
    tickers = ["A", "B", "C", "SHY"]

    cost = estimate_costs(raw_fake, tickers, overrides=["SHY"])
    # 1. always true: only the three pre-registered levels
    assert set(cost.unique()) <= {2, 5, 10}
    # 2. known answers: $50m → 10 bp, $500m → 5 bp, $2bn → 2 bp; SHY overridden to 2 bp despite low volume
    assert cost["A"] == 10 and cost["B"] == 5 and cost["C"] == 2 and cost["SHY"] == 2
    # 3. inputs matter: volume after the research period is ignored
    raw_later = raw_fake.copy()
    raw_later.loc["2022-01-01":, ("Volume", "A")] = 1e12              # A becomes hugely liquid after 2021
    assert estimate_costs(raw_later, tickers, overrides=["SHY"])["A"] == 10