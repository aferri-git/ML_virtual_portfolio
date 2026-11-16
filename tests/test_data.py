# Test functions for data.py

# imports
import pandas as pd
import numpy as np
from vportfolio.data import split_prices, apply_fixes, make_returns, make_eligible, make_rebal_dates, make_rf_daily, estimate_costs, validate_raw, save_raw
from tests.helpers import calm_fake_raw, make_fake_raw
from vportfolio.params import BENCHMARK, RISK_FREE, RISK_FREE, UNIVERSE

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
    # 3. inputs matter: a table without the ticker (the benchmark has no UUP) passes through unchanged
    other_close = pd.DataFrame({"AGG": [100.0, 101.0, 102.0], "SPY": [150.0, 151.0, 152.0]},
                               index = pd.to_datetime(["2007-06-18", "2007-06-19", "2007-06-20"]))
    before = other_close.copy()
    other_fixed = apply_fixes(other_close, [("UUP", "2007-06-20", "2007-06-19")])
    pd.testing.assert_frame_equal(other_fixed, before)

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

# validate_raw
def has(problems, text):
    return any(text in problem for problem in problems)

def test_validate_raw():
    raw = calm_fake_raw()
    start = "2019-01-01" # the fake history starts in 2019, not at DATA_START
    last = raw.index[-1]
    # 1. always true: clean data gives no problems
    assert validate_raw(raw, start = start) == []
    # 2. known answer: break one thing at a time, that problem is reported
    no_ticker = raw.drop(columns = [("Close", "EEM")])
    assert has(validate_raw(no_ticker, start = start), "missing columns")
    twice = pd.concat([raw, raw.iloc[[-1]]])
    assert has(validate_raw(twice, start = start), "appear twice")
    reversed_dates = raw.iloc[::-1]
    assert has(validate_raw(reversed_dates, start = start), "not in order")
    late = validate_raw(raw, start = "2010-01-01")
    assert len(late) == 1 and has(late, "history starts late")
    no_close = raw.copy()
    no_close.loc[last, ("Close", "TLT")] = np.nan
    problems = validate_raw(no_close, start = start)
    assert len(problems) == 1 and has(problems, "last row has no Close")
    zero_price = raw.copy()
    zero_price.loc[raw.index[5], ("Low", "XLE")] = 0
    problems = validate_raw(zero_price, start = start)
    assert len(problems) == 1 and has(problems, "Low at or below zero")
    jump = raw.copy()
    jump.loc[last, ("Close", "UUP")] *= 3
    problems = validate_raw(jump, start = start)
    assert len(problems) == 1 and has(problems, "last day move")
    # 3. inputs matter: a 10% move passes a loose limit and fails a tight one
    move = raw.copy()
    move.loc[last, ("Close", "EEM")] = raw["Close"]["EEM"].iloc[-2] * 1.10
    assert validate_raw(move, start = start, max_move = 0.30) == []
    assert has(validate_raw(move, start = start, max_move = 0.05), "last day move")
    # 3. inputs matter: the rate ticker may be empty on the last row (bond holiday)
    no_rate = raw.copy()
    for field in ["Open", "High", "Low", "Close", "Volume"]:
        no_rate.loc[last, (field, "^IRX")] = np.nan
    assert validate_raw(no_rate, start = start) == []

# save_raw
def test_save_raw(tmp_path):
    raw = make_fake_raw("2022-03-15")
    path = save_raw(raw, "2026-12-01", tmp_path)
    # 1. always true: the file exists, in the folder we passed
    assert path.exists() and path.parent == tmp_path
    # 2. known answer: the name carries the run date, and reading it back gives the same table
    assert path.name == "yahoo_prices_2026-12-01.parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(path), raw, check_freq = False)
    # 3. inputs matter: another date makes another file
    other_path = save_raw(raw, "2026-12-02", tmp_path)
    assert other_path != path and len(list(tmp_path.iterdir())) == 2
    # 3. inputs matter: saving a different table on the same date keeps the first file
    save_raw(raw * 2, "2026-12-01", tmp_path)
    pd.testing.assert_frame_equal(pd.read_parquet(path), raw, check_freq = False)
