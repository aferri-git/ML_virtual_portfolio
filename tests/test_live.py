## Test functions for live.py

# Imports
import pandas as pd
import numpy as np
import pytest
from vportfolio.params import  CAP, LOOKBACK, RAW_FILE
from vportfolio.live import new_days, day_kind, normal_day, decide_targets, benchmark_data, trade_day, day_returns, daily_step
from vportfolio import load_data, allocators, pipelines, engine, live
from tests import helpers

# new_days
def test_new_days():
    dates = pd.bdate_range("2026-11-25", "2026-12-04")
    # 1. always true: in date order, and only dates from the download
    result = new_days(dates, pd.Timestamp("2026-12-02"))
    assert result.is_monotonic_increasing and result.isin(dates).all()
    # 2. known answer: first run starts at the live start; later runs start after the last recorded day
    assert list(new_days(dates, None)) == list(pd.bdate_range("2026-12-01", "2026-12-04"))
    assert list(result) == [pd.Timestamp("2026-12-03"), pd.Timestamp("2026-12-04")]
    assert len(new_days(dates, pd.Timestamp("2026-12-04"))) == 0
    # 3. inputs matter: a later start drops the earlier days; a download that ends before the start gives nothing
    assert new_days(dates, None, start = "2026-12-03")[0] == pd.Timestamp("2026-12-03")
    november = pd.bdate_range("2026-11-02", "2026-11-27")
    assert len(new_days(november, None)) == 0

# day_kind
def test_day_kind():
    test_dates = pd.bdate_range("2026-10-28", "2026-12-04")
    test_rebal = pd.Series(pd.to_datetime(["2026-10-30", "2026-11-30"])) # Friday, Monday
    kinds = day_kind(test_dates, test_rebal, test_dates)
    # 1. always true: one label per day to process, and only the three labels
    assert kinds.index.equals(test_dates)
    assert set(kinds.unique()) <= {"normal", "decision", "trade"}
    # 2. known answer: month-ends are decisions; the next trading day is a trade, across a weekend too
    assert kinds["2026-10-30"] == "decision" and kinds["2026-11-02"] == "trade" # Fri → Mon
    assert kinds["2026-11-30"] == "decision" and kinds["2026-12-01"] == "trade"
    assert (kinds == "normal").sum() == len(test_dates) - 4
    # 3. inputs matter: a trade day is found even when its decision day is not among the days to process
    assert list(day_kind(test_dates[test_dates == "2026-12-01"], test_rebal, test_dates)) == ["trade"]
    # 3. inputs matter: a download ending on the decision day labels it, with no trade day yet and no error
    short = test_dates[test_dates <= "2026-11-30"]
    assert day_kind(short[-1:], test_rebal, short).iloc[0] == "decision"
    # 3. inputs matter: no decision days, every day is normal
    no_rebal = pd.Series([], dtype = "datetime64[ns]")
    assert (day_kind(test_dates, no_rebal, test_dates) == "normal").all()

# normal_day
def test_normal_day():
    test_holdings = pd.Series({"A": 0.5, "B": 0.5})
    test_returns = pd.Series({"A": 0.10, "B": 0.0, "C": 0.30}) # C is not held
    day_return, new_holdings = normal_day(test_holdings, test_returns)
    # 1. always true: new weights sum to 1, same tickers as before
    assert np.isclose(new_holdings.sum(), 1)
    assert list(new_holdings.index) == ["A", "B"]
    # 2. known answer: 50/50, A +10%, B flat → +5%; A grows to 0.55/1.05
    assert np.isclose(day_return, 0.05)
    assert np.isclose(new_holdings["A"], 0.55 / 1.05)
    # 2. known answer: two days in a row equal the engine's buy-and-hold (1.10 × 1.10 and 0.90 → 1.055)
    r2, h2 = normal_day(new_holdings, pd.Series({"A": 0.10, "B": -0.10}))
    assert np.isclose((1 + day_return) * (1 + r2), 0.5 * 1.21 + 0.5 * 0.90)
    # 3. inputs matter: a ticker not held does not change the result
    assert np.isclose(normal_day(test_holdings, test_returns.drop("C"))[0], day_return)
    # 3. inputs matter: a held ticker with no return stops with an error
    with pytest.raises(ValueError):
        normal_day(test_holdings, pd.Series({"A": 0.10, "B": np.nan}))

@pytest.mark.skipif(not RAW_FILE.exists(), reason = "needs the local research data")
# decide_targets
def test_decide_targets():
    # access processed data for testing
    raw = pd.read_parquet(RAW_FILE)
    processed = pipelines.data_pipeline(raw)
    bench = load_data.load_benchmark()
    test_day = processed["rebal_dates"].iloc[-1]
    nco = decide_targets(allocators.nco, processed, test_day)
    ew = decide_targets(allocators.equal_weights, processed, test_day)
    sf = decide_targets(allocators.sixty_forty, bench, test_day, cap = None)
    # 1. always true: every portfolio is fully invested, with no negative weights
    for name, w in [("nco", nco), ("ew", ew), ("sf", sf)]:
        assert np.isclose(w.sum(), 1) and (w >= 0).all(), (name, w.sum(), w.min())
    # 2. known answers: NCO respects the cap; 60/40 is 60% SPY and 40% AGG
    assert nco.max() <= CAP + 1e-9
    assert np.isclose(sf["SPY"], 0.6) and np.isclose(sf["AGG"], 0.4)
    # 2. known answer: identical to calling the engine directly, as the backtest does
    assert nco.equals(engine.new_target(allocators.nco, processed["returns"], processed["eligible"],
                                        test_day, CAP, LOOKBACK, processed["returns"].columns))
    # 3. inputs matter: tickers not eligible get nothing; different allocators give different weights
    not_eligible = ~processed["eligible"].loc[test_day]
    assert (nco[not_eligible] == 0).all()
    assert not nco.equals(ew)

# benchmark_data
def test_benchmark_data():
    fake = helpers.make_fake_raw("2022-03-15")
    b = benchmark_data(fake)
    # 1. always true: four tables, each with SPY and AGG only, on the raw data's dates
    assert set(b.keys()) == {"close", "open", "returns", "eligible"}
    for key, table in b.items():
        assert table.columns.tolist() == ["SPY", "AGG"] and table.index.equals(fake.index), key
    # 2. known answer: same values as the raw data; returns are the closes' daily change
    pd.testing.assert_frame_equal(b["close"], fake["Close"][["SPY", "AGG"]])
    assert np.isclose(b["returns"]["SPY"].iloc[1], fake["Close"]["SPY"].iloc[1] / fake["Close"]["SPY"].iloc[0] - 1)
    # 3. inputs matter: other tickers give other columns
    assert benchmark_data(fake, ["SPY"])["close"].columns.tolist() == ["SPY"]

# trade_day
def test_trade_day():
    h = pd.Series({"A": 0.5, "B": 0.5})
    t = pd.Series({"A": 0.6, "B": 0.4})
    on = pd.Series({"A": 0.02, "B": 0.0})
    intra = pd.Series({"A": 0.01, "B": -0.01})
    bp = pd.Series({"A": 10.0, "B": 10.0})
    r, new_h, trades = trade_day(h, t, on, intra, bp)

    # 1. always true: weights at the close sum to 1; the trades end exactly at the targets; costs are not negative
    assert np.isclose(new_h.sum(), 1)
    assert np.allclose(trades.set_index("ticker")["weight_after"], t)
    assert (trades["cost"] >= 0).all()
    # 2. known answer: overnight +1%, then cost on 0.19 of turnover at 10 bp, then +0.2% during the day
    open_w = pd.Series({"A": 0.51, "B": 0.50}) / 1.01
    cost = (t - open_w).abs().sum() * 0.001
    assert np.isclose(r, 1.01 * (1 - cost) * 1.002 - 1)
    assert np.isclose(trades["cost"].sum(), cost)
    # 2. known answer: the first day (nothing held) matches the backtest engine's first trade day
    day1, day2 = pd.Timestamp("2026-12-01"), pd.Timestamp("2026-12-02")
    open_p = pd.DataFrame({"A": [100.0, 100.0], "B": [100.0, 100.0]}, index = [day1, day2])
    close_p = pd.DataFrame({"A": [101.0, 101.0], "B": [99.0, 99.0]}, index = [day1, day2])
    r0, h0, trades0 = trade_day(pd.Series(dtype = float), t, on, intra, bp)
    engine_cost = engine.trading_costs(t, pd.Series(0.0, index = t.index), bp)["cost"]
    engine_nav = engine.holding_nav(1 - engine_cost, t, close_p, open_p, day1, day2).iloc[0]
    assert np.isclose(1 + r0, engine_nav)
    assert len(trades0) == 2 and (trades0["weight_before"] == 0).all()
    # 3. inputs matter: no overnight move and targets equal to holdings → no trades, no cost
    r_same, _, trades_same = trade_day(h, h, pd.Series({"A": 0.0, "B": 0.0}), intra, bp)
    assert trades_same.empty and np.isclose(r_same, 0.0)
    # 3. inputs matter: higher costs give a lower return
    r_dear, _, _ = trade_day(h, t, on, intra, bp * 10)
    assert r_dear < r
    # 3. inputs matter: a held ticker that is no longer targeted is sold, and it is not lost
    h_c = pd.Series({"A": 0.4, "B": 0.4, "C": 0.2})
    r_c, h_c_new, trades_c = trade_day(h_c, t, pd.Series({"A": 0.0, "B": 0.0, "C": 0.0}), intra, bp)
    sold = trades_c.set_index("ticker").loc["C"]
    assert sold["weight_before"] == 0.2 and sold["weight_after"] == 0.0
    assert "C" not in h_c_new.index

# day_returns
def test_day_returns():
    days = pd.bdate_range("2026-12-01", periods = 3)
    close = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index = days)
    open_ = pd.DataFrame({"A": [100.0, 105.0, 110.0]}, index = days)
    prices = {"close": close, "open": open_, "returns": close.pct_change(fill_method = None)}
    overnight, intraday, full = day_returns(prices, days[1])
    # 1. always true: overnight and intraday chain into the full day: (1 + o) × (1 + i) = 1 + full
    assert np.isclose((1 + overnight["A"]) * (1 + intraday["A"]), 1 + full["A"])
    # 2. known answer: close 100 → open 105 is +5%; open 105 → close 110 is +4.76%; close to close +10%
    assert np.isclose(overnight["A"], 0.05)
    assert np.isclose(intraday["A"], 110 / 105 - 1)
    assert np.isclose(full["A"], 0.10)
    # 3. inputs matter: the next day uses its own previous close (110), not the first day's
    overnight_3, _, _ = day_returns(prices, days[2])
    assert np.isclose(overnight_3["A"], 0.0)          # close 110 → open 110

# daily_step
def test_daily_step():
    raw, processed, bench, portfolios = helpers.make_fake_live()
    start = "2021-06-01" # the trading day after the May 2021 month-end
    con = helpers.make_store()
    done = daily_step(con, processed, bench, portfolios, start)
    navs = con.execute("SELECT * FROM nav").df()

    # 1. always true: one NAV per day and portfolio; weights sum to 1 every day; a second run adds nothing
    assert len(navs) == len(done) * 2
    sums = con.execute("SELECT date, portfolio, sum(weight) AS total FROM holdings GROUP BY date, portfolio").df()
    assert np.allclose(sums["total"], 1)
    assert len(daily_step(con, processed, bench, portfolios, start)) == 0
    assert con.execute("SELECT count(*) FROM nav").fetchone()[0] == len(navs)

    # 2. known answer: the first day is a trade day; the 60/40 trades SPY and AGG and targets 0.6 / 0.4
    first_trades = con.execute("SELECT * FROM trades WHERE trade_date = '2021-06-01' AND portfolio = 'sf'").df()
    assert set(first_trades["ticker"]) == {"SPY", "AGG"}
    sf_targets = con.execute("SELECT ticker, weight FROM targets WHERE portfolio = 'sf'").df()
    assert np.allclose(sf_targets.groupby("ticker")["weight"].first().loc[["SPY", "AGG"]], [0.6, 0.4])

    # 3. inputs matter: running in two parts gives the same NAVs as one run (the daily case)
    con_parts = helpers.make_store()
    short_raw = raw.loc[:"2021-09-15"]
    daily_step(con_parts, pipelines.data_pipeline(short_raw), benchmark_data(short_raw), portfolios, start)
    daily_step(con_parts, processed, bench, portfolios, start)
    parts = con_parts.execute("SELECT * FROM nav ORDER BY date, portfolio").df()
    pd.testing.assert_frame_equal(parts, navs.sort_values(["date", "portfolio"]).reset_index(drop = True))

    # 3. inputs matter: a start that is not a trade day stops with an error and writes nothing
    con_bad = helpers.make_store()
    with pytest.raises(ValueError):
        daily_step(con_bad, processed, bench, portfolios, "2021-06-02")
    assert con_bad.execute("SELECT count(*) FROM nav").fetchone()[0] == 0