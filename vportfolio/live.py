# Live trading engine
    ## 1. Know where the last run stopped
    ## 2. Know what kind of day it is (i.e. last day of the month)
    ## 3. Valuing the portfolio as it moves on normal days
    ## 4. Running the allocators live on decision days
    ## 5. Trading after decision days
    # 6. Recording results

# Imports
import pandas as pd
from vportfolio.params import LIVE_START, CAP, LOOKBACK, LIVE_PORTFOLIOS
from vportfolio import engine, allocators, store

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

# helper function to build the three kinds of returns a day needs
## returns the three values fo overnight (yesterday's close and today's open), intraday (today's open and close) and full (yesterday's and today's close)
def day_returns(data, day):
    close, open_ = data["close"], data["open"]
    previous_close = close.iloc[close.index.get_loc(day) - 1]
    overnight = open_.loc[day] / previous_close - 1
    intraday = close.loc[day] / open_.loc[day] - 1
    full = data["returns"].loc[day]
    return overnight, intraday, full

# Process every trading day not yet recorded, for every live portfolio
## returns the days that were processed
def daily_step(con, processed, bench, portfolios = LIVE_PORTFOLIOS, start = LIVE_START):
    # 1. where each portfolio stopped
    last = store.last_date(con) # get last date
    navs = store.last_nav(con).set_index("portfolio")["nav"].to_dict() # get last nav
    holdings = {name: rows.set_index("ticker")["weight"] # gets the weights into a dictionary
                for name, rows in store.last_holdings(con).groupby("portfolio")}
    # 2. which days are new, and what kind each one is
    dates = processed["returns"].index
    rebal = processed["rebal_dates"]
    todo = new_days(dates, last, start)
    kinds = day_kind(todo, rebal, dates)
    portfolio_data = {}
    for name, settings in portfolios.items():
        if settings["allocator"] == "sixty_forty":
            portfolio_data[name] = bench # the 60/40 runs on SPY/AGG data
        else:
            portfolio_data[name] = processed # everyone else runs on the universe
    # here we start he loop of actions
    for day in todo:
        nav_rows, holding_rows, target_rows, trade_rows = [], [], [], []
        # 3. every portfolio on this day
        for name, settings in portfolios.items():
            pdata = portfolio_data[name]
            held = holdings.get(name, pd.Series(dtype = float))
            nav = navs.get(name, 1.0)
            overnight, intraday, full = day_returns(pdata, day)
            if kinds[day] == "trade":
                allocator = getattr(allocators, settings["allocator"])
                decision = rebal[rebal < day].iloc[-1]
                targets = decide_targets(allocator, pdata, decision, settings["cap"], settings["lookback"])
                r, new_held, trades = trade_day(held, targets, overnight, intraday, processed["cost_bp"])
                target_rows.append(pd.DataFrame({"decision_date": decision, "portfolio": name,
                                                 "ticker": targets.index, "weight": targets.values}))
                trade_rows.append(trades.assign(trade_date = day, portfolio = name))
            else:
                if held.empty:
                    raise ValueError(f"{name} holds nothing on {day.date()}: the first live day must be a trade day")
                r, new_held = normal_day(held, full)
            nav = nav * (1 + r)
            navs[name], holdings[name] = nav, new_held
            nav_rows.append({"date": day, "portfolio": name, "nav": nav, "daily_return": r})
            holding_rows.append(pd.DataFrame({"date": day, "portfolio": name,
                                              "ticker": new_held.index, "weight": new_held.values}))

        # 4. the returns used that day, for every ticker with a price (SPY appears in both sets, kept once)
        used = pd.concat([processed["returns"].loc[day], bench["returns"].loc[day]])
        used = used[~used.index.duplicated()].dropna()
        return_rows = pd.DataFrame({"date": day, "ticker": used.index, "daily_return": used.values})
        # 5. write the whole day at once: all tables, or none
        con.begin()
        try:
            store.append_rows(con, "nav", pd.DataFrame(nav_rows))
            store.append_rows(con, "holdings", pd.concat(holding_rows))
            store.append_rows(con, "asset_returns", return_rows)
            if target_rows:
                store.append_rows(con, "targets", pd.concat(target_rows))
                store.append_rows(con, "trades", pd.concat(trade_rows))
            con.commit()
        except Exception:
            con.rollback()
            raise
    return todo