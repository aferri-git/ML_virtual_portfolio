# Chaining of various functions into pipelines
    # this will make future code leaner and less prone to mistakes

# Imports
import pandas as pd
from vportfolio import engine,  metrics, trials, data
from vportfolio.params import CAP, LOOKBACK, BACKTEST, TRIALS_LOG, RESEARCH_START, UNIVERSE, BENCHMARK, RISK_FREE, LIVE_RAW, DATA_START

# a pipeline to execute a trial on each new allocator
## will save the return results in a parquet file and update trials log with relevant information for the trial for future evaluation
def trial_pipeline(name, allocator, role, data, cap=CAP, lookback=LOOKBACK, folder=BACKTEST, trials_log = TRIALS_LOG):
    # run the engine
    result = engine.backtest_engine(allocator,
                                    data["close"], data["open"], data["returns"],
                                    data["eligible"], data["rebal_dates"],
                                    data["cost_bp"],
                                    cap, lookback)
    # save NAV and turnover
    nav = result["NAV"]
    turnover = result["Turnover"]
    # run the perofrmance calculator
    performance = metrics.performance_calculator(nav, data["rf_daily"])
    # save the NAV returns in parquet
    path = trials.save_returns(name, nav, folder)
    # make the new row to be logged with all parameters
    new_row = trials.make_new_row(name, allocator, role, performance, turnover, path, cap,lookback)
    # update the logs
    trials.update_trials_log(pd.DataFrame([new_row]), trials_log)
    return new_row # return the information about the trial run

# Data ingestion pipeline
## Will return a dictionary with all data divided as per the names in data/processed
def data_pipeline(raw):
    prices = data.split_prices(raw)
    universe_close = data.apply_fixes(prices["universe_close"])
    benchmark_close = data.apply_fixes(prices["benchmark_close"])
    universe_returns = data.make_returns(universe_close)
    benchmark_returns = data.make_returns(benchmark_close)
    rebuilt = data.make_rebal_dates(universe_returns.index, RESEARCH_START)
    eligible = data.make_eligible(universe_returns)
    rf_daily = data.make_rf_daily(prices["irx"], universe_returns.index)
    cost_bp = data.estimate_costs(raw, UNIVERSE + BENCHMARK)
    return {"benchmark_returns" : benchmark_returns,
            "close" : universe_close,
            "cost_bp" : cost_bp,
            "eligible" : eligible,
            "open" : prices["universe_open"],
            "rebal_dates" : rebuilt,
            "returns" : universe_returns,
            "rf_daily" : rf_daily}

# Will run the three functions and not save anything in case there is an error
## Returns the dataframe of problems, if any, or the downloaded raw data if everything is fine
def live_path(run_date, folder = LIVE_RAW):
    return folder / f"yahoo_prices_{run_date}.parquet"

def download_pipeline(run_date, start = DATA_START, folder = LIVE_RAW, download = data.download_raw):
    path = data.live_path(run_date, folder)
    if path.exists():
        return pd.read_parquet(path)
    raw_data = download(run_date, start)
    problems = data.validate_raw(raw_data, start = start)
    if problems == []:
        data.save_raw(raw_data, run_date, folder)
        return raw_data
    else:
        raise ValueError(f"bad download for {run_date}: {problems}")
