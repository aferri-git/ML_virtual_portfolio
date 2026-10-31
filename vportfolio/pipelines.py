# Chaining of various functions into pipelines
    # this will make future code leaner and less prone to mistakes
    
# Imports
import pandas as pd
import numpy as np
from vportfolio import allocators, engine,  metrics, trials
from vportfolio.params import CAP, LOOKBACK, BACKTEST, TRIALS_LOG

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