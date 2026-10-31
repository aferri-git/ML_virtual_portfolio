# Functions to store data from trials

# Imports
import pandas as pd
from vportfolio.params import CAP, LOOKBACK, TRIALS_LOG, BACKTEST
from vportfolio.allocators import equal_weights

# Storing of NAV returns data from backtesting of different allocators into a parquet file
def save_returns(name, nav, folder = BACKTEST):
    folder.mkdir(parents=True, exist_ok=True) # creates the folder if it doesn't exist
    strat_returns = nav.pct_change(fill_method=None).dropna().to_frame(name)
    path = folder / f"{name}_returns.parquet"
    strat_returns.to_parquet(path) # in this case we need to overwrite the file if it already exists
    if not pd.read_parquet(path).equals(strat_returns): # check if the file was saved correctly
        raise ValueError(f"{name} did not save correctly")
    return path # returns the file's location for reference

# Creates a new row for the trials logs
## will return a dictionary with relevant information about the trial run
def make_new_row(name, allocator, role, performance, turnover, returns_file, cap = CAP, lookback = LOOKBACK):
    now = pd.Timestamp.today().strftime("%Y-%m-%d")
    row = {"date": now,
           "name": name,
           "allocator": allocator.__name__,
           "settings": f"cap={cap}, lookback={lookback}",
           "role": role,
           "ann_return": performance["ann_return"],
           "ann_vol": performance["ann_vol"],
           "sharpe": performance["sharpe"],
           "avg_turnover" : turnover.mean(), 
           "max_drawdown" : performance["max_drawdown"],
           "returns_file": str(returns_file)}                                          
    return row

# Add new strategy runs to the trials log, replacing older rows with the same name
## will return the updated log as a dataframe
def update_trials_log(new_rows, path=TRIALS_LOG):
    # load the existing log, or start empty if the file is missing or empty
    if path.exists() and path.stat().st_size > 0:
        log = pd.read_csv(path)
    else:
        log = pd.DataFrame()

    # remove old rows for the strategies being written now
    if not log.empty:
        log = log[~log["name"].isin(new_rows["name"])] # new log is old log without the rows that appear in new_row

    # add the new rows below the remaining ones
    log = pd.concat([log, new_rows], ignore_index=True)

    # save the whole log, overwriting the file
    log.to_csv(path, index=False)
    return log