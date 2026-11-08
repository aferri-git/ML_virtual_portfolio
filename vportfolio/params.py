# Project-wide universal variables

# Imports
from pathlib import Path

# Paths
## set project root by finding it through test file's own location
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

# Parameters
UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "SHY", "TIP",
            "LQD", "HYG", "EMB", "EMLC", "GLD", "DBC", "XLE", "VNQ",
            "UUP", "FXY", "FXA", "FXF"]
BENCHMARK = ["AGG"]
RISK_FREE = ["^IRX"]

LOOKBACK = 504
CAP = 0.20
RESEARCH_START, RESEARCH_END = "2005-01-01", "2021-12-31"

RESULTS = ROOT / "results"
BACKTEST = RESULTS / "backtest"
TRIALS_LOG = RESULTS / "trials_log.csv"

# Introduced for data loading
RAW_FILE = RAW / "yahoo_prices_2026-10-23.parquet"

# Introduced for testing of data loading
DATA_KEYS = (["close", "open", "returns", "eligible", "rebal_dates", "cost_bp", "rf_daily"])

# Introduced for the run on the holdout period
HOLDOUT = RESULTS / "holdout"
HOLDOUT_LOG = HOLDOUT / "holdout_log.csv"
HOLDOUT_START = "2021-12-31"   # first decision date: its trades happen in January 2022

# Introduced for the live portfolio
DATA_START = "2003-01-01"
COST_OVERRIDES = ["SHY", "IEF", "TLT", "AGG", "TIP", "LQD"] # forced to 2 bp
DATA_FIXES = [("UUP", "2007-06-20", "2007-06-19")] # (ticker, bad date, date to copy from)