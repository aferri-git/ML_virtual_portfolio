## Project-wide universal variables

## Imports
from pathlib import Path

## Paths
# set project root by finding it through test file's own location
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

## Parameters
UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "SHY", "TIP",
            "LQD", "HYG", "EMB", "EMLC", "GLD", "DBC", "XLE", "VNQ",
            "UUP", "FXY", "FXA", "FXF"]
BENCHMARK = ["AGG"]

LOOKBACK = 504
CAP = 0.20
RESEARCH_START, RESEARCH_END = "2005-01-01", "2021-12-31"

RESULTS = ROOT / "results"
BACKTEST = RESULTS / "backtest"
