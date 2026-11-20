# Run the live paper portfolios for one day:
    # download
    # validate
    # save
    # process
    # trade record
    # then one line in the log

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from vportfolio import data, live, pipelines, store
from vportfolio.params import LIVE_DB, LIVE_RAW, LIVE_START, ROOT, LIVE_PORTFOLIOS, DATA_START

LOG_FILE = ROOT / "logs" / "daily.log"

# One daily run: download and check the prices, process them, update the store
## returns a short summary: run date, days processed, latest NAVs
def run(run_date, db_path = LIVE_DB, start = LIVE_START, raw_folder = LIVE_RAW,
        download = data.download_raw, data_start = DATA_START, portfolios = LIVE_PORTFOLIOS):
    raw = pipelines.download_pipeline(run_date, data_start, raw_folder, download)
    processed = pipelines.data_pipeline(raw)
    bench = data.benchmark_data(raw)
    con = store.open_store(db_path)
    try:
        done = live.daily_step(con, processed, bench, portfolios, start)
        navs = store.last_nav(con).set_index("portfolio")["nav"].round(4).to_dict()
    finally:
        con.close()
    return {"run_date": str(run_date), "days": [d.date().isoformat() for d in done], "navs": navs}

# read the options from the command line (all optional: by default, today and the real store)
def main():
    parser = argparse.ArgumentParser(description = "Run the live paper portfolios for one day")
    parser.add_argument("--date", default = date.today().isoformat(), help = "run date, YYYY-MM-DD")
    parser.add_argument("--db", default = str(LIVE_DB), help = "path of the store")
    parser.add_argument("--start", default = LIVE_START, help = "first live day")
    parser.add_argument("--raw", default = str(LIVE_RAW), help = "folder for the daily downloads")
    args = parser.parse_args()
    # every run adds lines to the log file
    LOG_FILE.parent.mkdir(parents = True, exist_ok = True)
    logging.basicConfig(filename = LOG_FILE, level = logging.INFO,
                        format = "%(asctime)s %(levelname)s %(message)s")
    try:
        summary = run(args.date, Path(args.db), args.start, Path(args.raw))
        logging.info(f"run {summary['run_date']} [{Path(args.db).name}]: "
             f"{len(summary['days'])} new day(s) {summary['days']}, NAV {summary['navs']}")
    except Exception:
        logging.exception(f"run {args.date} [{Path(args.db).name}] failed")
        sys.exit(1)


if __name__ == "__main__":
    main()