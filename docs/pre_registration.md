# Pre-registration (written 21-10-2026, before any backtest)

## Hypothesis
A systematic multi-asset allocator using denoised, cluster-based allocation (NCO, HRP)
delivers better risk-adjusted returns than equal-weight and 60/40 allocations,
net of costs. The best-supported allocator will run the live paper portfolio.

## Universe

| Exposure | Benchmark index | ETF proxy ticker |
|---|---|---|
| US large cap equity | S&P 500 | SPY |
| US growth equity | Nasdaq-100 | QQQ |
| US small cap equity | Russell 2000 | IWM |
| Developed ex-US equity | MSCI EAFE | EFA |
| EM equity | MSCI Emerging Markets | EEM |
| US Treasuries, long | ICE US Treasury 20+ Year | TLT |
| US Treasuries, mid | ICE US Treasury 7-10 Year | IEF |
| US Treasuries, short | ICE US Treasury 1-3 Year | SHY |
| US inflation-linked | ICE US Treasury Inflation Linked | TIP |
| US IG credit | iBoxx USD Liquid Investment Grade | LQD |
| US HY credit | iBoxx USD Liquid High Yield | HYG |
| EM hard currency debt | JPM EMBI Global Core | EMB |
| EM local currency debt | JPM GBI-EM Global Core | EMLC |
| Gold | Gold spot (LBMA) | GLD |
| Broad commodities | DBIQ Optimum Yield Diversified Commodity | DBC |
| Energy (equity proxy) | Energy Select Sector | XLE |
| US real estate | MSCI US IMI Real Estate 25/50 | VNQ |
| US dollar | US Dollar Index (DXY) futures | UUP |
| Japanese yen | JPY/USD spot | FXY |
| Australian dollar | AUD/USD spot | FXA |
| Swiss franc | CHF/USD spot | FXF |

**Selection rule:** liquid US-listed ETFs with long daily histories (most from 2007
or earlier), covering the main asset classes and their characteristics:
- Equities: US large cap, growth, small cap, developed ex-US, EM
- Duration: long / mid / short US Treasuries
- Inflation: US TIPS
- Credit: US investment grade, US high yield
- EM debt: hard currency (EMB), local currency (EMLC)
- Commodities: gold, broad basket, energy (equity proxy via XLE)
- Real estate: US REITs
- FX: USD (DXY), JPY, AUD, CHF

**Rationale:**
- Cross-asset coverage so clustering can recover economic groups
- Deliberately correlated groups (SPY/QQQ/IWM, TLT/IEF/SHY, LQD/HYG) to test
  NCO/HRP as a more stable alternative to Markowitz
- FX chosen by behaviour: DXY carries the dollar factor; single currencies are added
  only if they behave differently and have a small DXY weight. Hence JPY and CHF
  (safe havens) and AUD (risk-on/commodities); EUR excluded as ~58% of DXY
- EM debt and FX reflect my PM background

## Data sources
- ETF prices: Yahoo Finance via yfinance (daily, adjusted for dividends and splits)
- Risk-free rate: 13-week US T-bill yield (^IRX), Yahoo Finance
- AGG downloaded as a benchmark component only; never held by the allocators
- Further providers (e.g. FRED for macro features) may be added later;
  any addition will be logged in decision_log.md

## Periods
- Data from: 2003-01-01 (gives the 2-year lookback for the first rebalance)
- Research (backtest): 2005-01-01 to 2021-12-31
- Holdout (used once): 2022-01-01 to the live start date
- Live paper trading: from the live start date onwards
- Expanding universe: an ETF becomes eligible once it has 504 trading days of history;
  no ETF is used before it existed

## Rules
- Rebalance: monthly, last trading day
- Execution: next trading day's open
- Trading costs by average daily dollar volume over the research period:
  >$1bn = 2 bp, $100m-$1bn = 5 bp, <$100m = 10 bp. Sensitivity check with costs doubled
- Positions: long-only, no leverage, max 20% per ETF (UCITS single-fund limit; stops
  risk-based allocators piling into the lowest-volatility asset)
- Covariance lookback: 2 years of daily returns (504 days, ~24x the number of assets)
- Sharpe ratio: annualised, in excess of the 13-week T-bill yield
- Benchmarks, rebalanced monthly with the same costs:
  - Equal weight of the eligible universe
  - 60/40: 60% SPY / 40% AGG
- Success (holdout period): net Sharpe above both benchmarks and Deflated Sharpe
  Ratio >= 0.95; max drawdown reported

## Known issues
- Early years have a narrower universe 
- EMLC starts in 2010 and joins the universe in 2012