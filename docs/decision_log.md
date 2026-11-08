# Decision log

## 22-10-2026

- All NaNs seem related to fund inception; there are no empty cells after each fund started trading
- The number of trading days is more or less in line with yearly values (250-253)
- The number of days with consecutive 0% return, which could indicate data filling at the source, is not considerable
    x SHY is short term treasury ETF; after checking the years in which 0 returns are concentrated I believe it follows US rates change patterns and its low duration
    x UUP dates revolve around xmas
    x for HYG a crosscheck of ishares and nasdq data shows similar price behaviour (https://www.ishares.com/us/products/239565/ishares-iboxx-high-yield-corporate-bond-etf https://www.nasdaq.com/market-activity/etf/hyg/historical?page=1&rows_per_page=10&timeline=y1)

## 23-10-2026
- Presence of extreme returns
    x UUP displayed a strange 2 days behaviour which was cross-checked with DXY
- SHY, IEF, AGG and TIP show 10bp of trading costs based on volume estimates
    x In this case I will over-write the rule and for simplicity reduce the cost to 2bps. I will reflect this in pre_registration

## 28-10-2026
- After seeing the results, I realised that between equal weights and the more complex models described in de Prado's book, there might be added value in risk-weighting; I want to make sure to isolate this in diagnostics
- In terms of the 60/40 vs Equal Allocation results, they make sense in theory. US Equities, large caps in particular, has done remarkably better than many other asset classes or regional exposures; the performance gap is also very much regime-dependent rather than constant with an inflection point in late 2013 early 2014

## 30-10-2026
- Denoising: constant residual eigenvalue method. Targeted shrinkage not tested: it adds a tuning parameter, and with it extra trials
- Test allocator: long-only minimum variance, raw vs denoised covariance, as the allocator most sensitive to covariance noise so any effect of denoising shows most clearly
- No detoning for minimum variance (market risk must stay in the covariance); detoning kept for clustering in NCO
- All decided before any denoising results

## 02-11-2026
- Fitted σ² retained as declared. Diagnostic: across every year-end window 2005–2021 (11–21 ETFs) the fit sits at the upper bound σ² = 1.0, so the fitted ceiling equals
  the unfitted one for this universe.

## 03-11-2026
- HRP implemented as published (López de Prado 2016; AFML ch. 16): raw sample correlation, correlation distance, single linkage, recursive bisection by count. The 20% cap is applied by the engine
- Decided before any HRP backtest

 ## 04-11-2026
- ONC: base version (MLAM ch. 4), k-means for 2–10 clusters, 10 random starts each, best grouping chosen by mean/std of silhouette scores, run on the denoised and detoned correlation
    x The book's refinement step (re-clustering low-quality clusters) is not implemented
    x On the window ending 31-12-2021 it finds 2 clusters (rate-sensitive vs growth/commodity), consistent with 3 signal factors minus the market factor
- NCO: denoised covariance for the optimisation, ONC clusters for the grouping, long-only minimum variance within and across clusters
    x No cap inside NCO: the engine applies the 20% cap to the final weights
    x The book's version allows short positions; long-only is used here, as pre-registered
- All decided before any NCO backtest
- After NCO sanity checking: pro-rata capping inflates near-zero weights when NCO is concentrated (EMLC 0.5% → 20%), so the cap is applied by re-running NCO on the remaining assets instead

## 07-11-2026
- In the DSR calculation, N counts all 7 logged trials (benchmarks and diagnostic included). The book’s effective number of independent trials, which would be lower given the candidates’ high correlations, is not used: the plain count is simpler and stricter. Written before the run
- Holdout selection: NCO is taken to the holdout, alone
    x The pre-registration did not define "best-supported allocator", so the rule is set now, after the research results and before any holdout data is used
    x Rule: among candidates whose Sharpe ratios are statistically indistinguishable, take the one with the lowest maximum drawdown
    x Candidates' Sharpes (0.47–0.53) differ by less than one standard error (0.25; 0.11 for the difference, given correlations of 0.88–0.91). PSR vs equal weight 0.41–0.56
    x NCO has the lowest drawdown of all strategies (−9.7% vs −11.7% HRP, −14.0% min_var)
- Caveats
    x Maximum drawdown is a single-episode estimate, noisier than the Sharpe
    x NCO's turnover (8.3%) is above min_var's (6.8%)
    x NCO's research DSR is 0.93, below the pre-registered 0.95; HRP (0.96) would have been chosen on Sharpe or DSR alone
- Only NCO is run on the holdout: the period is too short (under 5 years) to separate two candidates correlated at 0.91, and a single pre-committed test keeps the holdout clean
- Success criteria unchanged: net Sharpe above equal weight and 60/40, and DSR ≥ 0.95

## 07-11-2026 (after the holdout run)
- Holdout run once, as planned; no changes made after seeing the results
- Hypothesis rejected: NCO met neither success criterion (Sharpe −0.63, DSR …)
- Live paper portfolio: NCO is still run live, labelled as not having passed the holdout, alongside equal weight and 60/40 as benchmarks
    x The purpose of the live stage is the data and automation infrastructure (daily data pipeline, monthly rebalancing, results store, dashboard) and an honest ongoing record, not a claim that NCO outperforms
    x No further strategies are tested on the holdout period; any new strategy would need a new pre-registration and a new holdout

## 08-11-2026
- Live rules written, see pre_registration.md → Amendments → Live paper portfolio
    x Start moved to the next month-end after deployment (1 December), not 1 October: October data was already seen in the holdout run
    x Prices stored as fetched, because Yahoo revises adjusted history after dividends