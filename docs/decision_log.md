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