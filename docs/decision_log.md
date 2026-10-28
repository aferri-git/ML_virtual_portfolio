# Decision log

# 22-10-2026

- All NaNs seem related to fund inception; there are no empty cells after each fund started trading
- The number of trading days is more or less in line with yearly values (250-253)
- The number of days with consecutive 0% return, which could indicate data filling at the source, is not considerable
    x SHY is short term treasury ETF; after checking the years in which 0 returns are concentrated I believe it follows US rates change patterns and its low duration
    x UUP dates revolve around xmas
    x for HYG a crosscheck of ishares and nasdq data shows similar price behaviour (https://www.ishares.com/us/products/239565/ishares-iboxx-high-yield-corporate-bond-etf https://www.nasdaq.com/market-activity/etf/hyg/historical?page=1&rows_per_page=10&timeline=y1)

# 23-10-2026
- Presence of extreme returns
    x UUP displayed a strange 2 days behaviour which was cross-checked with DXY
- SHY, IEF, AGG and TIP show 10bp of trading costs based on volume estimates
    x In this case I will over-write the rule and for simplicity reduce the cost to 2bps. I will reflect this in pre_registration

# 28-10-2026
- After seeing the results, I realised that between equal weights and the more complex models described in de Prado's book, there might be added value in risk-weighting; I want to make sure to isolate this in diagnostics
- In terms of the 60/40 vs Equal Allocation results, they make sense in theory. US Equities, large caps in particular, has done remarkably better than many other asset classes or regional exposures; the performance gap is also very much regime-dependent rather than constant with an inflection point in late 2013 early 2014