# Decision log

# 22-10-2026

- All NaNs seem related to fund inception; there are no empty cells after each fund started trading
- The number of trading days is more or less in line with yearly values (250-253)
- The number of days with consecutive 0% return, which could indicate data filling at the source, is not considerable
    x SHY is short term treasury ETF; after checking the years in which 0 returns are concentrated I believe it follows US rates change patterns and its low duration
    x UUP dates revolve around xmas
    x for HYG a crosscheck of ishares and nasdq data shows similar price behaviour (https://www.ishares.com/us/products/239565/ishares-iboxx-high-yield-corporate-bond-etf https://www.nasdaq.com/market-activity/etf/hyg/historical?page=1&rows_per_page=10&timeline=y1)