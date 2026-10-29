# Metrics Estimators
    # calculate performance metrics for a given portfolio allocation and returns data
    
# Imports
import pandas as pd
import numpy as np

# performance_calculator
# returns a series with annualized returns and volatility, sharpe ratio and max drawdown
def performance_calculator (nav, rf_daily):
    # calculate daily returns
    daily_returns = nav.pct_change().dropna()
    # annualised return
    total_growth = (nav.iloc[-1]/nav.iloc[0])
    n_years = len(daily_returns) / 252
    annualised_returns = total_growth**(1/n_years)-1
    # annualised volatility
    annualised_volatility = daily_returns.std()*np.sqrt(252)
    # Sharpe Ratio
    rf_alligned = rf_daily.reindex(daily_returns.index).fillna(0)
    excess_returns = daily_returns - rf_alligned
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    # max drawdown
    drawdown = nav / nav.cummax() - 1
    max_drawdown = drawdown.min()
    return pd.Series({
        "ann_return":   annualised_returns,
        "ann_vol":      annualised_volatility,
        "sharpe":       sharpe,
        "max_drawdown": max_drawdown,
    })