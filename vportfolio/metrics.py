# Metrics Estimators
    # calculate performance metrics for a given portfolio allocation and returns data
    
# Imports
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, norm

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
    
# calculate the PSR on given inputs
# will return a probability between 0 and 1

def psr(nav, rf_daily, sr_threshold = 0):
    # calculate daily returns
    daily_returns = nav.pct_change(fill_method=None).dropna()
    # allign the indices of returns and risk free rate, and fill NAs with 0s
    rf_alligned = rf_daily.reindex(daily_returns.index).fillna(0)
    daily_excess_returns = daily_returns - rf_alligned
    # caluclate daily Sharpe
    d_sharpe = daily_excess_returns.mean() / daily_excess_returns.std()
    # define additional variables (ingredients of uncertainty)
    T = len(daily_excess_returns)
    skew_ = skew(daily_excess_returns)
    kurt_ = kurtosis(daily_excess_returns, fisher=False) 
    # convert annual treshold to daily
    d_tr = sr_threshold / np.sqrt(252)
    # PSR formula
    z = (d_sharpe - d_tr)*np.sqrt(T-1)/np.sqrt(1-skew_*d_sharpe+(kurt_-1)/4*d_sharpe**2)
    # turn z to a probability between 0 and 1
    return norm.cdf(z)