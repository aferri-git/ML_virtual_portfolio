## Test functions for engine.py
    # Tests requirements are within the assert statements of each function

# Imports
import pandas as pd
import numpy as np
import pytest
from vportfolio.metrics import performance_calculator

# constant growth case with 0 risk free rate
@pytest.mark.filterwarnings("ignore:divide by zero") # daily returns are constant, so std = 0, which causes a divide by zero warning in sharpe ratio calculation
def test_constant_growth():
    start = pd.Timestamp('2020-01-01')
    dates = pd.bdate_range(start=start, end=start+pd.Timedelta(days=3))
    rf_daily = pd.Series(0, index=dates)
    nav = pd.Series(1.01 ** np.arange(3), index=dates)
    results = performance_calculator(nav, rf_daily)
    assert np.isclose(results['ann_vol'], 0, atol=1e-5)
    assert np.isclose(results['max_drawdown'], 0, atol=1e-5)
    assert np.isclose(results['ann_return'], (1.01 ** 252) -1, atol=1e-5)
    
# known drawdown case of -50%
def test_known_drawdown():
    start = pd.Timestamp('2020-01-01')
    dates = pd.bdate_range(start=start, end=start+pd.Timedelta(days=3))
    rf_daily = pd.Series(0, index=dates)
    nav = pd.Series([1, 2.0, 1.0], index=dates)
    results = performance_calculator(nav, rf_daily)
    assert np.isclose(results['max_drawdown'], -0.5, atol=1e-5)
    
# sharpe testing based on two different risk free rates with constant growth
@pytest.mark.filterwarnings("ignore:divide by zero")
def test_sharpe():
    start = pd.Timestamp('2020-01-01')
    dates = pd.bdate_range(start=start, end=start+pd.Timedelta(days=3))
    rf_daily = pd.Series(0, index=dates)
    nav = pd.Series(1.01 ** np.arange(3), index=dates)
    pd.Series(1.01 ** np.arange(3), index=dates)
    rf_daily_up = pd.Series(0.03, index=dates)
    results1 = performance_calculator(nav, rf_daily)
    results2 = performance_calculator(nav, rf_daily_up)
    
    assert results1['sharpe'] > 0 # 0 risk free rate
    assert results2['sharpe'] < 0 # risk free rate higher than NAV returns
    
    
    