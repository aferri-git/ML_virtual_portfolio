## Test functions for metrics.py

# Imports
import pandas as pd
import numpy as np
import pytest
from vportfolio.metrics import performance_calculator

# helper: a NAV built from a list of daily returns, starting at 1, with a matching risk-free series
def make_nav(daily_returns, rf=0.0):
    dates = pd.bdate_range("2020-01-01", periods=len(daily_returns) + 1)
    nav = pd.Series(np.cumprod([1.0] + [1 + r for r in daily_returns]), index=dates)
    rf_daily = pd.Series(rf, index=dates)
    return nav, rf_daily

# 1. always true: on any NAV, all four metrics come back, vol is never negative, drawdown never positive
def test_metrics_always_true():
    rng = np.random.default_rng(0)
    nav, rf_daily = make_nav(rng.normal(0.0003, 0.01, size=500)) # one random NAV path
    results = performance_calculator(nav, rf_daily)
    assert set(results.index) == {"ann_return", "ann_vol", "sharpe", "max_drawdown"}
    assert results.notna().all()
    assert results["ann_vol"] >= 0
    assert results["max_drawdown"] <= 0

# 2. known answers
@pytest.mark.filterwarnings("ignore:divide by zero") # constant returns → std 0 → the Sharpe divides by zero
def test_metrics_constant_growth():
    nav, rf_daily = make_nav([0.01, 0.01]) # +1% every day
    results = performance_calculator(nav, rf_daily)
    assert np.isclose(results["ann_return"], 1.01 ** 252 - 1, atol=1e-5) # 1% a day, compounded over a year
    assert np.isclose(results["ann_vol"], 0, atol=1e-12) # identical returns → no volatility
    assert np.isclose(results["max_drawdown"], 0, atol=1e-12) # never falls

def test_metrics_known_drawdown():
    nav, rf_daily = make_nav([1.0, -0.5, 0.5]) # NAV 1 → 2 → 1 → 1.5
    results = performance_calculator(nav, rf_daily)
    assert np.isclose(results["max_drawdown"], -0.5, atol=1e-12) # peak 2, trough 1

def test_metrics_zero_sharpe():
    nav, rf_daily = make_nav([0.02, 0.0] * 50, rf=0.01) # returns average 1%, equal to the risk-free rate
    results = performance_calculator(nav, rf_daily)
    assert np.isclose(results["sharpe"], 0, atol=1e-9) # no excess return → Sharpe 0

# 3. inputs matter: the risk-free rate is subtracted, so a higher rate lowers the Sharpe
def test_metrics_risk_free_matters():
    returns = [0.02, 0.0] * 50 # average 1% a day
    sharpe_low = performance_calculator(*make_nav(returns, rf=0.0))["sharpe"]
    sharpe_high = performance_calculator(*make_nav(returns, rf=0.03))["sharpe"]
    assert sharpe_low > 0 # return above the risk-free rate
    assert sharpe_high < 0 # return below the risk-free rate
    
    
    