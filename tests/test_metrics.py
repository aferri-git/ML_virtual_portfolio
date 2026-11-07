## Test functions for metrics.py

# Imports
import pandas as pd
import numpy as np
import pytest
from vportfolio.metrics import performance_calculator, psr, expected_max_sharpe, dsr

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
    

# PSR: Probability of Sharpe Ratio
def test_psr():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=2520) # 10 years of business days
    test_rf = pd.Series(0.0, index=dates) # risk-free rate of 0
    # a random NAV: daily returns drawn with mean 0.0004 and std 0.01 (the sample comes out weaker)
    returns = pd.Series(rng.normal(0.0004, 0.01, size=2520), index=dates)
    nav = (1 + returns).cumprod()
    # a "strong" NAV: draws rescaled so the sample has exactly mean 0.0004 and std 0.01 (daily Sharpe 0.04)
    draws = rng.normal(size=2520)
    strong = pd.Series((draws - draws.mean()) / draws.std() * 0.01 + 0.0004, index=dates)
    nav_strong = (1 + strong).cumprod()
    # 1. always true: the PSR is a probability, between 0 and 1
    test = psr(nav, test_rf)
    assert 0 <= test <= 1
    # 2. known answers
    # 2a. threshold equal to the strategy's own (annual) Sharpe → exactly 0.5
    nav_returns = nav.pct_change(fill_method=None).dropna() # the same returns psr uses
    own_sharpe = nav_returns.mean() / nav_returns.std() * np.sqrt(252)
    test2 = psr(nav, test_rf, own_sharpe)
    assert np.isclose(test2, 0.5, atol=1e-6)
    # 2b. a strong Sharpe over 10 years → very likely above 0
    assert psr(nav_strong, test_rf) > 0.95
    # 3. inputs matter
    # 3a. a higher threshold is harder to beat → lower PSR
    assert test2 < test
    # 3b. a shorter track record (100 days) with the same strength → lower PSR
    assert psr(nav_strong.iloc[:100], test_rf) < psr(nav_strong, test_rf)
    
# excected_max_sharpe and dsr
def test_dsr():
    # helper function to create fake sharpe ratios
    def fake_sharpes(n, mean, std, seed=0):
        draws = np.random.default_rng(seed).normal(size=n)
        return pd.Series((draws - draws.mean()) / draws.std() * std + mean) # exactly this mean and std
    # 10 years of business days
    rng = np.random.default_rng(0) # fix seed
    dates = pd.bdate_range("2020-01-01", periods=2520)
    test_rf = pd.Series(0.0, index=dates) # risk-free rate of 0
    # a strong NAV: exactly mean 0.0004 and std 0.01 per day (annual Sharpe = 0.63)
    draws = rng.normal(size=2520)
    strong = pd.Series((draws - draws.mean()) / draws.std() * 0.01 + 0.0004, index=dates)
    nav_strong = (1 + strong).cumprod()
    # 1. always true: the DSR is a probability between 0 and 1
    test = dsr(nav_strong, test_rf, fake_sharpes(7, 0.5, 0.2))
    assert 0 <= test <= 1
    # 2. known answer: all trials with the same Sharpe → SR0 = 0 → DSR equals the PSR against 0
    equal_sharpes = pd.Series([0.5] * 7)
    assert np.isclose(dsr(nav_strong, test_rf, equal_sharpes), psr(nav_strong, test_rf), atol=1e-9)
    # 3. inputs matter: more trials, or more spread among them, raise the bar → lower DSR
    assert dsr(nav_strong, test_rf, fake_sharpes(50, 0.5, 0.2)) < test   # 50 trials vs 7
    assert dsr(nav_strong, test_rf, fake_sharpes(7, 0.5, 0.4)) < test    # spread 0.4 vs 0.2