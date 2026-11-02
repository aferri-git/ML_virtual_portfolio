# Test functions for allocators.py
    # All allocators need to yield a total % allocation of 100$

# Imports
import pandas as pd
import numpy as np
from vportfolio.allocators import sixty_forty, equal_weights, risk_weighted, min_var, min_var_raw, min_var_denoised

# 60/40 allocator
def test_sixty_forty():
    target = 1
    weights = sixty_forty(_=None) # makes sure to ignore any argument given
    expected = pd.Series({"SPY": 0.6, "AGG": 0.4})
    assert weights.equals(expected)
    assert np.isclose(weights.sum(), 1)

# Equal weight allocator
def test_equal_weights():
    rng = np.random.default_rng(0) # fix the seed to ensure reproducibility
    window  = pd.DataFrame(rng.random((2,3)), columns = ["A", "B", "C"])
    weights = equal_weights(window)
    assert np.isclose(weights, 1/3).all()
    assert np.isclose(weights.sum(), 1)

# Risk weighted allocator
def test_risk_weighted():
    window =pd.DataFrame({"A": [1, -1], "B": [2, -2], "C": [4, -4]}) # fix the values to ensure volatility ratio of 1:2:4
    weights= risk_weighted(window)
    # vols are 1.414, 2.828, 5.657 -> ratio 1 : 2 : 4
    # inverse vols 1, 1/2, 1/4 -> sum 7/4
    # weights = (1, 1/2, 1/4) / (7/4) = 4/7, 2/7, 1/7
    expected = pd.Series({"A": 4/7, "B": 2/7, "C": 1/7})
    assert np.isclose(weights, expected, atol=1e-12).all() # check that the weights are correct
    assert np.isclose(weights.sum(), 1)

# Minimum variance allocator
def test_min_var():
    rng = np.random.default_rng(0)
    window_two = pd.DataFrame(rng.normal(size=(10000, 2)) * [0.01, 0.02], columns=["A", "B"])
    window_many = pd.DataFrame(rng.normal(size=(504, 10)) * np.linspace(0.002, 0.02, 10))
    # 1. always true: valid weights, and no other valid portfolio has lower variance
    w = min_var(window_many, cap=0.2)
    assert np.isclose(w.sum(), 1)
    assert (w >= -1e-8).all() and (w <= 0.2 + 1e-8).all()
    cov_m = window_many.cov().values
    best = w.values @ cov_m @ w.values
    compared = 0
    for _ in range(500):
        r = rng.dirichlet(np.ones(10) * 5) # random weights summing to 1, near even
        if (r <= 0.2).all(): # only portfolios that respect the cap
            assert best <= r @ cov_m @ r + 1e-12 # none beats the optimiser
            compared += 1
    assert compared > 100 # enough portfolios were actually checked
    # 2. known answer: two uncorrelated assets, vols 1% and 2%, no cap → 0.8 / 0.2
    w = min_var(window_two, cap=1)
    assert np.isclose(w["A"], 0.8, atol=0.02)
    assert np.isclose(w["B"], 0.2, atol=0.02)
    # 3. inputs matter: the cap binds, and denoising changes the weights
    w = min_var(window_two, cap=0.6)
    assert np.isclose(w["A"], 0.6, atol=0.01) # low-vol asset held at the cap
    assert np.isclose(w["B"], 0.4, atol=0.01)
    assert not np.allclose(min_var(window_many, denoise=True), min_var(window_many, denoise=False))

# And its wrappers
def test_min_var_wrappers():
    rng = np.random.default_rng(0)
    window = pd.DataFrame(rng.normal(size=(504, 10)) * np.linspace(0.002, 0.02, 10))
    assert min_var_raw(window).equals(min_var(window, denoise=False))
    assert min_var_denoised(window).equals(min_var(window, denoise=True))