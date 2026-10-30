# Test functions for allocators.py
    # All allocators need to yield a total % allocation of 100$
    
## 3 TESTS ##

# Imports
import pandas as pd
import numpy as np
from vportfolio.allocators import sixty_forty, equal_weights, risk_weighted

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
