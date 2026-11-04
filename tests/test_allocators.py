# Test functions for allocators.py
    # All allocators need to yield a total % allocation of 100$

# Imports
import pandas as pd
import numpy as np
from vportfolio.allocators import sixty_forty, equal_weights, risk_weighted, min_var, min_var_raw, min_var_denoised, hrp, nco_weights, nco_capped, nco    
from vportfolio.clustering import onc
from vportfolio.denoising import denoise_corr, detone_corr

# 60/40 allocator
def test_sixty_forty():
    weights = sixty_forty(None)
    expected = pd.Series({"SPY": 0.6, "AGG": 0.4})
    # 1. always true: weights sum to 1
    assert weights.sum() == 1
    # 2. known answer weights are 60 and 40
    assert weights.equals(expected)
    # 3. inputs matter: by design they don't, so the window is ignored
    rng = np.random.default_rng(0)
    assert sixty_forty(pd.DataFrame(rng.normal(size=(10, 3)))).equals(weights)

# Equal weight allocator
def test_equal_weights():
    rng = np.random.default_rng(0) # fix the seed to ensure reproducibility
    window  = pd.DataFrame(rng.random((2,3)), columns = ["A", "B", "C"])
    weights = equal_weights(window)
    # 1. always true: weights sum to 1 and are never negative
    assert np.isclose(weights.sum(), 1)
    assert (weights >= 0).all()
    # 2. known answer: 3 ETFs → 1/3 each
    assert np.isclose(weights, 1/3).all()
    # 3. inputs matter: the number of ETFs sets the weight (4 ETFs → 1/4 each)
    window_four = pd.DataFrame(rng.random((2, 4)), columns=["A", "B", "C", "D"])
    assert np.isclose(equal_weights(window_four), 1/4).all()

# Risk weighted allocator
def test_risk_weighted():
    window = pd.DataFrame({"A": [1, -1], "B": [2, -2], "C": [4, -4]}) # volatility ratio 1 : 2 : 4
    weights = risk_weighted(window)
    # 1. always true: weights sum to 1 and are never negative
    assert np.isclose(weights.sum(), 1)
    assert (weights >= 0).all()
    # 2. known answer: inverse vols 1, 1/2, 1/4 (sum 7/4) → 4/7, 2/7, 1/7
    expected = pd.Series({"A": 4/7, "B": 2/7, "C": 1/7})
    assert np.isclose(weights, expected, atol=1e-12).all()
    # 3. inputs matter: doubling C's volatility lowers its weight
    window_c_up = window.copy()
    window_c_up["C"] = window_c_up["C"] * 2
    assert risk_weighted(window_c_up)["C"] < weights["C"]

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

# Minimum variance wrappers: each passes its own setting through
def test_min_var_wrappers():
    rng = np.random.default_rng(0)
    window = pd.DataFrame(rng.normal(size=(504, 10)) * np.linspace(0.002, 0.02, 10))
    assert min_var_raw(window).equals(min_var(window, denoise=False))
    assert min_var_denoised(window).equals(min_var(window, denoise=True))
    
# hrp
def test_hrp():
    rng = np.random.default_rng(0)
    window_two = pd.DataFrame(rng.normal(size=(10000, 2)) * [0.01, 0.02], columns=["A", "B"])
    window_many = pd.DataFrame(rng.normal(size=(504, 10)) * np.linspace(0.002, 0.02, 10),
                            columns=list("ABCDEFGHIJ"))
    # 1. always true: weights sum to 1, all positive, in the window's column order
    w = hrp(window_many)
    assert np.isclose(w.sum(), 1)
    assert (w > 0).all()
    assert list(w.index) == list(window_many.columns)
    # 2. known answer: two uncorrelated assets, vols 1% and 2% → one split, α = 1 − 1/(1+4) → 0.8 / 0.2
    w = hrp(window_two)
    assert np.isclose(w["A"], 0.8, atol=0.02)
    assert np.isclose(w["B"], 0.2, atol=0.02)
    # 3. inputs matter: tripling A's volatility lowers its weight
    window_up = window_many.copy()
    window_up["A"] = window_up["A"] * 3
    assert hrp(window_up)["A"] < hrp(window_many)["A"]
    
# NCO_weights
def test_nco_weights():
    assets = ["A", "B", "C", "D"] # 4 fake assets
    cov_test = pd.DataFrame(np.diag([1, 4, 1, 4]), index=assets, columns=assets) # fake cov matrix
    labels_test = pd.Series([1, 1, 0, 0], index=assets) # 2 clusters
    w = nco_weights(cov_test, labels_test)
    # 1. always true: weights sum to 1, all positive
    assert np.isclose(w.sum(), 1)
    assert (w > 0).all()
    assert list(w.index) == assets
    # 2. known answer: 
    # within cluster weights = 1/var, so A = 1 and B = 0.25. Normalized it becomes 0.8 and 0.2 (same for C and D)
    # identical clusters get 0.5 each
    inv_var = 1 / np.diag(cov_test)
    expected = inv_var / inv_var.sum()
    assert np.allclose(w, expected)
    # 3. inputs matter: raise A's variance to 2
    cov_test2 = cov_test.copy()
    cov_test2.loc["A", "A"] = 2
    w2 = nco_weights(cov_test2, labels_test)
    # diagonal cov -> NCO collapses to inverse variance
    inv_var = 1 / np.diag(cov_test2)
    expected = inv_var / inv_var.sum()
    assert np.allclose(w2, expected)
    assert w2["A"] < w["A"]   # riskier A gets less weight

# NCO_cpped 
def test_nco_capped():
    assets = ["A", "B", "C", "D", "E"]
    variances = np.array([1, 2, 8, 8, 8])
    cov_cap = pd.DataFrame(np.diag(variances), index=assets, columns=assets)
    labels_cap = pd.Series([0, 0, 1, 1, 1], index=assets)
    cap = 0.3
    w = nco_capped(cov_cap, labels_cap, cap)
    # 1. always true
    assert np.isclose(w.sum(), 1)
    assert (w >= 0).all() and (w <= cap + 1e-12).all()
    # 2. known answer: locked names at the cap, free names ∝ 1/variance
    locked = np.isclose(w, cap)
    assert list(w.index[locked]) == ["A", "B"]          # two rounds of locking
    inv_var = pd.Series(1 / variances, index=assets)[~locked]
    assert np.allclose(w[~locked], inv_var / inv_var.sum() * (1 - cap * locked.sum()))
    # 3. cap not binding -> identical to plain nco_weights
    assert np.allclose(nco_capped(cov_cap, labels_cap, cap=1.0), nco_weights(cov_cap, labels_cap))
    # 4. infeasible cap raises
    try:
        nco_capped(cov_cap, labels_cap, cap=0.1)
        raise AssertionError("should have raised")
    except ValueError:
        pass

#NCO
def test_nco():
    # build 3 identical blocks of 4 assets: shared factor + own noise
    rng = np.random.default_rng(0)
    n_days, blocks, per_block = 500, 3, 4
    factors = rng.normal(0, 0.01, (n_days, blocks))
    returns = np.hstack([factors[:, [b]] + rng.normal(0, 0.005, (n_days, per_block))
                        for b in range(blocks)])
    n_assets = blocks * per_block
    window = pd.DataFrame(returns, columns=[f"X{i}" for i in range(n_assets)])
    w = nco(window, cap=0.2)
    # 1. always true: same order, no gaps, sums to 1, within bounds
    assert list(w.index) == list(window.columns)
    assert not w.isna().any()
    assert np.isclose(w.sum(), 1)
    assert (w >= 0).all() and (w <= 0.2 + 1e-12).all()
    # 2. known answer
    true_labels = pd.Series(np.repeat(range(blocks), per_block), index=window.columns)
    t = window.shape[0]
    labels = onc(detone_corr(denoise_corr(window.corr(), t)))
    # 2a. ONC recovers the blocks (label numbers may differ, groupings must match)
    assert pd.crosstab(labels, true_labels).gt(0).sum(axis=1).eq(1).all()
    assert labels.nunique() == blocks
    # 2b. each block gets ~1/3 (tight: noise averages out within a block)
    assert np.allclose(w.groupby(true_labels).sum(), 1 / blocks, atol=0.03)
    # 2c. each asset near 1/12 (loose: min-var amplifies noise in correlated assets)
    assert np.allclose(w, 1 / n_assets, atol=0.05)
    # 3. scale invariance: doubling returns leaves corr unchanged, cov x4 -> same weights
    assert np.allclose(nco(window * 2, cap=0.2), w, atol=1e-6)
    # 4. deterministic: same input, same output. in case NCO uses k-means with a random seed
    assert np.allclose(nco(window, cap=0.2), w)
