## Test functions for engine.py

# Imports
import pandas as pd
import numpy as np
from vportfolio.denoising import denoise_with_ceiling, fit_sigma2, kde, mp_pdf, denoise_corr

# small helper function to calculate the condition number
def cond(m):
    ev = np.linalg.eigvalsh(m)
    return ev.max() / ev.min()

# kde
def test_kde(): 
    # 1. always true: a proper density (area 1, never negative)
    x_test1 = np.linspace(-2, 7, 5000) # wide enough to hold every bell
    values1 = kde(np.array([1.0, 2.0, 4.0]), x_test1)
    assert np.isclose(np.trapezoid(values1, x_test1), 1, atol=0.01) # area is 1
    assert (values1 >= 0).all() # never negative
    # 2. known answer: one eigenvalue at 2 → one bell, peak at 2, height 1/(0.25·√2π) ≈ 1.596
    x_test2 = np.linspace(-2, 6, 5000)
    values2 = kde(np.array([2.0]), x_test2)
    assert np.isclose(x_test2[np.argmax(values2)], 2, atol=0.01) # peak at 2
    assert np.isclose(values2.max(), 1 / (0.25 * np.sqrt(2 * np.pi)), atol=0.01) # height for width 0.25
    # 3. inputs matter: the peak follows the eigenvalue, and its height follows the bandwidth
    values3 = kde(np.array([3.0]), x_test2) # move the eigenvalue
    assert np.isclose(x_test2[np.argmax(values3)], 3, atol=0.01) # peak moves to 3
    values4 = kde(np.array([2.0]), x_test2, bandwidth=0.5) # double the bandwidth
    assert np.isclose(values4.max(), values2.max() / 2, atol=0.01) # peak halves
    
# mp_pdf
def test_mp_pdf():
    # 1. always true: a proper density (area 1, no NaNs, never negative)
    x= np.linspace(0.001, 3, 10000) 
    values1 = mp_pdf(1, 24, x)
    assert np.isclose(np.trapezoid(values1, x), 1, atol=0.01) # area is 1
    assert not np.isnan(values1).any() # no NaNs (clip works)
    assert (values1 >= 0).all() # never negative
    # 2. known answer: σ² = 1, q = 24 → curve only between about 0.63 and 1.45
    assert (mp_pdf(1, 24, np.array([0.3, 2.0])) == 0).all() # zero outside the edges
    assert (mp_pdf(1, 24, np.array([1.0])) > 0).all() # positive inside
    # 3. inputs matter: halving σ² moves the curve to about 0.31–0.72
    assert (mp_pdf(0.5, 24, np.array([1.0])) == 0).all() # 1.0 now outside
    assert (mp_pdf(0.5, 24, np.array([0.5])) > 0).all() # 0.5 now inside
    
# fit_sigma2
def test_fit_sigma2():
    # 1. always true: output between the line i search for 0.01 to 1000
    rng = np.random.default_rng(0) # set a seed
    noise = rng.normal(size=(1000, 20)) # 1000 days, 20 unrelated ETFS: pure noise
    eig_noise = np.linalg.eigvalsh(pd.DataFrame(noise).corr())
    q_test = 1000 / 20 
    sigma2_noise = fit_sigma2(eig_noise, q_test)
    assert 0.01 <= sigma2_noise <= 1
    # 2. known answer: σ² = 1 for a pure noise case
    assert np.isclose(sigma2_noise, 1, atol=0.15)
    # 3. inputs matter: σ² < 1 if we add signals. also the biggest eigenvalue must be above ceiling
    market = noise + 2 * rng.normal(size=(1000, 1)) # same movement added to every column
    eig_market = np.linalg.eigvalsh(pd.DataFrame(market).corr())
    sigma2_market = fit_sigma2(eig_market, q_test)
    assert sigma2_market < 0.5 # far below the pure-noise σ² of 1

    signal = noise.copy()
    signal[:, :5] += rng.normal(size=(1000, 1)) # columns 0–4 share one movement
    eig_signal = np.linalg.eigvalsh(pd.DataFrame(signal).corr())
    sigma2_signal = fit_sigma2(eig_signal, q_test)
    ceiling = sigma2_signal * (1 + 1 / np.sqrt(q_test)) ** 2
    assert eig_signal.max() > ceiling # the shared movement stands out
    
# denoise_with_ceiling
def test_denoise_with_ceiling():
    rng = np.random.default_rng(0) # fixed seed
    random_corr = pd.DataFrame(rng.normal(size=(100, 5))).corr() # random matrix
    # 1. always true:
    result = denoise_with_ceiling(random_corr, 1.5)
    assert np.allclose(np.diag(result), 1) # diagonal is always 1
    assert np.allclose(result, result.T) # it's symmetrical
    assert (np.linalg.eigvalsh(result) > 0).all() # all eigenvalues are positive
    # 2. known answer: 
    identity_corr = pd.DataFrame(np.eye(5)) # using an identity matrix
    result = denoise_with_ceiling(identity_corr, 1.01) # ceiling above all eigenvals (1)
    assert (np.linalg.eigvalsh(result)).mean() ==1 # all eigenvals are noise so avg is 1
    result = denoise_with_ceiling(identity_corr, 0) # ceiling below all eigenvals
    assert np.allclose(result, np.eye(5))
    # 3. inputs matter: a higher ceiling will flatten more igenvalues --> condition number gets smaller
    assert cond(denoise_with_ceiling(random_corr, 1.5)) <= cond(denoise_with_ceiling(random_corr, 0.5))

# denoise_corr
def test_denoise_corr():
    # 1. always true: 
    rng = np.random.default_rng(0)
    random_corr = pd.DataFrame(rng.normal(size=(500, 6))).corr()
    result = denoise_corr(random_corr, t=500)
    assert np.allclose(np.diag(result), 1) # diagonal is 1
    assert np.allclose(result, result.T) # symmetric
    assert (np.linalg.eigvalsh(result) > 0).all() # positive eigenvalues
    assert result.index.equals(random_corr.index) # same labels

    # 2. known answer:
    m = np.eye(6) # 6 assets, no correlation yet (1s on the diagonal, 0s elsewhere)
    m[:3, :3] = 0.8 # block 1: assets 0–2 move together (correlation 0.8)
    m[3:, 3:] = 0.8 # block 2: assets 3–5 move together; the two blocks stay unrelated (0)
    np.fill_diagonal(m, 1) # put the diagonal back to 1: each asset fully correlates with itself
    block_corr = pd.DataFrame(m) # as a DataFrame, like a real correlation matrix
    result = denoise_corr(block_corr, t=500)
    assert np.allclose(result.iloc[:3, :3].values[np.triu_indices(3, 1)], 0.8, atol=0.05) # inside block 1 the correlations are still about 0.8
    assert np.allclose(result.iloc[:3, 3:], 0, atol=0.05) # between the two blocks the correlations are still about 0
    # 3. inputs matter: denoising makes the matrix better conditioned
    def cond(m):
        ev = np.linalg.eigvalsh(m)
        return ev.max() / ev.min()
    assert cond(denoise_corr(random_corr, t=500)) <= cond(random_corr)