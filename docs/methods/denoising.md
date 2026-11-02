# Denoising the covariance matrix

Based off my notes from López de Prado, *Machine Learning for Asset Managers*, ch. 2.

## Problem
Covariance matrices are estimated from a limited number of observations relative to the
number of parameters (21 ETFs → 231 parameters from 504 days). Part of what is estimated
is noise. Even when the matrix is invertible, small eigenvalues make it ill-conditioned:
inverting it, as optimisers do, amplifies estimation erros

**Consequence:** unstable weights, misallocation to assets whose apparent diversification
is a sampling artefact, and higher turnover and transaction costs

## Method
### Selected: Constant residual eigenvalue
The Marčenko–Pastur distribution gives the range of eigenvalues expected from pure noise,
given q = observations / assets. Eigenvalues above its upper bound are signal; those below
are treated as noise. Denoising replaces the noise eigenvalues with their average and
rebuilds the correlation matrix, removing noise while keeping the signal eigenvalues
unchanged

### Not tested: Targeted shrinkage
Shrinks only the noise eigenvalues. Not tested because it adds a tuning parameter, and 
with it extra trials.

## Why not the alternatives
- **Threshold method** (Jolliffe, 2002): keeps a fixed number of components or share of
  variance, chosen without testing what is noise.
- **Shrinkage** (Ledoit–Wolf): pulls the whole matrix towards a target, reducing noise
  but also diluting the signal.

## Effect
Raising the smallest eigenvalues lowers the condition number (largest / smallest
eigenvalue), so the matrix is better conditioned and the optimiser is more stable.

## Matrix-level findings (30-10-2026)
- Window ending 2021-12-31 (504 days × 21 ETFs, q = 24): 3 eigenvalues above the
  Marčenko–Pastur ceiling (guess was 4: risk appetite, rates, oil, dollar)
- Condition number: 560 raw → 33 denoised
- The condition number is roughly the most that inverting the matrix can amplify a
  small estimation error. A lower value means allocators using the denoised matrix
  should produce more stable weights; the raw vs denoised minimum-variance backtest
  will test whether this holds in practice

## Result of backtest on minimum variance allocator (2-11-2026)
- Min Var allocator works: lowest volatility and drawdown compared to other allocators
- It does not however pay off in returns --> the Sharpe is the lowest of the five
- Denoising does help the turnover, with slightly lower value for the denoised version

## Why the denoising effect is relatively small (2-11-2026)
- We have a lot of data per each asset; noise is modest for 21 ETFs and 504 days. The ML book shows large gains when data is scarce
- The 20% cap already limits the optimiser. The cap stops the allocator from betting heavily on random correlations, which is what denoising should prevent --> part of the job is already done a priori