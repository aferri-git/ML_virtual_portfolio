# Allocators L

All strategies are backtested over the research period (2005–2021), net of costs, with monthly rebalancing, as set out in the pre-registration.

## 60/40 (SPY / AGG)
**Role:** benchmark (external reference) 60% S&P 500 (SPY), 40% US aggregate bonds (AGG), rebalanced monthly. 
The traditional balanced portfolio that any allocator should be compared with

## Equal weight
**Role:** benchmark (same universe)
Equal weight across all eligible ETFs (expanding universe, 20% cap). Isolates the value added by any weighting method beyond simply holding the universe

## Risk-weighted (inverse volatility)
**Role:** diagnostic. Added on 2026-10-28, after the benchmark results (see amendment)
Weights proportional to 1 / volatility over the 504-day window, capped at 20%. Shows how much can be gained from volatility alone, ignoring correlations

## Minimum variance: raw and denoised
**Role:** candidates. Added on 2026-11-02
Long-only weights with the lowest portfolio variance, capped at 20%. Uses the full covariance matrix, so it depends on the estimated correlations. Run twice, identical
except for the covariance: sample (min_var_raw) and denoised (min_var_denoised). Method and denoising results: see [methods/denoising.md](methods/denoising.md)



## Resuls

### Minimum variance: 2-11-2026
| name             | role       |   ann_return |   ann_vol |   sharpe |   max_drawdown |   avg_turnover |
|:-----------------|:-----------|-------------:|----------:|---------:|---------------:|---------------:|
| sixty_forty      | benchmark  |        0.082 |     0.114 |    0.65  |         -0.357 |          0.022 |
| equal_weights    | benchmark  |        0.057 |     0.1   |    0.492 |         -0.337 |          0.038 |
| risk_weighted    | diagnostic |        0.042 |     0.052 |    0.588 |         -0.194 |          0.035 |
| min_var_raw      | candidate  |        0.026 |     0.034 |    0.438 |         -0.141 |          0.07  |
| min_var_denoised | candidate  |        0.027 |     0.034 |    0.469 |         -0.14  |          0.068 |

![Minimum variance vs benchmarks](figures/min_var.png)
![Minimum variance vs benchmarks](figures/min_var_ratio.png)

**Findings**
- Minimum variance delivers the lowest volatility (3.4%) and drawdown (−14%) of all strategies, but the weakest risk-adjusted return (Sharpe 0.44–0.47)
- It holds up in crises (2008, 2020) and lags in long expansions: level with 60/40 until about 2013, then falls steadily behind
- Its turnover is the highest (about 7% per rebalance, twice equal weight's): the optimiser reshuffles between near-equivalent low-volatility assets
- Risk-weighted, which ignores correlations, beats it on Sharpe (0.59): using the correlations made the portfolio less volatile, but not more efficient
- Denoising moved results in the expected direction (lower turnover, higher Sharpe), but the effect is too small to be meaningful (see methods/denoising.md)

Sharpe differences of less than about 0.25 are within one standard error over 17 years.

## Future work (min_var_backtested.ipynb)
Each allocator performs best in a different environment: minimum variance in crises
(2008, 2020), equity-heavy allocations in long expansions. A natural extension is a
second model that identifies the current regime and chooses the allocator accordingly
(regime switching; related to meta-labelling in López de Prado, *Advances in Financial
Machine Learning*). It is outside the pre-registered scope: the regime would have to
be detected in real time, not in hindsight, and every regime–allocator combination
would count as an additional trial for the Deflated Sharpe Ratio

### Hierarchical Risk Parity (HRP) 3-11-20256
| name             | role       |   ann_return |   ann_vol |   sharpe |   max_drawdown |   avg_turnover |
|:-----------------|:-----------|-------------:|----------:|---------:|---------------:|---------------:|
| sixty_forty      | benchmark  |        0.082 |     0.114 |    0.65  |         -0.357 |          0.022 |
| equal_weights    | benchmark  |        0.057 |     0.1   |    0.492 |         -0.337 |          0.038 |
| risk_weighted    | diagnostic |        0.042 |     0.052 |    0.588 |         -0.194 |          0.035 |
| min_var_raw      | candidate  |        0.026 |     0.034 |    0.438 |         -0.141 |          0.07  |
| min_var_denoised | candidate  |        0.027 |     0.034 |    0.469 |         -0.14  |          0.068 |
| hrp              | candidate  |        0.031 |     0.038 |    0.532 |         -0.117 |          0.122 |
![Minimum variance vs benchmarks](figures/hrp.png)

**Findings**
The turnover of the hrp model is the highest so far. It seems that the structure of the model is working against itself. 
- The tree reorders every month. Single linkage is sensitive: a small change in one correlation can move an ETF to a different branch, which changes the leaf order.
- Bisection is by count, not by branch. When the order shifts, an ETF can land in a different half at the top split, and its weight is multiplied by a very different α. A tiny change in the data becomes a large jump in weight.
- The cap amplifies it: about 70% of the portfolio gets redistributed every month, in proportions that follow these unstable small weights.