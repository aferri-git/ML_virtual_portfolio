# Probabilistic and Deflated Sharpe Ratios

Based on my notes from López de Prado, *Machine Learning for Asset Managers*, ch. 8, and Bailey & López de Prado (2012, 2014).

## Problem
A Sharpe ratio is an estimate, and it can be far from the true value even over long periods. Its error grows when the track record is short and when returns are not norma (negative skew, fat tails)

Example from testing: the same true Sharpe (0.63), drawn with five different random seeds, gave 10-year sample Sharpes of 0.15, 0.46, −0.01, 1.03 and 0.65

A second problem appears when several strategies are tested: the best of several noisy Sharpes looks better than it really is (see DSR below)

## Probabilistic Sharpe Ratio (PSR)
The PSR turns a Sharpe ratio into a probability: the probability that the strategy's true Sharpe is above a chosen threshold SR*

PSR(SR*) = Φ( (SR − SR*) · √(T − 1) / √(1 − γ₃ · SR + (γ₄ − 1)/4 · SR²) )

- SR: the strategy's Sharpe, in daily units (mean / std of daily excess returns)
- SR*: the threshold, converted from annual to daily (÷ √252)
- T: the number of daily returns; a longer track record increases the PSR
- γ₃: skewness; negative skew (occasional large losses) lowers the PSR
- γ₄: kurtosis (normal = 3); fat tails lower the PSR
- Φ: the normal cumulative distribution, which turns the result into a probability

For normal returns the denominator reduces to √(1 + SR²/2), which gives the standard error of about 0.25 over 17 years used so far

The threshold sets the question:
- SR* = 0: is the Sharpe positive at all?
- SR* = a benchmark's Sharpe: does the strategy truly beat that benchmark?

## Findings: PSR of every strategy (06-11-2026)

|                  |   vs_0 |   vs_6040 |   vs_ew |
|:-----------------|-------:|----------:|--------:|
| sixty_forty      |   1    |      0.5  |    0.74 |
| equal_weights    |   0.98 |      0.26 |    0.5  |
| risk_weighted    |   0.99 |      0.4  |    0.65 |
| hrp              |   0.99 |      0.31 |    0.56 |
| nco              |   0.97 |      0.23 |    0.46 |
| min_var_raw      |   0.96 |      0.19 |    0.41 |
| min_var_denoised |   0.97 |      0.23 |    0.46 |

- Sanity check: each benchmark against its own Sharpe gives 0.50, as expected
- Every strategy's Sharpe is very likely above 0 (0.96–1.00): all earn a real excess return over T-bills
- Against 60/40, the probability that any candidate truly beats it is only 0.19–0.31
- Against equal weight, the candidates are indistinguishable from a coin flip (0.41–0.56; HRP highest)
- Even 60/40 beats equal weight with only 0.74 probability, far from the 0.95 that would count as convincing evidence

**Conclusion:** every strategy’s Sharpe is very likely above 0 (0.96–1.00). Against equal weight’s Sharpe, the candidates are indistinguishable from a coin flip (0.41–0.56; HRP highest). Against 60/40’s Sharpe, the probability that any candidate truly beats it is only 0.19–0.31. The candidates offer lower risk, not a demonstrably better risk-adjusted return

## Deflated Sharpe Ratio (DSR)

N counts all 7 logged trials (benchmarks and diagnostic included). The book’s effective number of independent trials, which would be lower given the candidates’ high correlations, is not used: the plain count is simpler and stricter.


|                  |   psr_vs_0 |   dsr |
|:-----------------|-----------:|------:|
| sixty_forty      |       1    |  0.99 |
| equal_weights    |       0.98 |  0.94 |
| risk_weighted    |       0.99 |  0.98 |
| hrp              |       0.99 |  0.96 |
| nco              |       0.97 |  0.93 |
| min_var_raw      |       0.96 |  0.91 |
| min_var_denoised |       0.97 |  0.93 |

The luck-only bar is 0.104: by chance alone, the best of the 7 trials would be expected to reach an annual Sharpe of about 0.10.

The DSR is so low because it is based on how spread the sharpes are. In our trials, every strategy has a relatively comparable Sharpe ratio so the correlation makes the estimate rough

**Conclusions**
- Every strategy clears the luck bar with a solid margin
- HRP is the best candidate, with a DSR of 0.96. That is just above the pre-registered treshold of 0.95. The candidates earn a genuine positive risk-adjusted return, but not a demonstrably better one than equal weight or 60/40.
- NCO and min_var allocators are close to the bar but below it (0.91 and 0.93)