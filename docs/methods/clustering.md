# Clustering the ETF universe

Based on my notes from López de Prado, *Machine Learning for Asset Managers*, ch. 3–4,
and *Advances in Financial Machine Learning*, ch. 16.

## Problem
Optimisers treat the whole universe as one problem, so groups of similar ETFs (e.g.
SPY/QQQ, IEF/TLT) make the covariance matrix fragile. Cluster-based allocators first
group the ETFs that behave alike, then allocate between and within the groups.

## From correlation to distance
Clustering groups ETFs by how far apart they are. Correlation measures how alike two
ETFs are, so it is first converted to a distance (MLAM ch. 3):

d = √((1 − ρ) / 2)

ρ = +1 gives d = 0 (identical), ρ = 0 gives d ≈ 0.71 (unrelated), ρ = −1 gives d = 1
(opposite). Distance is never negative, zero only for identical series, symmetric, and
obeys the triangle inequality.

## Hierarchical tree (used by HRP)
Built from the bottom up: start with every ETF on its own, merge the two closest, measure
the distance from the new group to everything else (single linkage: the distance to its
closest member, as in the HRP paper), and repeat until everything sits in one group. HRP
uses the whole tree, uncut, on the raw correlation.

## Detoning (used by ONC)
Removes the market factor (the largest eigenvalue), which otherwise makes every pair look
related and blurs the groups. The correlation is denoised first and detoned second:
denoising needs the full matrix. The detoned matrix is singular, so it is used for
clustering only, never in an optimiser.

## Optimal number of clusters (ONC, used by NCO)
Base version of MLAM ch. 4: k-means for 2 to 10 clusters, 10 random starts each, on the
denoised and detoned correlation. Each grouping is scored by the mean / standard deviation
of its silhouette scores, which rewards groupings where every ETF fits well, and the best
one is kept. The book's refinement step (re-clustering low-quality clusters) is not
implemented.

## Findings (window ending 2021-12-31)

### Raw tree (03-11-2026)
- Three top-level structures, matching the three signal factors found in denoising:
  risk appetite (risk-on vs risk-off branches), rates (Treasuries form the tightest group,
  TLT–IEF at 0.18), and the dollar (UUP joins last, at 0.69: almost uncorrelated with
  everything). Gold is also a loner (0.54), separately from the dollar
- Against my expected groups: credit, equities and energy merge into one risk-on branch
  (HYG sits inside the equities, no separate energy group); Treasuries and the yen/franc
  form one risk-off branch; LQD moves to the risk-on branch

### Denoised tree (03-11-2026)
- Joins compress from 0.18–0.54 to 0.37–0.49: the near-duplicate pairs were carried by the
  smallest eigenvalues, which denoising flattens
- Gold joins the yen and franc as a safe-haven branch; TIP pairs with LQD
- This supports using the raw tree for HRP, as published

### ONC (04-11-2026)
- Two clusters: a rate-sensitive group (Treasuries, investment-grade and EM bonds, gold,
  yen, franc, Australian dollar) and a growth/commodity group (equities, REITs, high yield,
  commodities, energy, dollar)
- Consistent with denoising: 3 signal factors, minus the market factor removed by
  detoning, leaves 2
- The dollar sits apart from the other currencies because UUP is close to their mirror
  image; gold sits with them, as it tends to rise when the dollar and real rates fall

## Caveats
- The dendrogram's colours are a plotting default, not clusters chosen by a method
- Single linkage chains: loose ETFs attach one at a time rather than forming groups
- All findings are for one window; the tree and the clusters are rebuilt at every rebalance