# Test functions for clustering.py

# corr_to_distance
import pandas as pd
import numpy as np
from vportfolio.clustering import corr_to_distance, tree_order, cluster_variance


def test_corr_to_distance():
    rng = np.random.default_rng(0)
    random_corr = pd.DataFrame(rng.normal(size=(500, 6))).corr() # a random correlation matrix
    random_dist = corr_to_distance(random_corr) # its distances
    # 1. always true: symmetric, zero diagonal, every value between 0 and 1, no NaNs
    assert np.allclose(random_dist, random_dist.T)
    assert np.allclose(np.diag(random_dist), 0)
    assert ((random_dist >= 0) & (random_dist <= 1)).all().all()
    assert random_dist.notna().all().all()
    # 2. known answers: ρ = 1 → 0, ρ = 0 → √0.5 ≈ 0.707, ρ = −1 → 1
    corr_known = pd.DataFrame([[1, 1, 0],
                            [1, 1, -1],
                            [0, -1, 1]], index=list("ABC"), columns=list("ABC"))
    dist_known = corr_to_distance(corr_known)
    assert np.isclose(dist_known.loc["A", "B"], 0)
    assert np.isclose(dist_known.loc["A", "C"], np.sqrt(0.5))
    assert np.isclose(dist_known.loc["B", "C"], 1)
    # 3. inputs matter: the most correlated pair is the closest pair
    off_diag = ~np.eye(6, dtype=bool) # True everywhere except the diagonal
    corr_values = random_corr.where(off_diag) # hide the diagonal (NaN there)
    dist_values = random_dist.where(off_diag)
    assert corr_values.stack().idxmax() == dist_values.stack().idxmin()

# tree_order
def test_tree_order(): 
    # a block matrix: A and C move together, B and D move together, the two pairs unrelated
    corr = pd.DataFrame([[1.0, 0.0, 0.9, 0.0],
                        [0.0, 1.0, 0.0, 0.9],
                        [0.9, 0.0, 1.0, 0.0],
                        [0.0, 0.9, 0.0, 1.0]], index=list("ABCD"), columns=list("ABCD"))
    order = tree_order(corr)

    # 1. always true: every ticker appears exactly once
    assert sorted(order) == sorted(corr.columns)
    # 2. known answer: correlated pairs end up next to each other
    pos = {t: i for i, t in enumerate(order)}
    assert abs(pos["A"] - pos["C"]) == 1
    assert abs(pos["B"] - pos["D"]) == 1
    # 3. inputs matter: pair A with B instead, and the order follows
    corr_swap = corr.loc[list("ACBD"), list("ACBD")].copy()
    corr_swap.index = corr_swap.columns = list("ABCD")        # same pattern, relabelled: now A–B and C–D pair up
    pos2 = {t: i for i, t in enumerate(tree_order(corr_swap))}
    assert abs(pos2["A"] - pos2["B"]) == 1
    
# cluster_variance
def test_cluster_variance():
    cov = pd.DataFrame([[1, 0], [0, 4]], index=["A", "B"], columns=["A", "B"])
    result = cluster_variance(cov, ["A", "B"])
    # 1. always true: a valid covariance matrix
    assert result >= 0 # variance can never be negative
    assert result <= np.diag(cov).max() # never above the riskiest member's individual variance
    # 2. known answer: inverse-variance weights 0.8 / 0.2 → 0.8² × 1 + 0.2² × 4 = 0.8
    assert np.isclose(result, 0.8)
    # 3. inputs matter: 
    assert np.isclose(cluster_variance(cov, ["A"]), 1)        # a group of one: just its own variance
    cov_corr = pd.DataFrame([[1, 1], [1, 4]], index=["A", "B"], columns=["A", "B"])   # positive covariance
    assert cluster_variance(cov_corr, ["A", "B"]) > result        # moving together → less diversification → more risk