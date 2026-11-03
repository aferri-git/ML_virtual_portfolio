# Hierarchical clustering builds a tree from the bottom up:
    # 1. Start with every ETF on its own.
    # 2. Merge the two closest.
    # 3. Measure the distance from the new group to everything else (single linkage: the
    # istance to its closest member, as in the HRP paper), then repeat.
    # 4. Continue until everything sits in one group

# Imports
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# let's create a function to translate correlation to distance as future models will use it
# returns a distance matrix from a correlation matrix, using the formula d = sqrt((1-corr)/2)
def corr_to_distance(corr):
    float_correction = np.clip((1 - corr) / 2, 0, None) # rounding may cause some tiny negative which won't worrk in a square root
    return np.sqrt(float_correction)

# order the ETFs
# returns a list of tickers in the order they should be plotted, from left to right, based on hierarchical clustering
def tree_order(corr):
    dist = corr_to_distance(corr)
    link = linkage(squareform(dist, checks = False), method="single")
    order = leaves_list(link ) # the order of the leaves, left to right
    return list(corr.index[order]) # turn them into tickers

# calculate cluster variance
# returns the variance of a cluster of tickers, given the covariance matrix and the list of tickers in the cluster
def cluster_variance(cov, tickers):
    cluster_covariance = cov.loc[tickers,tickers] # calculate sub covariance of a group
    w = 1/np.diag(cluster_covariance) # inverse-variance weights inside the group
    w = w/w.sum() # normalized
    return w @ cluster_covariance @ w # cluster's variance