# Hierarchical clustering builds a tree from the bottom up:
    # 1. Start with every ETF on its own.
    # 2. Merge the two closest.
    # 3. Measure the distance from the new group to everything else (single linkage: the
    # istance to its closest member, as in the HRP paper), then repeat.
    # 4. Continue until everything sits in one group

# Imports
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

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

# try k = 2..max_k clusters with k-means, score each grouping by the silhouette (mean ÷ std)
# Return the best grouping as a Series of cluster labels indexed by ticker
def onc(corr, max_k=10, n_init=10):
    dist = corr_to_distance(corr)
    max_k = min(max_k, len(corr) - 1) # silhouettes need fewer clusters than ETFs
    best_score, best_labels = -np.inf, None # list to store best results. -np.inf makes sure that the first grouping will beat it
    for k in range(2, max_k + 1): # start at 2 and add 1 cluster to each iteration
        for seed in range(n_init): # for each k, try n_init random starts for cluster positioning
            labels = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(dist).labels_ # run the k-means optimizer for cluster centroids
            sil = silhouette_samples(dist, labels) # apply the scoring method to the k-means optimizer
            score = sil.mean() / sil.std()  # score the whole group with one number
        if score > best_score: # if this group beats best score
            best_score, best_labels = score, labels # add it to the empty series                        
    return pd.Series(best_labels, index=corr.index)