# The allocators will be:
    # 60% SPY / 40% AGG: external reference
    # Equal weight accross eligible ETFs: to isolate the value added by any future specific method
    # Ammended: risk-weighted allocator
    # The minimum variance allocator
    # The hierarchical risk parity allocator
    # Nested Clustered Optimisation
    

# Imports    
import pandas as pd
import numpy as np 
from scipy.optimize import minimize
from vportfolio.params import CAP
from vportfolio import denoising, clustering


# 60/40 allocator
## the allocator will take the data built by the engine but will not use it as the 60/40 split does not depend on it
def sixty_forty(_): # makes sure to ignore any argument given
    return pd.Series({"SPY": 0.6, "AGG": 0.4})

# Equal weight allocator
## the eligibility criteria filtered at its minimum 9 ETFs so we do not need to edit CAP requirement
def equal_weights(window):
    number_eligibles = len(window.columns)
    weights = 1/number_eligibles # calculate proportional weights equally among eligibles
    return pd.Series(weights, index = window.columns)

# Riskw weighted allocator
## the eligibility criteria filtered at its minimum 9 ETFs so we do not need to edit CAP requirement
def risk_weighted(window):
    risk_score = 1 / window.std()
    weights = risk_score / risk_score.sum()
    return pd.Series(weights, index = window.columns)

# Minimum-variance weights for a given covariance matrix (part 2)
# returns a Series of weights, long-only, each between 0 and the cap
def min_var_weights(cov, cap = CAP):
    cov_values = np.array(cov) / np.diag(cov).mean() # scaling the values depending on input
    n = cov.shape[1] # changed: number of assets, taken from the covariance (there is no window here)
    starting_weights = np.ones(n)/n # set the weights to equal values
    # portfolio variance: for two assets A and B, var_portfolio = w_A² · var_A + w_B² · var_B + 2 · w_A · w_B · cov_AB
    objective = lambda w: w @ cov_values @ w # changed: uses cov_values
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1} # the sum must add to one. the dictionary format is needed for scipy's minimize
    bounds = [(0, cap)] * n # each weight between 0 and the cap
    result = minimize(objective, starting_weights, method="SLSQP",
                  bounds=bounds, constraints=constraints,
                  options={"maxiter": 1000}) # use scipy minimizer with a safety margin
    if not result.success:
        raise ValueError(f"min_var optimisation failed: {result.message}") # Check that it worked before returning
    return pd.Series(result.x, index=cov.columns) # changed: labels come from the covariance

# Minimum variance allocator (part 1, then part 2)
# returns a Series with the weights of the ETFs in the window
def min_var(window, cap = CAP, denoise = False):
    if denoise:
        cov = denoising.denoised_cov(window) # take denoised cov if set to true
    else:
        cov = window.cov() # else take regular cov
    return min_var_weights(cov, cap) # the optimisation itself

# We need to write two small wrapper functions to call the min_var function with the denoise argument set to True or False. 
# This is needed because:
    # The engine will call the allocator without any arguments, and we need to specify whether we want to denoise or not
    # the manke_new_row will record the aallocator name with allocator.__name__ so we need to have two different functions to record the correct name
def min_var_raw(window):
    return min_var(window, denoise=False)

def min_var_denoised(window):
    return min_var(window, denoise=True)

# Hierarchical Risk Parity (HRP) allocator
# returns a series with the weights of the ETFs in the window, using the HRP algorithm
def hrp(window):
    cov = window.cov()
    order = clustering.tree_order(window.corr())
    w = pd.Series(1.0, index=order) # every ETF start at weight 1
    groups = [order] # the starting group is the whole ETFs list as per tree order
    while groups:
        new_groups = []
        for g in groups:
            if len(g) == 1:
                continue
            half = len(g) // 2 # // divides and rounds down: 4 → 2, 5 → 2
            left, right = g[:half], g[half:] # first half, second half
            v_left = clustering.cluster_variance(cov, left)
            v_right = clustering.cluster_variance(cov, right)
            alpha = 1 - v_left / (v_left + v_right) # left half's share
            w.loc[left] *= alpha # scale the left half's weights
            w.loc[right] *= 1 - alpha # and the right half's
            new_groups += [left, right]
        groups = new_groups
    return w.reindex(window.columns)

# Do all the math related to the NCO allocator
# will return an optimized set of final weights
def nco_weights(cov, labels):
    clusters = labels.unique()
    W = pd.DataFrame(0.0, index=cov.index, columns=clusters)
    # Split the matrix in sub groups based on NCO's labels
    for g in clusters:
        mask = cov.index.isin(labels.index[labels == g]) # I use a mask to filter the entire matrix into sub groups
        sub_cov = cov.loc[mask, mask]
        # Run a min_var allocator to obtain sub groups composition
        W.loc[mask, g] = min_var_weights(sub_cov,cap=None) # cap set to none or the function will fail as there need to be at least 5 assets
    # Calculate the reduced covariance matrix between clusters
    cov_r = W.T @ cov @ W
    # Run min_var allocator between clusters
    between_w = min_var_weights(cov_r, cap=None)
    # final asset weights
    return W @ between_w

# Run NCO repeatedly, locking any name that goes above cap at the cap and removing it from the problem
# returns a series of weights indexed by assets
def nco_capped(cov, labels, cap=CAP):
    if len(cov) * cap < 1: # feasibility check
        raise ValueError(f"cap {cap} infeasible for {len(cov)} assets")
    w = pd.Series(0.0, index=cov.index)
    free = cov.index
    budget = 1.0
    while True:
        sub = nco_weights(cov.loc[free, free], labels[free]) * budget # run NCO on free assets only, and use budget to scale them down to what is actually left
        over = sub > cap + 1e-12 # boolean mask of which assets went over the cap (excluding rounding noise)
        if not over.any():
            w[free] = sub
            return w
        w[sub.index[over]] = cap
        budget -= cap * over.sum()
        free = sub.index[~over]

# Obtain the final nco weights after adjusting the inputs
# will return a series with capped weights
def nco(window, cap=CAP):
    t = window.shape[0]
    corr_dt = denoising.detone_corr(denoising.denoise_corr(window.corr(), t)) # correlation: denoise → detone
    labels = clustering.onc(corr_dt) # ONC on that correlation
    cov = denoising.denoised_cov(window) # covariance: denoised only
    return nco_capped(cov, labels, cap).reindex(window.columns) # NCO with the cap applied inside