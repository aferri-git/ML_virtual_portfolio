# The allocators will be:
    # 60% SPY / 40% AGG: external reference
    # Equal weight accross eligible ETFs: to isolate the value added by any future specific method
    # Ammended: risk-weighted allocator
    # The minimum variance allocator
    

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

# Minimum variance allocator
# returns a dataframe with the weights of the ETFs in the window, using the minimum variance algorithm
def min_var(window, cap = CAP, denoise = False):
    if denoise:
    # use the array definition to keep the math fast
        cov = np.array(denoising.denoised_cov(window)) # take denoised cov if set to true
    else: 
        cov = np.array(window.cov()) # else take regular cov
    cov = cov * 252 * 10000 # annualised, in percent^2: same best weights, better-sized numbers, will allow the minimizer to run properly by scaling the covariance
    n = window.shape[1] # number of ETFs
    starting_weights = np.ones(n)/n # set the weights to equal values
    # below is the appliocation of portfolio variance's formula: for two assets A and B, covariance_A_B  = w_A² · var_A + w_B² · var_B + 2 · w_A · w_B · cov_AB
    objective = lambda w: w @ cov @ w # sets the objective to the min var portfolio
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1} # the sum must add to one. the dictionary format is needed for scipy's minimize
    bounds = [(0, cap)] * n # set the cap between 0 and desired cap (20%)
    result = minimize(objective, starting_weights, method="SLSQP",
                  bounds=bounds, constraints=constraints,
                  options={"maxiter": 1000}) # use scipy minimizer with a safety margin
    if not result.success:
        raise ValueError(f"min_var optimisation failed: {result.message}") # Check that it worked before returning
    return pd.Series(result.x, index=window.columns)

# We need to write two small wrapper functions to call the min_var function with the denoise argument set to True or False. 
# This is needed because:
    # The engine will call the allocator without any arguments, and we need to specify whether we want to denoise or not
    # the manke_new_row will record the aallocator name with allocator.__name__ so we need to have two different functions to record the correct name
def min_var_raw(window):
    return min_var(window, denoise=False)

def min_var_denoised(window):
    return min_var(window, denoise=True)


# Hierarchical Risk Parity (HRP) allocator
# returns a dataframe with the weights of the ETFs in the window, using the HRP algorithm
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