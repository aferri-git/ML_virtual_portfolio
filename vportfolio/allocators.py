# The allocators will be:
    # 60% SPY / 40% AGG: external reference
    # Equal weight accross eligible ETFs: to isolate the value added by any future specific method
    # Ammended: risk-weighted allocator

# Imports    
import pandas as pd
import numpy as np 


# 60/40 allocator
# the allocator will take the data built by the engine but will not use it as the 60/40 split does not depend on it
# it will return the portfolio allocatio and the engine will trade it
def sixty_forty(_): # makes sure to ignore any argument given
    return pd.Series({"SPY": 0.6, "AGG": 0.4})

# Equal weight allocator
# the eligibility criteria filtered at its minimum 9 ETFs so we do not need to edit CAP requirement
def equal_weights(window):
    number_eligibles = len(window.columns)
    weights = 1/number_eligibles # calculate proportional weights equally among eligibles
    return pd.Series(weights, index = window.columns)

# Riskw weighted allocator
# the eligibility criteria filtered at its minimum 9 ETFs so we do not need to edit CAP requirement
def risk_weighted(window):
    risk_score = 1 / window.std()
    weights = risk_score / risk_score.sum()
    return pd.Series(weights, index = window.columns)

