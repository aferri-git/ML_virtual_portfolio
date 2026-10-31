# Denoising Step
    # calculate correlation matrix
    # derive eigenvalues and eigenvectors
    # calculate the number of significant eigenvalues using the Marchenko-Pastur distribution
    # reconstruct the denoised correlation matrix using only the significant eigenvalues and their corresponding eigenvectors

# imports
import pandas as pd
import numpy as np
from scipy.stats import norm

# Apply the KDE to eigenvalues
def kde(eigenvalues, x, bandwidth = 0.25):
    # we use norm.pdf to create a normal distribution centered on each eigenvalues with bandwith 0.25 as per the book
    bells = []
    for e in eigenvalues:
        bells.append(norm.pdf(x, loc=e, scale=bandwidth))
    # then we take the average of the bells point by point
    kde_values = np.mean(bells, axis =0)
    return kde_values

# Show what the MP's distribution should look like
def mp_pdf(sigma2, q, x):
    lam_max = sigma2* (1+1/np.sqrt(q))**2 # we reuse the formula we deviced before and multiply it by new sigma2
    lam_min = sigma2* (1-1/np.sqrt(q))**2 # similar formula but with a minus inside parenthesis
    # MP's formula translated: height of the curve at each x
    inside = np.clip((lam_max - x) * (x - lam_min), 0, None)
    pdf = q / (2 * np.pi * sigma2 * x) * np.sqrt(inside)
    return pdf

# Return the σ² value whose MP distribution is closest to KDE
def fit_sigma2(eigenvals, q, bandwidth = 0.25):
    # first we created 1000 evenly spaced points from 0.001 to just passed the largest eigenvalue
    # this will build the X axis fo our distributions
    x_grid  = np.linspace(0.001, eigenvals.max()+1, 1000)
    # we then compute the KDE on it
    kde_values = kde(eigenvals, x_grid, bandwidth)
    # then we make a list of candidates: 1000 points from 0.01 to the largest eigenvalue+1
    candidates = np.linspace(0.01, 1, 1000)
    # for each of these candidates we compute the MP distribution and measure the gap with KDE
    gaps =[]
    for sigma2 in candidates:
        mp_values = mp_pdf(sigma2, q, x_grid)
        ssd = np.sum((kde_values -mp_values) ** 2)
        gaps.append(ssd)
    # we use the position of the smalles gap to return the best value for sigma2
    return candidates[np.argmin(gaps)]

# Apply all the denoising steps based on the ceiling method
# will return a dataframe with the new denoised matrix
def denoise_with_ceiling(corr, ceiling):
    # calculate eigenvals and eigenvect
    eigenvals, eigenvec = np.linalg.eigh(corr)
    # sort vals in descending order and flip the vector to follow the order
    eigenvals = np.flip(eigenvals)
    eigenvec = np.flip(eigenvec, axis=1)   # axis=1 → flip columns do to vectors' structure
    # filter noise-led eigenvals out
    noise_eigenvals = eigenvals [eigenvals <=ceiling]
    if len(noise_eigenvals) == 0:
        return corr # early return: nothing to denoise
    # calculate the average
    avg_noise_eigenvals = noise_eigenvals.mean()
    # set their value equal to their mean
    eigenvals_denoised = eigenvals.copy()
    # here we use a mask because is easier than a for loop
    eigenvals_denoised[eigenvals_denoised <= ceiling] = avg_noise_eigenvals
    # we now want to reverse engineer the matrix through the new set of denoised eigenvals
    denoised_matrix = eigenvec @ np.diag(eigenvals_denoised) @ eigenvec.T
    # rescale so the diagonal is exactly 1 (a proper correlation matrix)
    d = np.sqrt(np.diag(denoised_matrix))
    denoised_matrix = denoised_matrix / np.outer(d, d)
    # back into a DataFrame with the tickers
    denoised_corr = pd.DataFrame(denoised_matrix,
                             index=corr.index,
                             columns=corr.columns)
    return denoised_corr

# Join all functions together to calculate the denoised correlation matrix with the fitting methodology
    # calculate the σ² value whose MP distribution is closest to KDE
    # calculate the corresponding ceiling
    # apply all the denoising steps based on the ceiling method
    # will return a dataframe with the new denoised matrix based on fitted methodology
def denoise_corr(corr, t, bandwidth = 0.25):
    n = len(corr.columns)
    q = t / n
    eigenvals = np.linalg.eigvalsh(corr)
    sigma2 = fit_sigma2(eigenvals, q, bandwidth = bandwidth)
    ceiling = sigma2 * (1+1/np.sqrt(q))**2
    return denoise_with_ceiling(corr, ceiling)