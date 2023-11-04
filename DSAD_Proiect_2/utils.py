import numpy as np
import pandas as pd
import scipy.stats as sts


# Replace NA (not available, not applicable) or NaN (not a number) cells
# with column (variable) mean
# for a pandas DataFrame
def replace_NA(X):
    avgs = np.nanmean(X, axis=0)
    pos = np.where(np.isnan(X))
    print(pos[:])
    X[pos] = avgs[pos[1]]
    return X

# Standardize the column (variable) values
# for a pandas DataFrame
def standardize(X):
    # avgs = np.mean(X, axis=0 )
    avgs = 0
    stds = np.std(X, axis=0)
    Xstd = (X - avgs) / stds
    return Xstd

#normalizare in [-1, 1]
def narmalizare(X):
    avgs = np.mean(X, axis=0)
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    return (X - avgs) / (maxs - mins)

def evaluate(C, alpha, R):
    n = np.shape(C)[0]
    # Compute scores
    S = C / np.sqrt(alpha)

    # Compute cosines
    C2 = C * C
    suml = np.sum(C2, axis=1)
    q = np.transpose(np.transpose(C2) / suml)

    # Compute contributions
    beta = C2 / (alpha * n)

    # Compute commonalities
    R2 = R * R
    Comun = np.cumsum(R2, axis=1)
    return S, q, beta, Comun

def bartlett_test(n, l, x, e):
    m, q = np.shape(l)
    v = np.corrcoef(x, rowvar=False)
    psi = np.diag(e)
    v_ = l @ np.transpose(l) + psi
    I_ = np.linalg.inv(v_) @ v
    det_v_ = np.linalg.det(I_)
    trace_I = np.trace(I_)
    chi2_computed = (n - 1 - (2 * m + 4 * q - 5) / 2) * (trace_I - np.log(det_v_) - m)
    dof = ((m - q) * (m - q) - m - q) / 2
    chi2_estimated = sts.chi2.cdf(chi2_computed, dof)
    return chi2_computed, chi2_estimated

def bartlett_factor(x):
    n, m = np.shape(x)
    r = np.corrcoef(x, rowvar=False)
    chi2_computed = -(n - 1 - (2 * m + 5) / 6) * np.log(np.linalg.det(r))
    dof = m * (m - 1) / 2
    chi2_estimated = 1 - sts.chi2.cdf(chi2_computed, dof)
    return chi2_computed, chi2_estimated

def bartlett_wilks(r, n, p, q, m):
    r_inv = np.flipud(r)
    l = np.flipud(np.cumprod(1 - r_inv * r))
    dof = (p - np.arange(m)) * (q - np.arange(m))
    chi2_computed = (-n + 1 + (p + q + 1) / 2) * np.log(l)
    chi2_estimated = 1 - sts.chi2.cdf(chi2_computed, dof)
    return chi2_computed, chi2_estimated
