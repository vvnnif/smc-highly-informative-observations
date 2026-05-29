""" Some general utitlity functions to be used across files.
"""


import numpy as np
import numba
from numba import njit, prange


@njit
def ess(cur_log_pos, nxt_log_pos):
    """Compute the effective sample size for the set
    of weights the log of which is given by nxt_log_pos - cur_log_pos.
    
    Parameters
    -
    cur_log_opos  : ndarray of size (N,)
                    List of logs of evaluations of the 'proposal distribution'
                    (up to proportionality) in a set of sample points.
    nxt_log_pos   : ndarray of size (N,)
                    List of logs of evaluations of the 'target distribution'
                    (up to proportionality) in a set of sample points.
    
    Comments
    -
    Works in log-scale to avoid numerical difficulties.
    """
    
    cur = np.asarray(cur_log_pos)
    nxt = np.asarray(nxt_log_pos)

    log_w = nxt - cur

    m = np.max(log_w)
    w = np.exp(log_w - m)
    w = w / np.sum(w)

    return 1.0 / np.sum(w**2)


@njit
def normal_logpdf(x, loc, scale):
    """ Compute log pdf of normal distribution
    with mean loc and variance scale.
    """
    return -0.5 * np.log(2*np.pi*scale) - 0.5 * (x - loc)**2 / scale


@njit
def logsumexp(x):
    """Fast numerically stable logsumexp
    """
    m = np.max(x)

    s = 0.0
    for i in range(len(x)):
        s += np.exp(x[i] - m)

    return m + np.log(s)


@njit
def multinomial_resample(weights):
    """Get resampled indices based on
    normalized weight vector.
    Fast njit-compatible version of
    multinomial resampling
    """
    N = len(weights)
    
    cdf = np.empty(N)

    cdf[0] = weights[0]

    for i in range(1, N):
        cdf[i] = cdf[i-1] + weights[i]

    out = np.empty(N, dtype=np.int64)

    for i in range(N):

        u = np.random.rand()
        out[i] = np.searchsorted(cdf, u)

    return out
