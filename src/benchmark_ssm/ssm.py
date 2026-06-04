import numpy as np
import numba
from numba import njit, prange
from abc import ABC, abstractmethod

""" Simple benchmark SSM for experimentation
"""


def SimpleBenchmarkSSM_simulate(theta, us, T, rng=None):
    """Sample observations from benchmark model 

    Parameters
    -
    T : int, default=500
        Number of time steps to simulate.

    Returns
    -
    ys : ndarray of shape (T,)
        Simulated observations generated from the state space model.
    """
    
    # If no random generator is provided,
    # use the default.
    if rng is None:
        rng = np.random.default_rng()

    xs = np.zeros((T,)) # Storage for latent states
    ys = np.zeros((T,)) # Storage for observations
    xs[0] = 0. # Initial condition
    ys[0] = SimpleBenchmarkSSM_observation_mean(xs[0],theta)
    for i in range(1, T):
        # Evolve latent space according to formula
        # in Svensson et al. (2017).
        xs[i] = rng.normal(SimpleBenchmarkSSM_transition_mean(x=xs[i-1], u=us[i], theta=theta), 1)
        ys[i] = rng.normal(SimpleBenchmarkSSM_observation_mean(x=xs[i],theta=theta), 0.001)
    return xs, ys


@njit
def SimpleBenchmarkSSM_observation_mean(x, theta):
    """ Deterministic part of the
    distribution ``p(y_t|x_t,theta)`` in the
    state-space model.
    
    Parameters
    -
    x              : float
                     Latent space variable
    theta          : ndarray of shape (2,)
                     Parameters for the state-space model
                     
    Returns
    -
    Deterministic part of observation
    """
    return x


@njit
def SimpleBenchmarkSSM_transition_mean(x, theta, u):
    """ Deterministic part of the
    distribution ``p(x_t|x_t-1,theta)`` in the
    state-space model described above.
    
    Parameters
    -
    x              : float
                     Previous latent-space state
    theta          : ndarray of shape (2,)
                     Parameters for the state-space model
    u              : float
                     Previous exogenous input parameter
    
    Returns
    -
    Deterministic part of next latent-space state
    """
    return x + theta * np.tanh(x) + u
