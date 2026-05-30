import numpy as np
import numba
from numba import njit, prange
from abc import ABC, abstractmethod

""" Simple benchmark SSM for experimentation
    SSM is x_t+1 = theta * x_t + N(0,1)
           y_t = x_t + N(0.001)
"""

@njit # This makes it faster
def SimpleBenchmarkSSM_loglik(ys, theta, us, temp=0.001):
    """ Computes the log-likelihood of the tempered linear-Gaussian 
    state space model using the Kalman filter.
    
    Parameters
    -
    ys    : ndarray of shape (T,)
          List of observations on which to evaluate the log-likelihood.
    theta : float
          Parameter value at which to evaluate the log-likelihood.
    temp  : float
          Tempering strength of the state-space model for which to
          calculate the log-likelihood.
    us    : ndarray of shape (T,)
          Exogenous input variables.
    """
        
    # Initial state distribution: x_0 ~ N(0, 1)
    m = 0.0
    P = 1.0
    
    ll = 0.0
    T = len(ys)
    
    # Time t=0 observation update
    innovation = ys[0] - m
    S = P + temp
    ll += -0.5 * (np.log(2.0 * np.pi) + np.log(S) + (innovation ** 2) / S)
    
    # Kalman update for t=0
    K = P / S
    m = m + K * innovation
    P = (1.0 - K) * P

    # Loop through time steps
    for i in range(1, T):
        # Predict step
        pm = theta * m
        PP = (theta ** 2) * P + 1.0  # Process variance is 1.0
        
        # Update step
        innovation = ys[i] - pm
        S = PP + temp
        
        ll += -0.5 * (np.log(2.0 * np.pi) + np.log(S) + (innovation ** 2) / S)
        
        # Kalman Gain and posterior update
        K = PP / S
        m = pm + K * innovation
        P = (1.0 - K) * PP
        
    return ll


def SimpleBenchmarkSSM_simulate(theta, us, T, rng=None):
    """Sample observations from the toy example 
    linear-Gaussian state-space model (under the true
    theta parameter) described in Svensson et al. (2017).

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
        xs[i] = rng.normal(SimpleBenchmarkSSM_transition_mean(x=xs[i-1], u=us, theta=theta), 1)
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
    return theta * np.arctan(x)
