

''' Sampling from the non-linear state-space model with
equations x_t+1 = atan(x_t) + theta_1 * u_t + v_t, v_t ~ N(0,1)
          y_t   = |x_t| + theta_1 * theta_2 + e_t, e_t ~ N(0,0.01).
True parameters used for sampling are not given explicitly in the
paper, so let us pick these ourselves: theta_1 = 1, theta_2 = 1.
'''

import numpy as np
import numba
from numba import njit, prange
from abc import ABC, abstractmethod


""" Linear toy model from Svensson et al. (2017)
"""

@njit # This makes it faster
def Svensson2017LinearSSM_loglik(ys, theta, us, temp):
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
    
    theta1 = theta[0]
    theta2 = theta[1]

    # Initial state: x_0 = [0, 0]
    m0, m1 = 0.0, 0.0

    P00, P01 = 0.0, 0.0 # 0.0, 0.0 initial condition
    P10, P11 = 0.0, 0.0

    ll = 0.0

    for i in range(1, len(ys)):
        # Predict x_i from x_{i-1}
        pm0 = m0 + theta1 * m1 + theta2 * us[i]
        pm1 = 0.1 * m1

        # Predict covariance: A P A' + I
        PP00 = P00 + theta1 * P10 + theta1 * P01 + theta1 * theta1 * P11 + 1.0
        PP01 = 0.1 * (P01 + theta1 * P11)
        PP10 = 0.1 * (P10 + theta1 * P11)
        PP11 = 0.01 * P11 + 1.0

        # Observation: y_i = x_i[0] + noise
        innovation = ys[i] - pm0
        S = PP00 + temp

        ll += -0.5 * (
            np.log(2.0 * np.pi)
            + np.log(S)
            + innovation * innovation / S
        )

        # Kalman gain
        K0 = PP00 / S
        K1 = PP10 / S

        # Update mean
        m0 = pm0 + K0 * innovation
        m1 = pm1 + K1 * innovation

        # Update covariance
        P00 = (1.0 - K0) * PP00
        P01 = (1.0 - K0) * PP01
        P10 = PP10 - K1 * PP00
        P11 = PP11 - K1 * PP01
   
    return ll


def Svensson2017LinearSSM_simulate(u, theta, T, rng=None):
    """Sample observations from the toy example 
    linear-Gaussian state-space model (under the true
    theta parameter) described in Svensson et al. (2017).

    Parameters
    -
    T : int, default=500
        Number of time steps to simulate.
    us : ndarray of shape (T,)
        List of exogenous input variables.

    Returns
    -
    ys : ndarray of shape (T,)
        Simulated observations generated from the state space model.
    """
    
    # If no random generator is provided,
    # use the default.
    if rng is None:
        rng = np.random.default_rng()

    xs = np.zeros((T,2)) # Storage for latent states
    ys = np.zeros((T,)) # Storage for observations
    xs[0,:] = [0., 0.] # Initial condition
    ys[0] = Svensson2017LinearSSM_observation_mean(xs[0,:],theta)
    for i in range(1, T):
        # Evolve latent space according to formula
        # in Svensson et al. (2017).
        xs[i,:] = rng.multivariate_normal(Svensson2017LinearSSM_transition_mean(xs[i-1,:], theta, u[i]), 
                                          np.eye(2))
        ys[i] = Svensson2017LinearSSM_observation_mean(xs[i,:],theta)
    return xs, ys


@njit
def Svensson2017LinearSSM_observation_mean(x, theta):
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
    return np.asarray([1., 0.]) @ x


@njit
def Svensson2017LinearSSM_transition_mean(x, theta, u):
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
    return np.array([[1, 0.8],[0, 0.1]]) @ x + np.array([-1, 0]) * u



""" non-linear state-space model with
equations x_t+1 = atan(x_t) + theta_1 * u_t + v_t, v_t ~ N(0,1)
          y_t   = |x_t| + theta_1 * theta_2 + e_t, e_t ~ N(0,0.01).
True parameters used for sampling are not given explicitly in the
paper, so let us pick these ourselves: theta_1 = 1, theta_2 = 1.
"""


@njit
def Svensson2017NonLinearSSM_observation_variance():
    return 0.01

@njit
def Svensson2017NonLinearSSM_transition_variance():
    return 1.0

@njit
def Svensson2017NonLinearSSM_observation_mean(x, theta):
    """ Deterministic part of the
    distribution ``p(y_t|x_t,theta)`` in the
    state-space model described above.
    
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
    return np.abs(x) + theta[0] * theta[1]


@njit
def Svensson2017NonLinearSSM_transition_mean(x, theta, u):
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
    return np.atan(x) + theta[0] * u


def Svensson2017NonLinearSSM_simulate(u, theta, T, rng=None):
    """Sample observations from the non-linear
    state-space model described in Svensson et al. (2017).

    Parameters
    -
    T     : int, default=500
            Number of time steps to simulate.
    theta : ndarray of shape (2,):
            Parameters of underlying state-space model.
    u     : ndarray of shape (T,):
            List of exogenous input variables.
    rng   : numpy rng object
            Random number generator to use for drawing samples,
            so that we can consistently look at the same
            observation sample for fair comparisons
            later down the line
          
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
    xs[0] = 0 # Initial condition
    ys[0] = rng.normal(Svensson2017NonLinearSSM_observation_mean(xs[0], theta),
                       Svensson2017NonLinearSSM_observation_variance())
    for i in range(1, T):
        # Evolve latent space according to formula
        # in Svensson et al. (2017).
        xs[i] = rng.normal(Svensson2017NonLinearSSM_transition_mean(xs[i-1],theta,u[i-1]), 
                           Svensson2017NonLinearSSM_transition_variance())
        ys[i] = rng.normal(Svensson2017NonLinearSSM_observation_mean(xs[i],theta),
                           Svensson2017NonLinearSSM_observation_variance())
    return xs, ys