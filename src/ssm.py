

''' Sampling from the non-linear state-space model with
equations x_t+1 = atan(x_t) + theta_1 * u_t + v_t, v_t ~ N(0,1)
          y_t   = |x_t| + theta_1 * theta_2 + e_t, e_t ~ N(0,0.01).
True parameters used for sampling are not given explicitly in the
paper, so let us pick these ourselves: theta_1 = 1, theta_2 = 1.
'''

import numpy as np
import numba
from numba import njit, prange


@njit
def g(x, theta):
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
def f(x, theta, u):
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

    

def sample_state_space_model(us, T=200, theta=np.asarray([1,1]), rng=None):
    """Sample observations from the non-linear
    state-space model described in Svensson et al. (2017).

    Parameters
    -
    T     : int, default=500
            Number of time steps to simulate.
    theta : ndarray of shape (2,):
            Parameters of underlying state-space model.
    us    : ndarray of shape (T,):
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
    ys[0] = rng.normal(g(xs[0], theta),0.01)
    for i in range(1, T):
        # Evolve latent space according to formula
        # in Svensson et al. (2017).
        xs[i] = rng.normal(f(xs[i-1],theta,us[i-1]), 1)
        ys[i] = rng.normal(g(xs[i],theta),0.01)
    return ys