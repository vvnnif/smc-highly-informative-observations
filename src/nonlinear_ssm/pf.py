import numpy as np
import numba
from numba import njit, prange
from src.util import *
from src.nonlinear_ssm.ssm import *

""" Particle filter for the non-linear SSM in Svensson et al. (2017)
"""

@njit
def Svensson2017NonLinearSSM_bootstrap_pf(temp, n_particles, ys, us, theta, alternative_tempering=False):
    """ Bootstrap particle filter for the nonlinear
    state-space model defined above.
    
    Parameters
    -
    temp        : float
                  tempering parameter
    n_particles : int
                  number of particles to use
    ys          : ndarray of shape (T,)
                  List of observations from the 
                  underlying state-space model
    us          : ndarray of shape (T,)
                  List of exogenous input parameters
                  in the underlying state-space model
    theta       : ndarray of shape (2,)
                  Parameters of the underlying state-space
                  model
                  
    Returns
    -
    List of samples ``x_1:T`` of shape (n_particles,T) from the filtering
    distributions ``p(x_t|y_1:t,theta,temp)``, to be used
    in the approximation of the likelihood ``p(y_1:T|theta,temp)``
    
    List of internal resampling states ``a_2:T`` of shape (n_particlesT-1)

    Estimate of log-likelihood of the observations
    """
    
    # Ensure consistent shape for theta to make
    # numba work
    theta = np.asarray(theta).reshape(2,)
    
    T = len(ys) # Number of observations
    xs = np.zeros(shape=(n_particles,T)) # Particle storage
    aas = np.zeros(shape=(n_particles,T-1),dtype=np.int64) # Resampling storage
    
    # Baseline variance, only change if using the standard tempering scheme
    # (the one from Svensson et al.)
    var = 0.01
    if alternative_tempering == False:
        var += temp # Variance after tempering
    log_w = np.empty(n_particles) # Weight storage

    # Sample initial xs from ``p(x|theta)`` in the state-space model
    # Initial condition is just deterministically 0, so just set to 0
    xs[:,0] = np.zeros(n_particles)

    # Weight according to f(y_1|x_1,theta,temp) = N(|x_1| + theta1 * theta2, 0.01 + temp)
    # Recall: going from p(x|theta,temp) to p(x|y,theta,temp) = 
    # f(y|x,theta,temp)p(x|theta,temp) / p(y|theta,temp)
    # Recall: so up to x-proportionality, dQ(x)/dM(x) = f(y|x,theta,temp)    
    for i in range(n_particles):
        
        if alternative_tempering == False:
            log_w[i] = normal_logpdf(ys[0], loc=Svensson2017NonLinearSSM_observation_mean(xs[i,0],theta), 
                                            scale=var)
        else: # Alternatively, weigh according to f(y_1|x_1,theta,temp) = f(y_1|x_1,theta)^(1/temp)
            log_w[i] = normal_logpdf(ys[0], loc=Svensson2017NonLinearSSM_observation_mean(xs[i,0],theta), 
                                            scale=var) / temp
    
    # Iterate over timesteps
    lse = logsumexp(log_w) # Store logsumexp, use later
    log_lik = lse - np.log(n_particles) # Keep track of log-likelihood
    for t in range(1,T):

        # Normalize weights
        w_norm = np.exp(log_w - lse)
        
        # Resample indices according to the normalized weights
        aas[:,t-1] = multinomial_resample(w_norm)
        
        # Extend and reweight
        for i in range(n_particles):
            
            # Extend
            xs[i,t] = np.random.normal(
                                    loc=Svensson2017NonLinearSSM_transition_mean(
                                        xs[aas[i,t-1],t-1],
                                        theta=theta,
                                        u=us[t-1]
                                        ),
                                        scale=1)
            # Reweight
            if alternative_tempering == False:
                log_w[i] = normal_logpdf(ys[t], loc=Svensson2017NonLinearSSM_observation_mean(xs[i,t],theta), 
                                                scale=var)
            else: # Alternatively, weigh according to f(y_1|x_1,theta,temp) = f(y_1|x_1,theta)^(1/temp)
                log_w[i] = normal_logpdf(ys[t], loc=Svensson2017NonLinearSSM_observation_mean(xs[i,t],theta), 
                                                scale=var) / temp
        
        
        # Recall: likelihood is prod_t=1^T 1/N * sum_n=1^N g(y_t|x_t^n,temp)
        # So this can quickly be obtained from the PF by summing the
        # unnormalized weights
        lse = logsumexp(log_w) # Store logsumexp, reuse later
        log_lik += lse - np.log(n_particles) # Update log-likelihod

    return xs, aas, log_lik
