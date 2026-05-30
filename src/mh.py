import numpy as np
import numba
from numba import njit, prange
from src.util import *
from src.pf import *
from src.ssm import *


""" Implementation of standard Metropolis-Hastings for the linear
    state-space model from Svensson et al. (2017)
"""

def Svensson2017LinearSSM_mh(initial_theta, temp, ys, us, n_iters=1, proposal_sd=0.01, verbose=False):
    """Draw samples from the posterior ``p(theta|y_1:T,lambda)`` 
    of the tempered linear-Gaussian state space model described 
    in Svensson et al. (2017)  using a random-walk 
    Metropolis-Hastings algorithm with a bivariate normal proposal.
    
    Parameters
    -
    initial_theta : float
                    Initial point from whence to construct the
                    Metropolis-Hastings Markov chain.
    temp          : float
                    Tempering parameter of the distribution to sample
                    from.
    ys            : ndarray of shape (T,)
                    List of observations on which the distribution to
                    sample from is conditioned.
    n_iters       : int, default=1
                    Number of iterations to run the Metropolis-Hastings
                    algorithm for.
    proposal_sd   : float, default=0.01
                    Standard deviation of the proposal distribution.
    verbose       : Boolean, default=False
                    Whether or not to print additional diagnostic information
                    while running the algorithm
    """
    
    if(verbose):
        print(f'Starting Metropolis-Hastings run\n \
                Parameters: \n \
                initial position = {initial_theta}\n \
                tempering        = {temp}\n \
                iterations       = {n_iters}\n \
                proposal_sd      = {proposal_sd}')
    samples = np.zeros((n_iters,2))
    samples[0,:] = initial_theta
    # Cache the log-likelihood so it doesn't have to be computed
    # twice each loop
    current_log_posterior = Svensson2017LinearSSM_loglik(ys=ys, us=us, theta=initial_theta, temp=temp) \
                          + Svensson2017LinearSSM_log_prior(initial_theta)
    n_accepts = 0
    for i in range(1, n_iters):
        y = samples[i-1,:]
        # Random walk Metropolis-Hastings with multivariate normal proposal
        z = np.random.multivariate_normal(y, proposal_sd**2 * np.eye(2))
        d = np.random.uniform(0, 1)
        new_log_posterior = Svensson2017LinearSSM_loglik(ys=ys, us=us, theta=z, temp=temp) \
                          + Svensson2017LinearSSM_log_prior(z)
        # Compute acceptance probability in log scale to avoid underflow
        log_alpha = new_log_posterior - current_log_posterior
        if np.log(d) < min(log_alpha, 0):
            samples[i] = z
            current_log_posterior = new_log_posterior
            n_accepts += 1
        else:
            samples[i] = y
    if(verbose):
        print(f'Finished Metropolis-Hastings run \n \
                Acceptance rate: {n_accepts / n_iters}')
    return samples


def Svensson2017LinearSSM_log_prior(theta):
    """Returns the log of the prior distribution on the parameter.
    The prior is uniform on [0,2.5] x [-2.5,0].

    Parameters
    -
    theta : float
            Point at which to evaluate the log of the prior.
    """
    if 0 <= theta[0] <= 2.5 and -2.5 <= theta[1] <= 0:
        return 0.0
    else:
        return -np.inf
    

def Svensson2017LinearSSM_sample_prior(size=1):
    """Samples points from the prior distribution on theta.
    The prior is the uniform distribution on [0,2.5] x [-2.5,0].
    
    Parameters
    -
    size : int
           Number of points to sample.
    """
    if size == 1:
        return np.asarray([np.random.uniform(0, 2.5), np.random.uniform(-2.5,0)])
    else:
        return np.asarray([[np.random.uniform(0, 2.5), np.random.uniform(-2.5,0)] for _ in range(size)])


""" Implementation of particle Metropolis-Hastings (PMH)
for this state-space model.
"""

@njit
def Svensson2017NonLinearSSM_log_prior(theta):
    """Compute prior probability of observing
    parameter theta.
    
    Parameters
    -
    theta : ndarray of shape (2,)
            Parameter of state-space model
    """
    if 0 <= theta[0] <= 2 and 0 <= theta[1] <= 2:
        return 0.0
    else:
        return -np.inf

@njit
def Svensson2017NonLinearSSM_sample_prior(n=1):
    """Sample from uniform prior
    on theta.
    
    Parameters
    -
    n : int
        Number of samples to draw
    """
    return np.random.uniform(0,2,size=(n,2))


@njit
def Svensson2017NonLinearSSM_pmmh(
    theta, 
    xs,
    aas,
    log_lik, 
    us,
    ys,
    temp,
    n_iters=1, 
    proposal_sd=0.01, 
    verbose=False,
    alternative_tempering=False
    ):
    """Draw samples from the posterior ``p(theta|y_1:T,temp)`` 
    of the tempered non-linear state space model described above,
    using a random-walk particle Metropolis-Hastings algorithm with 
    bivariate normal proposal.
    
    Parameters
    -
    theta         : ndarray of shape (2,)
                    Initial parameter value
    xs            : ndarray of shape (n_particles,T)
                    Particles used for sampling the initial
                    parameter value
    aas           : ndarray of shape (n_particles,T-1)
                    Internal resampling state of the particle
                    filter used for sampling the initial parameter
                    value
    log_lik       : np.float64
                    Log-likelihood obtained from the particle filter
                    used when sampling the initial parameter value
    temp          : float
                    Tempering parameter of the distribution to sample
                    from.
    ys            : ndarray of shape (T,)
                    List of observations on which the distribution to
                    sample from is conditioned.
    n_iters       : int, default=1
                    Number of iterations to run the Metropolis-Hastings
                    algorithm for.
    proposal_sd   : float, default=0.01
                    Standard deviation of the proposal distribution.
    verbose       : Boolean, default=False
                    Whether or not to print additional diagnostic information
                    while running the algorithm
                    
    Returns
    -
    thetas : ndarray of shape (n_iters,2)
             List of M-H parameter samples
    xss    : ndarray of shape (n_iters, n_particles,T)
             List of internal particles for estimation of M-H samples
    aass   : ndarray of shape (n_iters,n_particles,T-1)
             List of internal particle filter resampling states for
             estimation of M-H samples
    """
    T = len(ys) # Number of observations
    n_particles = xs.shape[0] # Number of particles
    n_accepts = 0 # Keep track of acceptance rate
    var = 0.01 + temp # Observation variance after tempering
    
    # Keep track of all samples, including the initial sample
    thetas = np.zeros(shape=(n_iters+1,2))
    thetas[0] = theta.flatten()

    log_liks = np.zeros(shape=(n_iters,)) # Keep track of sample log likelihoods
    
    # Keep track of all internal states, including the initial state
    xss = np.zeros(shape=(n_iters+1,n_particles,T))
    aass = np.zeros(shape=(n_iters+1,n_particles,T-1),dtype=np.int64)
    xss[0,:,:] = xs
    aass[0,:,:] = aas
     
    for i in range(1,n_iters+1):
        
        # Propose new parameter according to a
        # random walk proposal
        # We have to do it explicitly (without, for example,
        # np.multivariate_normal) to make it work
        # with numba. Worth it!
        eps0 = np.random.normal()
        eps1 = np.random.normal()
        new_theta = np.array([
            thetas[i-1, 0] + proposal_sd * eps0,
            thetas[i-1, 1] + proposal_sd * eps1
        ])
        
        # Run particle filter using this new parameter value
        new_xss, new_aass, new_log_lik = Svensson2017NonLinearSSM_bootstrap_pf(
                                    temp=temp,
                                    n_particles=n_particles,
                                    ys=ys,
                                    us=us,
                                    theta=new_theta,
                                    alternative_tempering=alternative_tempering)
        
        # Sample value uniform on [0,1]
        d = np.random.uniform(0,1)
        
        # Compute log of the acceptance probability
        log_alpha = new_log_lik + Svensson2017NonLinearSSM_log_prior(new_theta) - \
                    log_lik - Svensson2017NonLinearSSM_log_prior(thetas[i-1])
        
        # If d < min(alpha,1), keep thetas[i], xss[:,:,i], aas[:,:,i] the same.
        # Otherwise, change them back to their previous values.
        if np.log(d) < min(log_alpha, 0):
            log_lik = new_log_lik
            xss[i,:,:] = new_xss
            aass[i,:,:] = new_aass
            thetas[i] = new_theta
            log_liks[i] = log_lik
            n_accepts += 1
        else:
            thetas[i] = thetas[i-1]
            log_liks[i] = log_lik
            xss[i,:,:] = xss[i-1,:,:]
            aass[i,:,:] = aass[i-1,:,:]
    
    accept_rate = n_accepts / n_iters
    return thetas[1:], xss[1:,:,:], aass[1:,:,:], log_liks, accept_rate


@njit
def Svensson2017NonLinearSSM_pmmh_adaptive(
    theta, 
    xs,
    aas,
    log_lik, 
    us,
    ys,
    temp,
    proposal_chol,
    n_iters=1, 
    verbose=False,
    alternative_tempering=False
    ):
    """Draw samples from the posterior ``p(theta|y_1:T,temp)`` 
    of the tempered non-linear state space model described above,
    using a random-walk particle Metropolis-Hastings algorithm with 
    bivariate normal proposal.
    This is an adaptive version of the PMMH algorithm, taking as random
    walk parameter the Cholesky decomposition of its desired covariance
    matrix, rather than a numerical standard deviation.
    
    Parameters
    -
    theta         : ndarray of shape (2,)
                    Initial parameter value
    xs            : ndarray of shape (n_particles,T)
                    Particles used for sampling the initial
                    parameter value
    aas           : ndarray of shape (n_particles,T-1)
                    Internal resampling state of the particle
                    filter used for sampling the initial parameter
                    value
    log_lik       : np.float64
                    Log-likelihood obtained from the particle filter
                    used when sampling the initial parameter value
    temp          : float
                    Tempering parameter of the distribution to sample
                    from.
    ys            : ndarray of shape (T,)
                    List of observations on which the distribution to
                    sample from is conditioned.
    n_iters       : int, default=1
                    Number of iterations to run the Metropolis-Hastings
                    algorithm for.
    proposal_chol : ndarray of shape (2,2)
                    Cholesky decomposition of the covariance matrix of the propoosal distribution
    verbose       : Boolean, default=False
                    Whether or not to print additional diagnostic information
                    while running the algorithm
                    
    Returns
    -
    thetas : ndarray of shape (n_iters,2)
             List of M-H parameter samples
    xss    : ndarray of shape (n_iters, n_particles,T)
             List of internal particles for estimation of M-H samples
    aass   : ndarray of shape (n_iters,n_particles,T-1)
             List of internal particle filter resampling states for
             estimation of M-H samples
    """
    T = len(ys) # Number of observations
    n_particles = xs.shape[0] # Number of particles
    n_accepts = 0 # Keep track of acceptance rate
    
    var = 0.01
    if alternative_tempering == False: # Do not change this when power tempering
        var += temp # Observation variance after tempering
    
    # Keep track of all samples, including the initial sample
    thetas = np.zeros(shape=(n_iters+1,2))
    thetas[0] = theta.flatten()

    log_liks = np.zeros(shape=(n_iters,)) # Keep track of sample log likelihoods
    
    # Keep track of all internal states, including the initial state
    xss = np.zeros(shape=(n_iters+1,n_particles,T))
    aass = np.zeros(shape=(n_iters+1,n_particles,T-1),dtype=np.int64)
    xss[0,:,:] = xs
    aass[0,:,:] = aas
     
    for i in range(1,n_iters+1):
        
        # Propose new parameter according to a
        # random walk proposal
        # We have to do it explicitly (without, for example,
        # np.multivariate_normal) to make it work
        # with numba. Worth it!
        eps0 = np.random.normal()
        eps1 = np.random.normal()
        delta_theta_0 = proposal_chol[0, 0] * eps0 + proposal_chol[0, 1] * eps1
        delta_theta_1 = proposal_chol[1, 0] * eps0 + proposal_chol[1, 1] * eps1
        
        new_theta = np.array([
            thetas[i-1, 0] + delta_theta_0,
            thetas[i-1, 1] + delta_theta_1
        ])
        
        # Run particle filter using this new parameter value
        new_xss, new_aass, new_log_lik = Svensson2017NonLinearSSM_bootstrap_pf(
                                    temp=temp,
                                    n_particles=n_particles,
                                    ys=ys,
                                    us=us,
                                    theta=new_theta,
                                    alternative_tempering=alternative_tempering)
        
        # Sample value uniform on [0,1]
        d = np.random.uniform(0,1)
        
        # Compute log of the acceptance probability
        log_alpha = new_log_lik + Svensson2017NonLinearSSM_log_prior(new_theta) - \
                    log_lik - Svensson2017NonLinearSSM_log_prior(thetas[i-1])
        
        # If d < min(alpha,1), keep thetas[i], xss[:,:,i], aas[:,:,i] the same.
        # Otherwise, change them back to their previous values.
        if np.log(d) < min(log_alpha, 0):
            log_lik = new_log_lik
            xss[i,:,:] = new_xss
            aass[i,:,:] = new_aass
            thetas[i] = new_theta
            log_liks[i] = log_lik
            n_accepts += 1
        else:
            thetas[i] = thetas[i-1]
            log_liks[i] = log_lik
            xss[i,:,:] = xss[i-1,:,:]
            aass[i,:,:] = aass[i-1,:,:]
    
    accept_rate = n_accepts / n_iters
    return thetas[1:], xss[1:,:,:], aass[1:,:,:], log_liks, accept_rate