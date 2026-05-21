import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.stats import norm
from numba import njit, prange
from concurrent.futures import ProcessPoolExecutor
numba.config.FULL_TRACEBACKS = True


''' Sampling from the non-linear state-space model with
equations x_t+1 = atan(x_t) + theta_1 * u_t + v_t, v_t ~ N(0,1)
          y_t   = |x_t| + theta_1 * theta_2 + e_t, e_t ~ N(0,0.01).
True parameters used for sampling are not given explicitly in the
paper, so let us pick these ourselves: theta_1 = 1, theta_2 = 1.
'''

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
    return np.atan(x) + theta[0] + u

    

def sample_state_space_model(us, T=200, theta=np.asarray([1,1])):
    """Sample observations from the non-linear
    state-space model described in Svensson et al. (2017).

    Parameters
    -
    T : int, default=500
        Number of time steps to simulate.
    theta : ndarray of shape (2,):
        Parameters of underlying state-space model.
    us : ndarray of shape (T,):
        List of exogenous input variables.

    Returns
    -
    ys : ndarray of shape (T,)
        Simulated observations generated from the state space model.
    """

    xs = np.zeros((T,)) # Storage for latent states
    ys = np.zeros((T,)) # Storage for observations
    xs[0] = 0 # Initial condition
    ys[0] = np.random.normal(g(xs[0], theta),0.01)
    for i in range(1, T):
        # Evolve latent space according to formula
        # in Svensson et al. (2017).
        xs[i] = np.random.normal(f(xs[i-1],theta,us[i-1]), 1)
        ys[i] = np.random.normal(g(xs[i],theta),0.01)
    return ys


@njit
def normal_logpdf(x, loc, scale):
    """ Compute log pdf of normal distribution
    with mean loc and variance scale.
    """
    z = (x - loc) / scale
    return -0.5 * z*z - np.log(scale) - 0.5*np.log(2*np.pi) 

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


@njit
def particle_filter(temp, n_particles, ys, us, theta):
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
    
    var = 0.01 + temp # Variance after tempering
    log_w = np.empty(n_particles) # Weight storage

    # Sample initial xs from ``p(x|theta)`` in the state-space model
    # Initial condition is just deterministically 0, so just set to 0
    xs[:,0] = np.zeros(n_particles)

    # Weight according to f(y_1|x_1,theta,temp) = N(|x_1| + theta1 * theta2, 0.01 + temp)
    # Recall: going from p(x|theta,temp) to p(x|y,theta,temp) = 
    # f(y|x,theta,temp)p(x|theta,temp) / p(y|theta,temp)
    # Recall: so up to x-proportionality, dQ(x)/dM(x) = f(y|x,theta,temp)    
    for i in range(n_particles):
        
        log_w[i] = normal_logpdf(ys[0], loc=g(xs[i,0],theta), scale=var)
    
    # Iterate over timesteps
    log_lik = 0.0 # Keep track of log-likelihood
    for t in range(1,T):

        # Recall: likelihood is prod_t=1^T 1/N * sum_n=1^N g(y_t|x_t^n,temp)
        # So this can quickly be obtained from the PF by summing the
        # unnormalized weights
        lse = logsumexp(log_w) # Store logsumexp, reuse later
        log_lik += lse - np.log(n_particles)

        # Normalize weights
        w_norm = np.exp(log_w - lse)
        
        # Resample indices according to the normalized weights
        aas[:,t-1] = multinomial_resample(w_norm)
        
        # Extend and reweight
        for i in range(n_particles):
            
            # Extend
            xs[i,t] = np.random.normal(loc=f(xs[aas[i,t-1],t-1],
                                         theta=theta,
                                         u=us[t-1]),
                                       scale=1)
            # Reweight
            log_w[i] = normal_logpdf(ys[t], loc=g(xs[i,t],theta), scale=var)

    return xs, aas, log_lik
    

@njit
def log_prior(theta):
    """Compute prior probability of observing
    parameter theta.
    
    Parameters
    -
    theta : ndarray of shape (2,)
            Parameter of state-space model
    """
    if -2 <= theta[0] <= 2 and -2 <= theta[1] <= 2:
        return 0.0
    else:
        return -np.inf

@njit
def sample_prior(n=1):
    """Sample from uniform prior
    on theta.
    
    Parameters
    -
    n : int
        Number of samples to draw
    """
    return np.random.uniform(-2,2,size=(n,2))


@njit
def particle_metropolis_hastings(
    theta, 
    xs,
    aas,
    log_lik, 
    us,
    ys,
    temp,
    n_iters=1, 
    proposal_sd=0.01, 
    verbose=False
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
        new_xss, new_aass, new_log_lik = particle_filter(temp=temp,
                                    n_particles=n_particles,
                                    ys=ys,
                                    us=us,
                                    theta=new_theta)
        
        # Sample value uniform on [0,1]
        d = np.random.uniform(0,1)
        
        # Compute log of the acceptance probability
        log_alpha = new_log_lik + log_prior(new_theta) - \
                    log_lik - log_prior(thetas[i-1])
        
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
def ess(cur_log_pos, nxt_log_pos):
    """Compute the effective sample size for the set
    of weights the log of which is given by nxt_log_pos - cur_log_pos.
    
    Parameters
    -
    cur_log_opos  : ndarray of size (N,)
                    List of logs of evaluations of the 'proposal distribution'
                    (up to proportionality) in a set of sample points.
    nxt_log_pos   : ndarray of size (N,)
                    List of logs of evaluations of the 'true distribution'
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
def get_log_pos(temp, thetas, ys, xss, aass):
    """ Compute the log of the
    posterior p(theta,(xs,aas)|temp,ys)
    up to proportionality.
    
    Parameters
    -
    temp   : float
             tempering parameter
    thetas : ndarray of size (n_sample_particles,2)
             list of parameter samples
    ys     : ndarray of size (T,)
             list of observations from the
             true state-space model
    xss    : ndarray of size (n_filter_particles,T,n_sample_particles)
             list of particles obtained for estimation
             of likelihoods p(ys|theta,(xs,aas),temp)
    aass   : ndarray of size (n_filter_particles,T-1,n_sample_particles)
             list of resampling indices obtained for
             estimation of likelihoods
             p(ys|theta,(xs,aas),temp)
    
    Returns
    -
    log_pos : ndarray of size (n_sample_particles,)
            list of logs of posteriors p(theta,(xs,aas)|temp,ys) up to
            proportionality, for theta in thetas
    """
    # recall xss.shape = (n_sample_particles,n_filter_particles,T)
    n_sample_particles, n_filter_particles, T = xss.shape
    var = 0.01 + temp # Observation variance in tempered model
    
    log_pos = np.zeros(shape=(n_sample_particles,),dtype=np.float64) # Store final result
    for m in range(n_sample_particles):
        
        for t in range(T):

            # Entries of w_t-1^n for fixed t
            inner = np.empty(shape=(n_filter_particles,))
            for n in range(n_filter_particles):
                inner[n] = normal_logpdf(ys[t],
                                         loc=g(xss[m,n,t],thetas[m]),
                                         scale=var)
            lse = logsumexp(inner) # Store logsumexp, need it later again
            log_pos[m] += lse

            # Second term only 1...T-1
            if t < T - 1:

                for n in range(n_filter_particles):

                    # w_t^n ~ g(y_t | x_t^a^n_{t+1},temp)
                    # so we want logpdf evaluated in y_t
                    # with mean g(x_t^a^n_{t+1},theta) and
                    # variance 0.01 + temp
                    log_pos[m] += normal_logpdf(ys[t],
                                           loc=g(xss[m,aass[m,n,t],t],thetas[m]),
                                           scale=var) \
                                - lse # Normalize
    return log_pos


@njit
def func_to_bisect(prev_log_pos, thetas, ys, xss, aass, temp):
    """ This is the function f: temp -> ess(prev_log_pos,nxt_log_pos(temp))
    Want to find value for temp s.t. f(temp) = 0.5*n_sample_particles

    Parameters
    -
    prev_log_pos : ndarray of shape (n_sample_particles,)
                 List of log of posteriors (up to proportionality)
                 for the previous tempering value
    thetas     : ndarray of shape (n_sample_particles,)
                 List of sample particles
    ys         : ndarray of shape (T,)
                 List of observations from the true
                 state-space model
    xss        : ndarray of shape (n_particles,T,n_sample_particles)
                 All particles used in likelihood approximation
                 corresponding to parameter values in thetas
    aass       : ndarray of shape (n_particles,T-1,n_sample_particles)
                 All resampling indices used in likelihood approxmimation
                 corresponding to parameter values in thetas
    temp       : np.float64
                 Tempering parameter

    Returns
    -
    The ESS of the weight-set computed using the previous log-posteriors
    and the next log-posteriors obtained by plugging in temp=temp
    """
    nxt_log_pos = get_log_pos(thetas=thetas,
                                 ys=ys,
                                 xss=xss,
                                 aass=aass,
                                 temp=temp)
    return ess(prev_log_pos, nxt_log_pos)


@njit
def bisect(prev_log_pos, 
           thetas, 
           ys, 
           xss, 
           aass, 
           c, 
           lo, 
           hi, 
           max_iter=50, 
           tol=1e-2, 
           verbose=False):
    """ Approximately compute the point of
    intersection between func_to_bisect 
    and the line y = c using the bisection method.
    
    Parameters
    -
    prev_log_pos : ndarray of shape (n_sample_particles,)
                 List of log of posteriors (up to proportionality)
                 for the previous tempering value
    thetas     : ndarray of shape (n_sample_particles,)
                 List of sample particles
    ys         : ndarray of shape (T,)
                 List of observations from the true
                 state-space model
    xss        : ndarray of shape (n_particles,T,n_sample_particles)
                 All particles used in likelihood approximation
                 corresponding to parameter values in thetas
    aass       : ndarray of shape (n_particles,T-1,n_sample_particles)
                 All resampling indices used in likelihood approxmimation
                 corresponding to parameter values in thetas
    c          : float
                 y-coordinate of the intersection line.
    lo         : float
                 x-coordinate of the lower extent of the 
                 initial bisection interval.
    hi         : float
                 x-coordinate of the upper extent of the
                 initial bisection interval.
    max_iter   : int
                 Number of iterations to try bisection for.
    tol        : float
                 How close to c a result can be to count
                 as an intersection. 
    verbose    : Boolean, default=False
                 Whether or not to display additional
                 diagnostic information.

    Returns
    -
    Tempering parameter for which the equation
    approximately holds.
    """
    
    lo, hi = lo, hi
    flo = func_to_bisect(xss=xss,
                         aass=aass,
                         prev_log_pos=prev_log_pos,
                         thetas=thetas,
                         ys=ys,
                         temp=lo) - c
    
    fhi = func_to_bisect(xss=xss,
                         aass=aass,
                         prev_log_pos=prev_log_pos,
                         thetas=thetas,
                         ys=ys,
                         temp=hi) - c
    
    # If many particles are active even at the lower bound
    # temperature, jump directly to the lower bound temperature.
    if flo > 0:
        return lo
    # Impossible for bisection to find an answer in this case.
    # This should not happen (algorithm would probably keep running forever
    # if we would not quit).
    if flo * fhi > 0:
        raise ValueError(f"Bisection requires opposite signs, got f(lo)-c={flo}, f(hi)-c={fhi}"
    )
    
    # Bisection method
    for i in range(max_iter):
        mid = (lo + hi) / 2

        fmid = func_to_bisect(xss=xss,
                         aass=aass,
                         prev_log_pos=prev_log_pos,
                         thetas=thetas,
                         ys=ys,
                         temp=mid) - c

                
        if np.abs(fmid) < tol:
            if verbose:
                print(f'Bisection converged! Midpoint is: {mid}, ESS at midpoint is: {fmid + c}.')
            return mid
        
        flo = func_to_bisect(xss=xss,
                         aass=aass,
                         prev_log_pos=prev_log_pos,
                         thetas=thetas,
                         ys=ys,
                         temp=lo) - c

        if flo * fmid < 0:
            hi = mid
        else:
            lo = mid

    if verbose:
        print(f'WARNING: Bisection did not converge.')
    return (lo + hi) / 2


@njit(parallel=True)
def run_parallel_mutation(
    thetas, xss, aass, log_liks, 
    temp, ys, us, mh_iters, mh_sd
):
    """
    Parallel mutation engine. 
    thetas: (n_sample_particles, 2)
    xss: (n_filter_particles, T, n_sample_particles)
    aass: (n_filter_particles, T-1, n_sample_particles)
    """
    n_samples = thetas.shape[0]
    accept_rates = np.empty(n_samples, dtype=np.float64)

    # Numba will distribute these iterations across all CPU cores
    for m in prange(n_samples):
        # We pass slices. Note: slicing the last dimension [:,:,m] is 
        # slow in NumPy but Numba handles it reasonably well.
        # Ensure particle_metropolis_hastings is @njit
        theta_chain, xss_chain, aass_chain, log_lik_chain, acc = particle_metropolis_hastings(
            theta=thetas[m],
            xs=xss[m,:,:],
            aas=aass[m,:,:],
            temp=temp,
            log_lik=log_liks[m],
            ys=ys,
            us=us,
            n_iters=mh_iters,
            proposal_sd=mh_sd,
            verbose=False
        )
        
        # Update the main arrays with the last link in the MH chain
        thetas[m] = theta_chain[-1]
        log_liks[m] = log_lik_chain[-1]
        xss[m,:,:] = xss_chain[-1,:,:]
        aass[m,:,:] = aass_chain[-1,:,:]
        accept_rates[m] = acc

    return accept_rates.mean()


def smc_sampler(ys, 
                us, 
                initial_mh_iters, 
                initial_mh_sd, 
                mh_iters, 
                mh_sd, 
                n_sample_particles,
                n_filter_particles, 
                temp, 
                min_temp, 
                alpha,
                max_smc_iters=30, 
                verbose=False
                ):
    """Tempered SMC sampler for approximate sampling from the
    parameter posterior of the linear-Gaussian state space model
    described in Svensson et al. (2017).
    
    Parameters
    -
    ys : ndarray of size (T,)
        List of observations to condition on in the posterior.
    us : ndarray of size (T,)
        List of exogenous input variables in the state-space model.
    initial_mh_iters : int
        Number of iterations to run the Metropolis-Hastings
        kernel for the initial sample.
    initial_mh_sd : float
        Standard deviation of the proposal distribution in the
        Metropolis-Hastings kernel for the initial sample.
    mh_iters : int
        Number of iterations to run the mutation Metropolis-Hastings
         kernel for.
    mh_sd : float
        Standard deviation of the proposal distribution in the mutation
        Metropolis-Hastings kernel.
    n_sample_particles : int
        Number of particles to simulate / number of samples to draw.
    n_filter_particles : int
        Number of particles to use for particle filtering
    temp: float
        Initial tempering parameter.
    min_temp : float
        Minimum tempering parameter; when reaching at most this temperature,
        terminate.
    alpha : float
            Adaptive tempering such that ESS remains approximately alpha * n_sample_particles
    max_smc_iters : int
        How many tempering iterations of the SMC sampler are allowed before
        we forcibly terminate.
    verbose : Boolean, default=False
        Whether or not to display additional diagnostic information
        
    Returns
    -
    samples : ndarray of size (n_particles,2)
              Approximate samples from the posterior p(theta|y_1:T,temp)
    """
    
    T = len(ys) # Number of observations
    p = 0 # Iteration counter
    ess_target = alpha * n_sample_particles # Aim to keep this ESS

    ######## Initial sample ########
    
    theta = sample_prior() # Starting parameter from prior
    
    # Particle filter, because particle approximations of
    # p(y_t|x_1:T,theta,temp) are needed for PMH
    xs, aas, log_lik = particle_filter(n_particles=n_filter_particles,
                              temp=temp,
                              ys=ys,
                              us=us,
                              theta=theta)
    
    # Draw the initial sample using PMH,
    # using a burn-in period of size initial_mh_iters - n_particles
    thetas, xss, aass, log_liks, accept_rate = particle_metropolis_hastings(theta=theta,
                                          xs=xs,
                                          aas=aas,
                                          temp=temp,
                                          log_lik=log_lik,
                                          us=us,
                                          ys=ys,
                                          n_iters=initial_mh_iters,
                                          proposal_sd=initial_mh_sd,
                                          verbose=False)
    thetas = thetas[-n_sample_particles:]
    log_liks = log_liks[-n_sample_particles:]
    xss = xss[-n_sample_particles:,:,:]
    aass = aass[-n_sample_particles:,:,:]
    
    ######## While tempering not sufficiently small... ########
    
    while temp > min_temp:
        
        p = p + 1 # Counter update

        if verbose:
            print(f'New iteration: p = {p}')
        
        ######## Adaptive tempering ########

        # Compute proportional log posterior for all samples
        # under the current tempering
        log_pos = get_log_pos(temp=temp,
                        thetas=thetas,
                        ys=ys,
                        xss=xss,
                        aass=aass)

        if verbose:
            print(f'Starting bisection...')
            t0 = time.time()

        # Intersect with ess_target to
        # find next tempering parameter
        temp = bisect(prev_log_pos=log_pos,
                          thetas=thetas,
                          ys=ys,
                          xss=xss,
                          aass=aass,
                          c=ess_target,
                          lo=min_temp,
                          hi=temp,
                          verbose=False)

        if verbose:
            t1 = time.time()
            print(f'Updated temperature, new temperature is: {temp}')
            print(f'Bisection took {t1-t0}s')

        nxt_log_pos = get_log_pos(temp=temp,
                          thetas=thetas,
                          ys=ys,
                          xss=xss,
                          aass=aass)

        ######## Normalize weights and resample ########

        log_w = nxt_log_pos - log_pos # weights propto next posterior / prev posterior
        w_norm = np.exp(log_w - logsumexp(log_w)) # Normalize

        idxs = multinomial_resample(w_norm) # Resampling indices

        # Resample
        thetas = thetas[idxs]
        log_liks = log_liks[idxs]
        xss = xss[idxs,:,:]
        aass = aass[idxs,:,:]
        
        ######## Mutate using PMH ########

        if verbose:
            print(f'Starting mutation...')
            t0 = time.time()

        # This one line replaces the entire mutation loop/pool logic
        mean_accept_rate = run_parallel_mutation(
            thetas, xss, aass, log_liks, 
            temp, ys, us, mh_iters, mh_sd
        )

        if verbose:
            t1 = time.time()
            print(f'Finished mutation. Mutation took {t1-t0}s')
            print(f'Mean accept rate: {mean_accept_rate}')

        # If accept rates of mutation steps are too bad,
        # terminate (heuristic from paper).
        if mean_accept_rate < 0.05:
            break 

    return thetas