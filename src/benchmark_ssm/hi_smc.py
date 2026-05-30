""" Implementation of the complete algorithm described in
Svensson et al., applied to their example nonlinear state-space
model.
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import numba
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
from scipy.stats import norm
from numba import njit, prange
from concurrent.futures import ProcessPoolExecutor
numba.config.FULL_TRACEBACKS = True

from src.util import *
from src.benchmark_ssm.ssm import *
from src.benchmark_ssm.pf import *
from src.benchmark_ssm.mh import *


""" Implement SMC sampler for benchmark SSM
"""

@njit
def SimpleBenchmarkSSM_log_pos(temp, thetas, ys, xss, aass, alternative_tempering=False):
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
    
    var = 0.001
    if alternative_tempering == False: # Only change variance when using standard tempering scheme
        var += temp 
    
    log_pos = np.zeros(n_sample_particles, dtype=np.float64)
    
    # Avoids unnecessary allocations
    inner = np.empty(n_filter_particles, dtype=np.float64)
    
    for m in range(n_sample_particles):
        theta_m = thetas[m]
        log_pos_m = 0.0
        
        for t in range(T):
            y_t = ys[t]
            
            # Calculate inner log-pdfs
            for n in range(n_filter_particles):
                if alternative_tempering == False:
                    inner[n] = normal_logpdf(y_t, loc=SimpleBenchmarkSSM_observation_mean(xss[m, n, t], theta_m), 
                                                scale=var)
                else: # Because f(y_1|x_1,theta,temp) = f(y_1|x_1,theta)^(1/temp) under alternative tempering scheme
                    inner[n] = normal_logpdf(y_t, loc=SimpleBenchmarkSSM_observation_mean(xss[m, n, t], theta_m), 
                                                scale=var) / temp
            
            lse = logsumexp(inner) 
            log_pos_m += lse

            # Second term for 0 to T-2
            if t < T - 1:
                for n in range(n_filter_particles):
                    idx = aass[m, n, t]
                    if alternative_tempering == False:
                        log_pos_m += normal_logpdf(y_t, loc=SimpleBenchmarkSSM_observation_mean(xss[m, idx, t], theta_m), 
                                                       scale=var) - lse
                    else:
                        log_pos_m += (normal_logpdf(y_t, loc=SimpleBenchmarkSSM_observation_mean(xss[m, idx, t], theta_m), 
                                                       scale=var) / temp) - lse
                    
        log_pos[m] = log_pos_m
        
    return log_pos


@njit
def SimpleBenchmarkSSM_bisect(prev_log_pos, 
           thetas, 
           ys, 
           xss, 
           aass, 
           c, 
           lo, 
           hi, 
           max_iter=50, 
           tol=1e-2, 
           verbose=False,
           alternative_tempering=False):
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
    
    # Evaluate lower bound
    nxt_log_pos_lo = SimpleBenchmarkSSM_log_pos(lo, thetas, ys, xss, aass, alternative_tempering)
    flo = ess(prev_log_pos, nxt_log_pos_lo) - c
    
    # Impossible in this case, return
    if flo > 0:
        return lo, nxt_log_pos_lo
        
    # Evaluate upper bound
    nxt_log_pos_hi = SimpleBenchmarkSSM_log_pos(hi, thetas, ys, xss, aass, alternative_tempering)
    fhi = ess(prev_log_pos, nxt_log_pos_hi) - c
    
    # Impossible in this case, return
    if flo * fhi > 0:
        return (lo + hi) / 2, nxt_log_pos_hi

    mid = (lo + hi) / 2
    nxt_log_pos_mid = SimpleBenchmarkSSM_log_pos(mid, thetas, ys, xss, aass, alternative_tempering)

    
    for i in range(max_iter):
        mid = (lo + hi) / 2
        nxt_log_pos_mid = SimpleBenchmarkSSM_log_pos(mid, thetas, ys, xss, aass, alternative_tempering)
        fmid = ess(prev_log_pos, nxt_log_pos_mid) - c
                
        # Found midpoint
        if np.abs(fmid) < tol:
            return mid, nxt_log_pos_mid
        
        # Sign check
        if flo * fmid < 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid

    return (lo + hi) / 2, nxt_log_pos_mid


@njit(parallel=True)
def SimpleBenchmarkSSM_parallel_mutate(thetas, 
                          xss, 
                          aass, 
                          log_liks, 
                          temp, 
                          ys, 
                          us, 
                          mh_iters, 
                          mh_sd,
                          alternative_tempering=False,
                          adaptive_pmh=False):
    """ Mutate particles using PMMH in parallel
    """
    n_samples = thetas.shape[0]
    accept_rates = np.empty(n_samples, dtype=np.float64)
    
    if adaptive_pmh:
        mh_sd = np.sqrt((2.38 / np.sqrt(2)) * np.var(thetas) + 1e-6)

    # Numba will distribute these iterations across all CPU cores
    for m in prange(n_samples):
        # We pass slices. Note: slicing the last dimension [:,:,m] is 
        # slow in NumPy but Numba handles it reasonably well.
        # Ensure particle_metropolis_hastings is @njit
        theta_chain, xss_chain, aass_chain, log_lik_chain, acc = SimpleBenchmarkSSM_pmmh(
                theta=thetas[m],
                xs=xss[m,:,:],
                aas=aass[m,:,:],
                temp=temp,
                log_lik=log_liks[m],
                ys=ys,
                us=us,
                n_iters=mh_iters,
                proposal_sd=mh_sd,
                verbose=False,
                alternative_tempering=alternative_tempering,
        )

        # Update the main arrays with the last link in the MH chain
        thetas[m] = theta_chain[-1]
        log_liks[m] = log_lik_chain[-1]
        xss[m,:,:] = xss_chain[-1,:,:]
        aass[m,:,:] = aass_chain[-1,:,:]
        accept_rates[m] = acc

    return accept_rates.mean()


def SimpleBenchmarkSSM_smc(ys, 
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
                verbose=False,
                alternative_tempering=False,
                adaptive_pmh=False
                ):
    """Tempered SMC sampler for approximate sampling from the
    parameter posterior of the nonlinear state space model
    described in Svensson et al. (2017).
    
    Parameters
    -
    ys                    : ndarray of size (T,)
                            List of observations to condition on in the posterior.
    us                    : ndarray of size (T,)
                            List of exogenous input variables in the state-space model.
    initial_mh_iters      : int
                            Number of iterations to run the Metropolis-Hastings
                            kernel for the initial sample.
    initial_mh_sd         : float
                            Standard deviation of the proposal distribution in the
                            Metropolis-Hastings kernel for the initial sample.
    mh_iters              : int
                            Number of iterations to run the mutation Metropolis-Hastings
                            kernel for.
    mh_sd                 : float
                            Standard deviation of the proposal distribution in the mutation
                            Metropolis-Hastings kernel.
    n_sample_particles    : int
                            Number of particles to simulate / number of samples to draw.
    n_filter_particles    : int
                            Number of particles to use for particle filtering
    temp: float
                            Initial tempering parameter.
    min_temp              : float
                            Minimum tempering parameter; when reaching at most this temperature,
                            terminate.
    alpha                 : float
                            Adaptive tempering such that ESS remains approximately alpha * n_sample_particles
    verbose               : Boolean, default=False
                            Whether or not to display additional diagnostic information
    alternative_tempering : Boolean, default=False
                            Whether to apply tempering in another way (see report)
    adaptive_pmh          : Boolean, default=False
                            Whether to keep mutation PMMH sd fixed, or to adapt it
                            to the the covariance of the particles                       
                            
    Returns
    -
    samples : ndarray of size (n_particles,2)
              Approximate samples from the posterior p(theta|y_1:T,temp)
    iteration_records : dictionary containing some diagnostic data
    """
    
    T = len(ys) # Number of observations
    p = 0 # Iteration counter
    iteration_records = {} # Keep diagnostic information
    ess_target = alpha * n_sample_particles # Aim to keep this ESS

    ######## Initial sample ########
    
    theta = SimpleBenchmarkSSM_sample_prior() # Starting parameter from prior
    
    # Particle filter, because particle approximations of
    # p(y_t|x_1:T,theta,temp) are needed for PMH
    xs, aas, log_lik = SimpleBenchmarkSSM_bootstrap_pf(
                            n_particles=n_filter_particles,
                            temp=temp,
                            ys=ys,
                            us=us,
                            theta=theta,
                            alternative_tempering=alternative_tempering)
    
    # Draw the initial sample using PMH,
    # using a burn-in period of size initial_mh_iters - n_particles
    thetas, xss, aass, log_liks, mean_accept_rate = SimpleBenchmarkSSM_pmmh(
                                        theta=theta,
                                        xs=xs,
                                        aas=aas,
                                        temp=temp,
                                        log_lik=log_lik,
                                        us=us,
                                        ys=ys,
                                        n_iters=initial_mh_iters,
                                        proposal_sd=initial_mh_sd,
                                        verbose=False,
                                        alternative_tempering=alternative_tempering)
    thetas = thetas[-n_sample_particles:]
    log_liks = log_liks[-n_sample_particles:]
    xss = xss[-n_sample_particles:,:,:]
    aass = aass[-n_sample_particles:,:,:]
        
    ######## While tempering not sufficiently small... ########
    
    while temp > min_temp:
        
        # Update diagnostics
        iteration_records[p] = {'temp': temp, 
                                'particles': thetas,
                                'log_likelihoods': log_liks,
                                'accept_rate': mean_accept_rate}
        
        p = p + 1 # Counter update

        if verbose:
            print(f'New iteration: p = {p}')
        
        ######## Adaptive tempering ########

        # Compute proportional log posterior for all samples
        # under the current tempering
        log_pos = SimpleBenchmarkSSM_log_pos(temp=temp,
                        thetas=thetas,
                        ys=ys,
                        xss=xss,
                        aass=aass)

        if verbose:
            print(f'Starting bisection...')
            t0 = time.time()

        # Intersect with ess_target to
        # find next tempering parameter
        temp, nxt_log_pos = SimpleBenchmarkSSM_bisect(prev_log_pos=log_pos,
                          thetas=thetas,
                          ys=ys,
                          xss=xss,
                          aass=aass,
                          c=ess_target,
                          lo=min_temp,
                          hi=temp,
                          verbose=False,
                          alternative_tempering=alternative_tempering)

        if verbose:
            t1 = time.time()
            print(f'Updated temperature, new temperature is: {temp}')
            print(f'Bisection took {t1-t0}s')

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
        mean_accept_rate = SimpleBenchmarkSSM_parallel_mutate(
                                            thetas=thetas, 
                                            xss=xss, 
                                            aass=aass, 
                                            log_liks=log_liks, 
                                            temp=temp, 
                                            ys=ys, 
                                            us=us, 
                                            mh_iters=mh_iters, 
                                            mh_sd=mh_sd,
                                            alternative_tempering=alternative_tempering,
                                            adaptive_pmh=adaptive_pmh
                                            )

        if verbose:
            t1 = time.time()
            print(f'Finished mutation. Mutation took {t1-t0}s')
            print(f'Mean accept rate: {mean_accept_rate}')

        # If accept rates of mutation steps are too bad,
        # terminate (heuristic from paper).
        if mean_accept_rate < 0.05:
            break 


    # Final diagnostic update
    iteration_records[p] = {'temp': temp, 
                            'particles': thetas,
                            'log_likelihoods': log_liks,
                            'accept_rate': mean_accept_rate}
    return thetas, iteration_records
