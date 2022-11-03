# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 22:13:26 2022

@author: cfai2
"""
import numpy as np
from scipy.integrate import solve_ivp, odeint
from multiprocessing import Pool
import os
import sys
import signal
from functools import partial

from forward_solver import dydt, dydt_numba
from sim_utils import MetroState, Grid, Solution
import pickle

## Constants
eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
q = 1.0 # [e]
q_C = 1.602e-19 # [C]
kB = 8.61773e-5  # [eV / K]
MIN_HMAX = 1e-2 # [ns]
MAX_TRIAL_ATTEMPTS = 100

def E_field(N, P, PA, dx, corner_E=0):
    if N.ndim == 1:
        E = corner_E + q_C / (PA.eps * eps0) * dx * np.cumsum(((P - PA.p0) - (N - PA.n0)))
        E = np.concatenate(([corner_E], E))
    elif N.ndim == 2:
        E = corner_E + q_C / (PA.eps * eps0) * dx * np.cumsum(((P - PA.p0) - (N - PA.n0)), axis=1)
        num_tsteps = len(N)
        E = np.concatenate((np.ones(shape=(num_tsteps,1))*corner_E, E), axis=1)
    return E

def model(init_dN, g, p, meas="TRPL", solver="solveivp", RTOL=1e-10, ATOL=1e-14):
    """ Calculate one simulation. """
    N = init_dN + p.n0
    P = init_dN + p.p0
    E_f = E_field(N, P, p, g.dx)
    
    init_condition = np.concatenate([N, P, E_f], axis=None)
    
    if solver=="solveivp":
        args = (g.nx, g.dx, p.n0, p.p0, p.mu_n, p.mu_p, p.ks, p.Cn, p.Cp, p.Sf, p.Sb, p.tauN, p.tauP, ((q_C) / (p.eps * eps0)), p.Tm)
        sol = solve_ivp(dydt_numba, [g.start_time,g.time], init_condition, args=args, t_eval=g.tSteps, method='LSODA', max_step=g.hmax, rtol=RTOL, atol=ATOL)
        data = sol.y.T
    elif solver=="odeint":
        args = (g.nx, g.dx, p.n0, p.p0, p.mu_n, p.mu_p, p.ks, p.Cn, p.Cp, p.Sf, p.Sb, p.tauN, p.tauP, ((q_C) / (p.eps * eps0)), p.Tm)
        data = odeint(dydt_numba, init_condition, g.tSteps, args=args, hmax=g.hmax, rtol=RTOL, atol=ATOL, tfirst=True)
    else:
        raise NotImplementedError

    s = Solution()
    s.N, s.P, E_f = np.split(data, [g.nx, 2*g.nx], axis=1)
    if meas == "TRPL":
        s.calculate_PL(g, p)
        return s.PL, (s.N[-1]-p.n0)
    elif meas == "TRTS":
        s.calculate_TRTS(g,p)
        return s.trts, (s.N[-1] - p.n0)
    else:
        raise NotImplementedError("TRTS or TRPL only")

def check_approved_param(new_p, param_info):
    """ Screen out non-physical or unrealistic proposed trial moves. """
    order = param_info['names']
    ucs = param_info.get('unit_conversions', {})
    checks = {}
    # mu_n and mu_p between 1e-1 and 1e6; a reasonable range for most materials
    if 'mu_n' in order:
        checks["mu_n_size"] = (new_p[order.index('mu_n')] - np.log10(ucs.get('mu_n', 1)) < 6) and (new_p[order.index('mu_n')] - np.log10(ucs.get('mu_n', 1)) > -1)
    else:
        checks["mu_n_size"] = True

    if 'mu_p' in order:
        checks["mu_p_size"] = (new_p[order.index('mu_p')] - np.log10(ucs.get('mu_p', 1)) < 6) and (new_p[order.index('mu_p')] - np.log10(ucs.get('mu_p', 1)) > -1)

    else:
        checks["mu_p_size"] = True

    if 'Sf' in order:
        checks["sf_size"] = (np.abs(new_p[order.index('Sf')] - np.log10(ucs.get('Sf', 1))) < 7)
    else:
        checks["sf_size"] = True

    if 'Sb' in order:
        checks["sb_size"] = (np.abs(new_p[order.index('Sb')] - np.log10(ucs.get('Sb', 1))) < 7)
    else:
        checks["sb_size"] = True

    # tau_n and tau_p must be *close* (within 3 OM) for a reasonable midgap SRH
    if 'tauN' in order and 'tauP' in order:
        checks["tn_tp_close"] = (np.abs(new_p[order.index('tauN')] - new_p[order.index('tauP')]) <= 3)
    else:
        checks["tn_tp_close"] = True

    approved = all(checks.values()) if len(checks) > 0 else True
    return approved

def select_next_params(p, means, variances, param_info, trial_function="box", logger=None):
    """ Trial move function: 
        box: uniform rectangle centered about current state. 
        gauss: gaussian centered about current state."""
    is_active = param_info["active"]
    do_log = param_info["do_log"]
    names = param_info["names"]
    mean = means.to_array(param_info)
    for i, param in enumerate(names):        
        if do_log[param]:
            mean[i] = np.log10(mean[i])
            
    cov = variances.cov
    approved = False
    attempts = 0
    while not approved:
        
        if trial_function == "box":
            new_p = np.zeros_like(mean)
            for i, param in enumerate(names):
                new_p[i] = np.random.uniform(mean[i] - cov[i,i], mean[i] + cov[i,i])
        elif trial_function == "gauss":
            try:
                assert np.all(cov >= 0)
                new_p = np.random.multivariate_normal(mean, cov)
            except Exception:
                if logger is not None:
                    logger.error("multivar_norm failed: mean {}, cov {}".format(mean, cov))
                new_p = mean
                
        approved = check_approved_param(new_p, param_info)
        attempts += 1
        if attempts > MAX_TRIAL_ATTEMPTS: break
    
    if logger is not None:
        logger.info(f"Found suitable parameters in {attempts} attempts")
        
    for i, param in enumerate(names):
        if is_active[param]:
            if do_log[param]:
                setattr(p, param, 10 ** new_p[i])
            else:
                setattr(p, param, new_p[i])
    return


def update_history(H, k, p, means, param_info):
    for param in param_info['names']:
        h = getattr(H, param)
        h[k] = getattr(p, param)
        h_mean = getattr(H, f"mean_{param}")
        h_mean[k] = getattr(means, param)
    return


def do_simulation(p, thickness, nx, iniPar, times, hmax, meas="TRPL", solver="solveivp", rtol=1e-10, atol=1e-14):
    g = Grid()
    g.thickness = thickness
    g.nx = nx
    g.dx = g.thickness / g.nx
    g.xSteps = np.linspace(g.dx / 2, g.thickness - g.dx/2, g.nx)
    
    g.time = times[-1]
    g.start_time = times[0]
    g.nt = len(times) - 1
    g.hmax = hmax
    g.tSteps = times
    
    sol, next_init_condition = model(iniPar, g, p, meas=meas, solver=solver, RTOL=rtol, ATOL=atol)
    return sol
    
def roll_acceptance(logratio):
    accepted = False
    if logratio >= 0:
        # Continue
        accepted = True
        
    else:
        accept = np.random.random()
        if accept < 10 ** logratio:
            # Continue
            accepted = True
    return accepted

def unpack_simpar(simPar, i):
    thickness = simPar[0][i] if isinstance(simPar[0], list) else simPar[0]
    nx = simPar[2]
    return thickness, nx

def anneal(t, anneal_mode, anneal_params):
    if anneal_mode is None or anneal_mode == "None":
        return anneal_params[2]

    elif anneal_mode == "exp":
        return anneal_params[0] * np.exp(-t / anneal_params[1]) + anneal_params[2]

    elif anneal_mode == "log":
        return (anneal_params[0] * np.log(2)) / (np.log(t / anneal_params[1] + 2)) + anneal_params[2]

    else:
        raise ValueError("Invalid annealing type")
        
def detect_sim_fail(sol, ref_vals):
    fail = len(sol) < len(ref_vals)
    if fail:
        sol2 = np.ones_like(ref_vals) * sys.float_info.min
        sol2[:len(sol)] = sol
        sol = np.array(sol2)
        
    return sol, fail

def detect_sim_depleted(sol):
    fail = np.any(sol < 0)
    if fail: sol = np.abs(sol) + sys.float_info.min
    return sol, fail

def almost_equal(x, x0, threshold=1e-10):
    if x.shape != x0.shape: return False

    return np.abs(np.nanmax((x - x0) / x0)) < threshold

def one_sim_likelihood(p, simPar, hmax, sim_flags, logger, args):
    i, iniPar, times, vals = args
    STARTING_HMAX = sim_flags["hmax"]
    RTOL = sim_flags["rtol"]
    ATOL = sim_flags["atol"]
    thickness, nx = unpack_simpar(simPar, i)
    hmax[i] = min(STARTING_HMAX, hmax[i] * 2) # Always attempt a slightly larger hmax than what worked at previous proposal
    
    while hmax[i] > MIN_HMAX:
        sol = do_simulation(p, thickness, nx, iniPar, times, hmax[i], meas=sim_flags["measurement"], 
                            solver=sim_flags["solver"], rtol=RTOL, atol=ATOL)
        
        #if verbose: 
        logger.info("{}: Simulation complete hmax={}; final t {}".format(i, hmax, times[len(sol)-1]))
        
        sol, fail = detect_sim_fail(sol, vals)
        if fail:
            logger.warning(f"{i}: Simulation terminated early!")

        sol, fail = detect_sim_depleted(sol)
        if fail:
            logger.warning(f"{i}: Carriers depleted!")
            hmax[i] = max(MIN_HMAX, hmax[i] / 2)
            logger.warning(f"{i}: Retrying hmax={hmax}")
        else:
            break
            # hmax[i] = max(MIN_HMAX, hmax[i] / 2)
            # if verbose:
            #     logger.info(f"{i}: Verifying convergence with hmax={hmax}...")
            # sol2 = do_simulation(p, thickness, nx, next_init_condition, times, hmax[i], meas=sim_flags["measurement"],
            #                      solver=sim_flags["solver"], rtol=RTOL, atol=ATOL)
            # if almost_equal(sol, sol2, threshold=RTOL):
            #     logger.info("Success!")
            #     break
            # else:
            #     if verbose:
            #         logger.info(f"{i}: Fail - not converged")
            #         if hmax[i] <= MIN_HMAX:
            #             logger.warning(f"{i}: MIN_HMAX reached")
    try:
        if sim_flags.get("self_normalize", False):
            sol /= np.nanmax(sol)
        # TODO: accomodate multiple experiments, just like bayes
        # skip_time_interpolation = almost_equal(sim_times, times)

        # if logger is not None: 
        #     if skip_time_interpolation:
        #         logger.info("Experiment {}: No time interpolation needed; bypassing".format(0))

        #     else:
        #         logger.info("Experiment {}: time interpolating".format(0))

        # if skip_time_interpolation:
        #     sol_int = sol
        # else:
        #     sol_int = griddata(sim_times, sol, times)

        likelihood = -np.sum((np.log10(sol) - vals)**2)

        # TRPL must be positive! Any simulation which results in depleted carrier is clearly incorrect
        if np.isnan(likelihood): raise ValueError(f"{i}: Simulation failed!")
    except ValueError as e:
        logger.warning(e)
        likelihood = -np.inf
    return likelihood
        

def run_iteration(p, simPar, iniPar, times, vals, hmax, sim_flags, verbose, logger, prev_p=None, t=0):
    # Calculates likelihood of a new proposed parameter set
    accepted = True
    logratio = 0 # acceptance ratio = 1
    p.likelihood = np.zeros(len(iniPar))
    
    if sim_flags.get("use_multi_cpus", False):
        raise NotImplementedError
        #with Pool(sim_flags["num_cpus"]) as pool:
        #    likelihoods = pool.map(partial(one_sim_likelihood, p, simPar, hmax, sim_flags, logger), zip(np.arange(len(iniPar)), iniPar, times, vals))
        #    p.likelihood = np.array(likelihoods)
                
    else:
        for i in range(len(iniPar)):
            p.likelihood[i] = one_sim_likelihood(p, simPar, hmax, sim_flags, logger, (i, iniPar[i], times[i], vals[i]))
            
    if prev_p is not None:
        T = anneal(t, sim_flags["anneal_mode"], sim_flags["anneal_params"])
        logratio = (np.sum(p.likelihood) - np.sum(prev_p.likelihood)) / T
        if verbose and logger is not None: 
            logger.debug(f"Temperature: {T}")
        if np.isnan(logratio): logratio = -np.inf
        
        if verbose and logger is not None: 
            logger.info("Partial Ratio: {}".format(10 ** logratio))
            
        
        accepted = roll_acceptance(logratio)

    if prev_p is not None and accepted:
        prev_p.likelihood = p.likelihood
    return accepted

def kill_from_cl(signal_n, frame):
    raise KeyboardInterrupt("Terminate from command line")

def all_signal_handler(func):
    for s in signal.Signals:
        try:
            signal.signal(s, func)
        except (ValueError, OSError):
            continue
    return

def metro(simPar, iniPar, e_data, sim_flags, param_info, initial_variance, verbose, logger, initial_guess):
    # Setup
    logger.info("PID: {}".format(os.getpid()))
    all_signal_handler(kill_from_cl)

    num_iters = sim_flags["num_iters"]
    DA_mode = sim_flags.get("delayed_acceptance", "off")
    checkpoint_freq = sim_flags["checkpoint_freq"]
    load_checkpoint = sim_flags["load_checkpoint"]
    STARTING_HMAX = sim_flags["hmax"]
    
    if sim_flags.get("use_multi_cpus", False):
        sim_flags["num_cpus"] = min(os.cpu_count(), len(iniPar))
        logger.info("Taking {} cpus".format(sim_flags["num_cpus"]))
    
    times, vals, uncs = e_data
    # As init temperature we take cN, where c is specified in main and N is number of observations
    sim_flags["anneal_params"][0] *= sum([len(time)-1 for time in times])
    sim_flags["anneal_params"][2] *= sum([len(time)-1 for time in times])

    if load_checkpoint is not None:
        with open(os.path.join(sim_flags["checkpoint_dirname"], load_checkpoint), 'rb') as ifstream:
            MS = pickle.load(ifstream)
            np.random.set_state(MS.random_state)
            starting_iter = int(load_checkpoint[load_checkpoint.find("_")+1:load_checkpoint.rfind(".pik")])+1
    else:
        MS = MetroState(param_info, initial_guess, initial_variance, num_iters)
        starting_iter = 1
    
        # Calculate likelihood of initial guess
        MS.running_hmax = [STARTING_HMAX] * len(iniPar)
        accept = run_iteration(MS.prev_p, simPar, iniPar, times, vals, MS.running_hmax, sim_flags, verbose, logger)
        update_history(MS.H, 0, MS.prev_p, MS.means, param_info)

    for k in range(starting_iter, num_iters):
        try:
            logger.info("#####")
            logger.info("Iter {}".format(k))
            logger.info("#####")
            # Identify which parameter to move
            if sim_flags.get("one_param_at_a_time", 0):
                picked_param = MS.means.actives[k % len(MS.means.actives)]
            else:
                picked_param = None
                
            # Select next sample from distribution
            
            if sim_flags.get("adaptive_covariance", "None") == "None":
                MS.variances.mask_covariance(picked_param)

            select_next_params(MS.p, MS.means, MS.variances, MS.param_info, sim_flags["proposal_function"], logger)
            
            if verbose: MS.print_status(logger)
                    
            if DA_mode == "off":
                accepted = run_iteration(MS.p, simPar, iniPar, times, vals, MS.running_hmax, sim_flags, verbose, logger, prev_p=MS.prev_p, t=k)
                
            elif DA_mode == 'DEBUG':
                accepted = False
                
            if verbose and not accepted:
                logger.info("Rejected!")
                
            if accepted:
                MS.means.transfer_from(MS.p, param_info)
                MS.H.accept[k] = 1
                #MS.H.ratio[k] = 10 ** logratio
                
            update_history(MS.H, k, MS.p, MS.means, param_info)
        except KeyboardInterrupt:
            logger.info("Terminating with k={} iters completed:".format(k-1))
            MS.H.truncate(k, param_info)
            break
        
        if checkpoint_freq is not None and k % checkpoint_freq == 0:
            chpt_fname = os.path.join(sim_flags["checkpoint_dirname"], f"checkpoint_{k}.pik")
            logger.info(f"Saving checkpoint at k={k}; fname {chpt_fname}")
            MS.random_state = np.random.get_state()
            MS.checkpoint(chpt_fname)
        
    MS.H.apply_unit_conversions(param_info)
    MS.H.final_cov = MS.variances.cov
    return MS.H
