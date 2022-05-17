# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 22:13:26 2022

@author: cfai2
"""
import numpy as np
from scipy.integrate import solve_ivp
from multiprocessing import Pool
import os
import sys
from functools import partial
import logging

from forward_solver import dydt
from sim_utils import Grid, Solution, Parameters, History, HistoryList, Covariance

## Constants
eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
q = 1.0 # [e]
q_C = 1.602e-19 # [C]
kB = 8.61773e-5  # [eV / K]

def E_field(N, P, PA, dx, corner_E=0):
    if N.ndim == 1:
        E = corner_E + q_C / (PA.eps * eps0) * dx * np.cumsum(((P - PA.p0) - (N - PA.n0)))
        E = np.concatenate(([corner_E], E))
    elif N.ndim == 2:
        E = corner_E + q_C / (PA.eps * eps0) * dx * np.cumsum(((P - PA.p0) - (N - PA.n0)), axis=1)
        num_tsteps = len(N)
        E = np.concatenate((np.ones(shape=(num_tsteps,1))*corner_E, E), axis=1)
    return E

def model(init_dN, g, p):
    N = init_dN + p.n0
    P = init_dN + p.p0
    E_f = E_field(N, P, p, g.dx)
    
    init_condition = np.concatenate([N, P, E_f], axis=None)
    args = (g,p)
    sol = solve_ivp(dydt, [g.start_time,g.time], init_condition, args=args, t_eval=g.tSteps, method='BDF', max_step=g.hmax, rtol=1e-5, atol=1e-8)
    data = sol.y.T
    s = Solution()
    s.N, s.P, E_f = np.split(data, [g.nx, 2*g.nx], axis=1)
    s.calculate_PL(g, p)
    return s.PL, (s.N[-1]-p.n0)

def draw_initial_guesses(initial_guesses, num_initial_guesses):
    initial_guess_list = []
    param_is_iterable = {param:isinstance(initial_guesses[param], (list, tuple, np.ndarray)) for param in initial_guesses}
    
    for ig in range(num_initial_guesses):
        initial_guess = {}
        for param in initial_guesses:
            if param_is_iterable[param]:
                initial_guess[param] = initial_guesses[param][ig]
            else:
                initial_guess[param] = initial_guesses[param]
                        
        initial_guess_list.append(initial_guess)
    return initial_guess_list

def select_from_box(p, means, variances, param_info, picked_param, logger):
    is_active = param_info["active"]
    do_log = param_info["do_log"]
    names = param_info["names"]
    mean = means.asarray(param_info)
    for i, param in enumerate(names):        
        if do_log[param]:
            mean[i] = np.log10(mean[i])
            
    cov = variances.cov
    new_p = np.zeros_like(mean)
    for i, param in enumerate(names):
        new_p[i] = np.random.uniform(mean[i] - cov[i,i], mean[i] + cov[i,i])
        
    for i, param in enumerate(names):
        if is_active[param]:
            if do_log[param]:
                setattr(p, param, 10 ** new_p[i])
            else:
                setattr(p, param, new_p[i])
    return
    

def select_next_params(p, means, variances, param_info, picked_param, logger):
    is_active = param_info["active"]
    do_log = param_info["do_log"]
    names = param_info["names"]
    mean = means.asarray(param_info)
    for i, param in enumerate(names):        
        if do_log[param]:
            mean[i] = np.log10(mean[i])
            
    cov = variances.cov

    try:
        new_p = np.random.multivariate_normal(mean, cov)
    except Exception:
        logger.error("multivar_norm failed: mean {}, cov {}".format(mean, cov))
        new_p = mean

    # for i, param in enumerate(names):
    #     if is_active[param]:
    #         if picked_param is None or i == picked_param[1]:
    #             if do_log[param]:
    #                 setattr(p, param, 10 ** new_p[i])
    #             else:
    #                 setattr(p, param, new_p[i])
    #         else:
    #             setattr(p, param, getattr(means, param))

    for i, param in enumerate(names):
        if is_active[param]:
            if do_log[param]:
                setattr(p, param, 10 ** new_p[i])
            else:
                setattr(p, param, new_p[i])

    return

def update_covariance_AP(p, variances, history, param_info, picked_param, t, last_r, LAP_params=(1,0.8,0.234)):
    do_log = param_info["do_log"]
    names = param_info["names"]
    is_active = param_info["active"]
    num_actives = 0
    for param, active in is_active.items():
        if active:
            num_actives += 1
    a = LAP_params[1]
    b = LAP_params[0]
    r_opt = LAP_params[2]
    maximum = 3

    new_X = p.asarray(param_info)[:, None]
    for i, param in enumerate(names):
        if do_log[param]:
            new_X[i] = np.log10(new_X[i])

    if not hasattr(variances, "X_all"):
        # First time initialization
        variances.prev_X_all = history.get_XT_AM(param_info, t-2)
        variances.prev_X_all = np.mean(variances.prev_X_all, axis=1)[:, None]
        variances.X_all = history.get_XT_AM(param_info, t-1)
        variances.X_all = np.mean(variances.X_all, axis=1)[:, None]

        variances.ID = np.zeros((len(new_X), len(new_X)))
        for i, param in enumerate(names):  
            if is_active[param]:
                variances.ID[i,i] = 1
        variances.r_nk = np.sum(history.accept[:t+1]) / t
    else:
        variances.prev_X_all = variances.X_all
        variances.X_all += (new_X - variances.X_all) / (t-1)
        variances.r_nk += (history.accept[t] - variances.r_nk) / (t-1)

    dx = variances.prev_X_all - new_X
    Sigma = np.matmul(dx, dx.T)

    gamma_1 = np.power(t, -a)
    gamma_2 = b * gamma_1
    if picked_param is None:
        variances.little_sigma *= np.exp(gamma_2*(last_r - r_opt))
        variances.big_sigma += gamma_1 * (Sigma - variances.big_sigma)
        variances.cov = variances.little_sigma * variances.big_sigma
    else:
        i = picked_param[1]
        variances.little_sigma[i] *= np.exp(gamma_2*(last_r - r_opt))
        variances.big_sigma += gamma_1 * (Sigma - variances.big_sigma)
        variances.cov = np.zeros_like(variances.cov)
        variances.cov[i,i] = variances.little_sigma[i] * variances.big_sigma[i,i]


    for i in range(len(variances.cov)):
        if variances.cov[i,i] > maximum: variances.cov[i,i] = maximum

    return

def update_covariance_AM(p, variances, history, param_info, picked_param, t, eps=1e-5):
    do_log = param_info["do_log"]
    names = param_info["names"]
    is_active = param_info["active"]
    num_actives = 0
    for param, active in param_info['active'].items():
        if active:
            num_actives += 1
    C = 2.4 ** 2 / num_actives
    new_X = p.asarray(param_info)[:, None]
    
    for i, param in enumerate(names):        
        if do_log[param]:
            new_X[i] = np.log10(new_X[i])
        
    
    if not hasattr(variances, "X_all"):
        # First time initialization
        variances.prev_X_all = history.get_XT_AM(param_info, t-2)
        variances.prev_X_all = np.mean(variances.prev_X_all, axis=1)[:, None]
        variances.X_all = history.get_XT_AM(param_info, t-1)
        variances.X_all = np.mean(variances.X_all, axis=1)[:, None]
        
        variances.ID = np.zeros((len(new_X), len(new_X)))
        for i, param in enumerate(names):  
            if is_active[param]:
                variances.ID[i,i] = 1
    else:
        variances.prev_X_all = variances.X_all
        variances.X_all += (new_X - variances.X_all) / (t-1)

    # Contributions
    # base = (t-2)/(t-1)*variances.cov
    # prev = (t-1) * np.matmul(variances.prev_X_all, variances.prev_X_all.T)
    # current = (t) * np.matmul(variances.X_all, variances.X_all.T)
    # new = np.matmul(new_X, new_X.T)
    minimum = eps * variances.ID
    maximum = 3
    # variances.cov = base + (C/(t-1)) * (prev - current + new + minimum)

    dx = variances.prev_X_all - new_X
    base = (t-2)/(t-1)*variances.big_sigma
    variances.big_sigma = base + (1/t) * (np.matmul(dx, dx.T) + minimum)

    if picked_param is None:
        variances.cov = variances.little_sigma * variances.big_sigma
    else:
        i = picked_param[1]
        variances.cov = np.zeros_like(variances.cov)
        variances.cov[i,i] = variances.little_sigma[i] * variances.big_sigma[i,i]

    for i in range(len(variances.cov)):
        if variances.cov[i,i] > maximum: variances.cov[i,i] = maximum
    return

def update_covariance(variances, picked_param):
    # Induce a univariate gaussian if doing one-param-at-a-time
    if picked_param is None:
        variances.cov = variances.little_sigma * variances.big_sigma
    else:
        i = picked_param[1]
        variances.cov = np.zeros_like(variances.cov)
        variances.cov[i,i] = variances.little_sigma[i] * variances.big_sigma[i,i]

def update_covariance_RAM(p_just_prop, means, variances, history, param_info, picked_param, t, LAP_params=(1,0.8,0.234)):
    names = param_info["names"]
    is_active = param_info["active"]
    sn = np.linalg.cholesky(variances.big_sigma)
    a = LAP_params[1]
    b = LAP_params[0]
    r_opt = LAP_params[2]
    gamma = b * np.power(t, -a)

    new_X = p_just_prop.asarray(param_info)[:, None]
    mean = means.asarray(param_info)[:,None]
    un = np.matmul(np.linalg.inv(variances.big_sigma), new_X - mean)

    if not hasattr(variances, "ID"):
        # First time initialization
        variances.ID = np.zeros((len(new_X), len(new_X)))
        for i, param in enumerate(names):  
            if is_active[param]:
                variances.ID[i,i] = 1

        variances.r_nk = np.sum(history.accept[:t+1]) / t
    else:
        variances.r_nk += (history.accept[t] - variances.r_nk) / (t-1)

    adapt_factor = variances.ID + gamma * (variances.r_nk - r_opt) * np.matmul(un, un.T) / np.linalg.norm(un)**2
    variances.big_sigma = np.matmul(np.matmul(sn, adapt_factor), sn.T)

    if picked_param is None:
        variances.cov = variances.little_sigma * variances.big_sigma
    else:
        raise NotImplementedError("Adaptive RAM with 1AAT WIP")
    return

def update_means(p, means, param_info):
    for param in param_info['names']:
        setattr(means, param, getattr(p, param))
    return

def print_status(p, means, param_info, logger):
    is_active = param_info['active']
    ucs = param_info["unit_conversions"]
    for param in param_info['names']:
        if is_active.get(param, 0):
            logger.info("Next {}: {:.3e} from mean {:.3e}".format(param, getattr(p, param) / ucs.get(param, 1), getattr(means, param) / ucs.get(param, 1)))
            
    return

def update_history(H, k, p, means, param_info):
    for param in param_info['names']:
        h = getattr(H, param)
        h[k] = getattr(p, param)
        h_mean = getattr(H, f"mean_{param}")
        h_mean[k] = getattr(means, param)
    return

def do_simulation(p, thickness, nx, iniPar, times):
    g = Grid()
    g.thickness = thickness
    g.nx = nx
    g.dx = g.thickness / g.nx
    g.xSteps = np.linspace(g.dx / 2, g.thickness - g.dx/2, g.nx)
    
    g.time = times[-1]
    g.start_time = times[0]
    g.nt = len(times) - 1
    g.dt = g.time / g.nt
    g.hmax = 4
    g.tSteps = times
    
    sol, next_init_condition = model(iniPar, g, p)
    return sol, next_init_condition
    
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

def convert_DA_times(DA_time_subs, total_len):
    # DA_time_subs: if int, treat as number of equally spaced time subs
    # If list, treat as percentages over which to subdivide data
    # e.g. [10,30,60] splits data into 0-10%, 10-30%, 30-60%, and 60-100%
    if isinstance(DA_time_subs, list):
        splits = np.array(DA_time_subs) / 100 * total_len
        splits = splits.astype(int)
    else:
        splits = DA_time_subs
    return splits

def subdivide(y, splits):
    # Split 'y' into 'splits' groups, then transfers the end of each to the beginning of the next
    if isinstance(splits, list):
        num_time_subs = len(splits) + 1
    else:
        num_time_subs = splits
    y = np.split(y, splits)
    y[0] = np.insert(y[0], 0, 0)
    for j in range(1, num_time_subs):
        y[j] = np.insert(y[j], 0, y[j-1][-1])
    return y

def init_param_managers(param_info, initial_guess, initial_variance, num_iters):
    # Initializes a current parameters, previous parameters, 
    # history, means, and variances objects
    p = Parameters(param_info, initial_guess)
    p.apply_unit_conversions(param_info)
    
    H = History(num_iters, param_info)
    
    prev_p = Parameters(param_info, initial_guess)
    prev_p.apply_unit_conversions(param_info)
    
    means = Parameters(param_info, initial_guess)
    means.apply_unit_conversions(param_info)
    
    variances = Covariance(param_info)
    
    
    for param in param_info['names']:
        if param_info['active'][param]:
            variances.set_variance(param, initial_variance)
            
    iv_arr = 0
    if isinstance(initial_variance, dict):
        iv_arr = np.ones(len(variances.cov))
        for i, param in enumerate(param_info['names']):
            if param_info['active'][param]:
                iv_arr[i] = initial_variance[param]
    
    elif isinstance(initial_variance, (float, int)):
        iv_arr = initial_variance
            
    variances.little_sigma = np.ones(len(variances.cov)) * iv_arr
    variances.big_sigma = variances.cov * iv_arr**-1
    return p, prev_p, H, means, variances
    
def run_DA_iteration(p, simPar, iniPar, DA_time_subs, num_time_subs, times, vals, tf, verbose, logger, prev_p=None):
    # Calculates likelihood of a new proposed parameter set
    accepted = True
    logratio = 0 # acceptance ratio = 1
    cu_logratio = 0
    p.likelihood = np.zeros((len(iniPar), num_time_subs))
    for i in range(len(iniPar)):
        thickness, nx = unpack_simpar(simPar, i)
        splits = convert_DA_times(DA_time_subs, len(times[i][1:]))
        
        subdivided_times = subdivide(times[i][1:], splits)
        subdivided_vals = subdivide(vals[i][1:], splits)
        
        next_init_condition = iniPar[i]
        for j in range(num_time_subs):
            sol, next_init_condition = do_simulation(p, thickness, nx, next_init_condition, subdivided_times[j])
            try:
                if len(sol) < len(subdivided_vals[j]):
                    sol2 = np.ones_like(subdivided_vals[j]) * sys.float_info.min
                    sol2[:len(sol)] = sol
                    sol = np.array(sol)
                    logger.warning(f"{i}: Simulation terminated early!")
                if np.any(sol < 0):
                    sol = np.abs(sol + sys.float_info.min)
                    logger.warning(f"{i}: Carriers depleted!")
                p.likelihood[i, j] -= np.sum((np.log10(sol[1:]) - subdivided_vals[j][1:])**2)
                # TRPL must be positive! Any simulation which results in depleted carrier is clearly incorrect
                if np.isnan(p.likelihood[i,j]): raise ValueError(f"{i}: Simulation failed!")
            except ValueError as e:
                logger.warning(e)
                p.likelihood[i,j] = -np.inf
                #fail_i = 0
                #while os.path.exists(os.path.join("Fails", f"fail_{fail_i}.npy")):
                #    fail_i += 1
                #np.save(os.path.join("Fails", f"fail_{fail_i}.npy"), p.asarray())
                #logger.error("Simulation failed! Wrote to {}".format(os.path.join("Fails", f"fail_{fail_i}.npy")))

            p.likelihood[i, j] /= tf
            
            if prev_p is not None:
                logratio = p.likelihood[i, j] - prev_p.likelihood[i, j]
                
                if np.isnan(logratio): logratio = -np.inf
                
                cu_logratio += min(0, logratio)
                if verbose: 
                    logger.info("Partial Ratio: {}, Cu: {}".format(10 ** logratio, 10 ** cu_logratio))
                    
                
                accepted = roll_acceptance(logratio)
                if not accepted:
                    return accepted, min(1, 10 ** cu_logratio)

    if prev_p is not None and accepted:
        prev_p.likelihood = p.likelihood
    return accepted, min(1, 10 ** cu_logratio)


def start_metro_controller(simPar, iniPar, e_data, sim_flags, param_info, initial_guess_list, initial_variance, logger):
    #num_cpus = 2
    num_cpus = min(os.cpu_count(), sim_flags["num_initial_guesses"])
    logger.info(f"{num_cpus} CPUs marshalled")
    logger.info(f"{len(initial_guess_list)} MC chains needed")
    with Pool(num_cpus) as pool:
        histories = pool.map(partial(metro, simPar, iniPar, e_data, sim_flags, param_info, initial_variance, False, logger), initial_guess_list)
        
    history_list = HistoryList(histories, param_info)
    
    return history_list
        
def metro(simPar, iniPar, e_data, sim_flags, param_info, initial_variance, verbose, logger, initial_guess):
    # Setup
    #np.random.seed(42)
    
    num_iters = sim_flags["num_iters"]
    DA_mode = sim_flags["delayed_acceptance"]
    DA_time_subs = sim_flags["DA time subdivisions"]
    
    if isinstance(DA_time_subs, list):
        num_time_subs = len(DA_time_subs)+1
    else:
        num_time_subs = DA_time_subs
    
    times, vals, uncs = e_data    
    tf = sum([len(time)-1 for time in times]) * sim_flags["tf"]
    
    p, prev_p, H, means, variances = init_param_managers(param_info, initial_guess, initial_variance, num_iters)
    
    # Calculate likelihood of initial guess
    run_DA_iteration(prev_p, simPar, iniPar, DA_time_subs, num_time_subs, times, vals, tf, verbose, logger)
    last_r = sim_flags["LAP_params"][2]
    
    if DA_mode == 'on':
        pass
    elif DA_mode == 'off':
        prev_p.likelihood = np.sum(prev_p.likelihood)
    elif DA_mode == 'cumulative':
        rand_i = np.arange(len(iniPar))
        np.random.shuffle(rand_i)
        prev_p.cumulikelihood = np.cumsum(prev_p.likelihood[rand_i])
    
    update_history(H, 0, prev_p, means, param_info)

    for k in range(1, num_iters):
        try:
            # Identify which parameter to move
            if sim_flags.get("one_param_at_a_time", 0):
                picked_param = means.actives[k % len(means.actives)]
            else:
                picked_param = None
                
            # Select next sample from distribution
            if (k > sim_flags.get("AM_activation_time", np.inf) and 
                  sim_flags.get("adaptive_covariance", "None") == "LAP"):
                update_covariance_AP(means, variances, H, param_info, picked_param, k, last_r, sim_flags["LAP_params"])
                if verbose: logger.info("New covariance: {}".format(variances.trace()))

            elif (k > sim_flags.get("AM_activation_time", np.inf) and 
                  sim_flags.get("adaptive_covariance", "None") == "AM"):
                update_covariance_AM(means, variances, H, param_info, picked_param, k)
                if verbose: logger.info("New covariance: {}".format(variances.trace()))

            elif sim_flags.get("adaptive_covariance", "None") == "None":
                update_covariance(variances, picked_param)

            if sim_flags["proposal_function"] == "gauss":
                select_next_params(p, means, variances, param_info, picked_param, logger)
            elif sim_flags["proposal_function"] == "box":
                select_from_box(p, means, variances, param_info, picked_param, logger)

            if verbose: print_status(p, means, param_info, logger)
            else: logger.info(f"Iter {k}")
            # Calculate new likelihood?

            if DA_mode == "on":
                accepted, last_r = run_DA_iteration(p, simPar, iniPar, DA_time_subs, num_time_subs,
                                            times, vals, tf, verbose, logger, prev_p)

            elif DA_mode == "cumulative":
                p.likelihood = np.zeros(len(iniPar))
                p.cumulikelihood = np.zeros(len(iniPar))
                for i in range(len(iniPar)):
                    thickness, nx = unpack_simpar(simPar, rand_i[i])
                    sol, next_init_condition = do_simulation(p, thickness, nx, iniPar[rand_i[i]], times[rand_i[i]])
                    p.likelihood[rand_i[i]] -= np.sum((np.log10(sol) - vals[rand_i[i]])**2)
                    p.likelihood[rand_i[i]] /= tf
                
                    # Compare with prior likelihood
                    if i > 0: p.cumulikelihood[i] = p.cumulikelihood[i-1] + p.likelihood[rand_i[i]]
                    else: p.cumulikelihood[i] = p.likelihood[rand_i[i]]
                    logratio = p.cumulikelihood[i] - prev_p.cumulikelihood[i]
                    logger.info("Partial Ratio: {}".format(10 ** logratio))
                
                    accepted = roll_acceptance(logratio)
                    if not accepted:
                        break
                    
                np.random.shuffle(rand_i)
                if accepted:
                    prev_p.likelihood = p.likelihood
                    
                prev_p.cumulikelihood = np.cumsum(prev_p.likelihood[rand_i])
                    
            elif DA_mode == "off":
                p.likelihood = np.zeros(len(iniPar))
                for i in range(len(iniPar)):
                    thickness, nx = unpack_simpar(simPar, i)
                    sol, next_init_condition = do_simulation(p, thickness, nx, iniPar[i], times[i])
                    try:
                        logger.info("Simulation complete; final t {}".format(times[i][len(sol)-1]))
                        if len(sol) < len(vals[i]):
                            sol2 = np.ones_like(vals[i]) * sys.float_info.min
                            sol2[0:len(sol)] = sol
                            sol = np.array(sol2)
                            logger.warning(f"{i}: Simulation stopped early!")

                        if np.any(sol < 0):
                            sol = np.abs(sol + sys.float_info.min)
                            logger.warning(f"{i}: Carriers depleted!")
                        p.likelihood[i] -= np.sum((np.log10(sol) - vals[i])**2)
                        if np.isnan(p.likelihood[i]): raise ValueError(f"{i}: Simulation failed!")
                    except ValueError as e:
                        logger.warning(e)
                        p.likelihood[i] = -np.inf

                    p.likelihood[i] /= tf
                
                # Compare with prior likelihood
                p.likelihood = np.sum(p.likelihood)
                logratio = p.likelihood - prev_p.likelihood
                if np.isnan(logratio): 
                    logratio = -np.inf
                    logger.warning("Invalid logratio; autorejecting")

                logger.info("Partial Ratio: {}".format(10 ** logratio))
            
                accepted = roll_acceptance(logratio)
                
                if accepted:
                    prev_p.likelihood = p.likelihood
                
            elif DA_mode == 'DEBUG':
                accepted = False
                
            if verbose and not accepted:
                logger.info("Rejected!")
                
            logger.info("Iter {}".format(k))
            logger.info("#####")
            if accepted:
                update_means(p, means, param_info)
                H.accept[k] = 1
                
                #H.ratio[k] = 10 ** logratio
                
            update_history(H, k, p, means, param_info)
        except KeyboardInterrupt:
            logger.info("Terminating with k={} iters completed:".format(k-1))
            H.truncate(k, param_info)
            break
        
    H.apply_unit_conversions(param_info)
    H.final_cov = variances.cov
    return H
