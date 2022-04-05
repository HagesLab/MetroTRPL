# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 22:13:26 2022

@author: cfai2
"""
import numpy as np
from scipy.integrate import solve_ivp
from multiprocessing import Pool
import os
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
    sol = solve_ivp(dydt, [g.start_time,g.time], init_condition, args=args, t_eval=g.tSteps, method='BDF', max_step=g.hmax)
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

def select_next_params(p, means, variances, param_info):
    is_active = param_info["active"]
    do_log = param_info["do_log"]
    names = param_info["names"]
    mean = means.asarray(param_info)
    for i, param in enumerate(names):        
        if do_log[param]:
            mean[i] = np.log10(mean[i])
            
    cov = variances.cov
    new_p = np.random.multivariate_normal(mean, cov)
    
    for i, param in enumerate(names):
        if is_active[param]:
            if do_log[param]:
                setattr(p, param, 10 ** new_p[i])
            else:
                setattr(p, param, new_p[i])
        
    return

def update_means(p, means, param_info):
    for param in param_info['names']:
        setattr(means, param, getattr(p, param))
    return

def print_status(p, means, param_info):
    is_active = param_info['active']
    ucs = param_info["unit_conversions"]
    for param in param_info['names']:
        if is_active.get(param, 0):
            print("Next {}: {} from mean {}".format(param, getattr(p, param) / ucs.get(param, 1), getattr(means, param) / ucs.get(param, 1)))
            
    return

def update_history(H, k, p, means, param_info):
    for param in param_info['names']:
        h = getattr(H, param)
        h[k] = getattr(p, param)
        h_mean = getattr(H, f"mean_{param}")
        h_mean[k] = getattr(means, param)
            
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

def init_param_managers(param_info, initial_guess, initial_variances, num_iters):
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
            variances.set_variance(param, initial_variances[param])
    
    return p, prev_p, H, means, variances
    
def run_DA_iteration(p, simPar, iniPar, DA_time_subs, num_time_subs, times, vals, tf, prev_p=None):
    # Calculates likelihood of a new proposed parameter set
    accepted = True
    p.likelihood = np.zeros((len(iniPar), num_time_subs))
    for i in range(len(iniPar)):
        thickness, nx = unpack_simpar(simPar, i)
        splits = convert_DA_times(DA_time_subs, len(times[i][1:]))
        
        subdivided_times = subdivide(times[i][1:], splits)
        subdivided_vals = subdivide(vals[i][1:], splits)
        
        next_init_condition = iniPar[i]
        for j in range(num_time_subs):
            sol, next_init_condition = do_simulation(p, thickness, nx, next_init_condition, subdivided_times[j])
            p.likelihood[i, j] -= np.sum((np.log10(sol[1:]) - subdivided_vals[j][1:])**2)
            p.likelihood[i, j] /= tf
            
            if prev_p is not None:
                logratio = p.likelihood[i, j] - prev_p.likelihood[i, j]
                print("Partial Ratio: {}".format(10 ** logratio))
                
                accepted = roll_acceptance(logratio)
                if not accepted:
                    return

    if prev_p is not None and accepted:
        prev_p.likelihood = p.likelihood
    return accepted

def start_metro_controller(simPar, iniPar, e_data, sim_flags, param_info, initial_guess_list, initial_variances):
    #num_cpus = 2
    num_cpus = min(os.cpu_count(), sim_flags["num_initial_guesses"])
    print(f"{num_cpus} CPUs marshalled")
    print(f"{len(initial_guess_list)} MC chains needed")
    with Pool(num_cpus) as pool:
        histories = pool.map(partial(metro, simPar, iniPar, e_data, sim_flags, param_info, initial_variances), initial_guess_list)
        
    history_list = HistoryList(histories, param_info)
    
    return history_list
        
def metro(simPar, iniPar, e_data, sim_flags, param_info, initial_variances, initial_guess):
    # Setup
    np.random.seed(42)
    
    num_iters = sim_flags["num_iters"]
    DA_mode = sim_flags["delayed_acceptance"]
    DA_time_subs = sim_flags["DA time subdivisions"]
    
    if isinstance(DA_time_subs, list):
        num_time_subs = len(DA_time_subs)+1
    else:
        num_time_subs = DA_time_subs
    
    times, vals, uncs = e_data    
    tf = sum([len(time)-1 for time in times]) * (1/2500)
    
    p, prev_p, H, means, variances = init_param_managers(param_info, initial_guess, initial_variances, num_iters)
    
    # Calculate likelihood of initial guess
    run_DA_iteration(prev_p, simPar, iniPar, DA_time_subs, num_time_subs, times, vals, tf)
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
            # Select next sample from distribution
            
            select_next_params(p, means, variances, param_info)
    
            print_status(p, means, param_info)
            # Calculate new likelihood?
            
            if DA_mode == "on":
                accepted = run_DA_iteration(p, simPar, iniPar, DA_time_subs, num_time_subs,
                                            times, vals, tf, prev_p)
                
                
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
                    print("Partial Ratio: {}".format(10 ** logratio))
                
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
                    p.likelihood[i] -= np.sum((np.log10(sol) - vals[i])**2)
                    p.likelihood[i] /= tf
                
                # Compare with prior likelihood
                p.likelihood = np.sum(p.likelihood)
                logratio = p.likelihood - prev_p.likelihood
            
                print("Partial Ratio: {}".format(10 ** logratio))
            
                accepted = roll_acceptance(logratio)
                
                if accepted:
                    prev_p.likelihood = p.likelihood
                
            if not accepted:
                print("Rejected!")
                
            print("Iter {}".format(k))
            print("#####")
            if accepted:
                update_means(p, means, param_info)
                H.accept[k] = 1
                
                #H.ratio[k] = 10 ** logratio
                
            update_history(H, k, p, means, param_info)
        except KeyboardInterrupt:
            print("Terminating with k={} iters completed:".format(k-1))
            H.truncate(k, param_info)
            break
        
    H.apply_unit_conversions(param_info)
    return H
