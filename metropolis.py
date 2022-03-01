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
from sim_utils import Grid, Solution, Parameters, History, HistoryList

## Constants
eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
q = 1.0 # [e]
q_C = 1.602e-19 # [C]
kB = 8.61773e-5  # [eV / K]

def E_field(N, P, PA, dx):
    if N.ndim == 1:
        corner_E = 0
        E = corner_E + q_C / (PA.eps * eps0) * dx * np.cumsum(((P - PA.p0) - (N - PA.n0)))
        E = np.concatenate(([corner_E], E))
    elif N.ndim == 2:
        corner_E = 0
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

def select_next_params(p, means, variances, param_info):
    is_active = param_info["active"]
    do_log = param_info["do_log"]
    for param in param_info["names"]:
        if is_active.get(param, 0):
            mean = getattr(means, param)
            var = getattr(variances, param)
            
            if do_log.get(param, 0):    
                setattr(p, param, 10 ** np.random.normal(loc=np.log10(mean), scale=var))
                
            else:
                setattr(p, param, np.random.normal(loc=mean, scale=var))
                while getattr(p, param) <= 0:
                    # Negative values not allowed - choose again
                    setattr(p, param, np.random.normal(loc=mean, scale=var))
        
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

def start_metro_controller(simPar, iniPar, e_data, sim_flags, param_infos):
    #num_cpus = 2
    num_cpus = os.cpu_count()
    print(f"{num_cpus} CPUs detected")
    print(f"{len(param_infos)} MC chains needed")
    with Pool(num_cpus) as pool:
        histories = pool.map(partial(metro, simPar, iniPar, e_data, sim_flags), param_infos)
        
    history_list = HistoryList(histories, param_infos)
    
    return history_list
        
    

def metro(simPar, iniPar, e_data, sim_flags, param_info):
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
    tf = sum([len(time)-1 for time in times]) * (1/2000)
    p = Parameters(param_info)
    p.apply_unit_conversions(param_info)
    
    H = History(num_iters, param_info)
    
    prev_p = Parameters(param_info)
    prev_p.apply_unit_conversions(param_info)
    
    means = Parameters(param_info)
    means.apply_unit_conversions(param_info)
    
    variances = Parameters(param_info)
    variances.mu_n = 5
    variances.mu_p = 5
    variances.apply_unit_conversions(param_info)
    variances.B = 0.1
    variances.p0 = 0.1
    variances.Sf = 0.1
    variances.Sb = 0.1
    variances.tauN = 20
    variances.tauP = 20
    
    # Calculate likelihood of initial guess
    prev_p.likelihood = np.zeros((len(iniPar), num_time_subs))
    for i in range(len(iniPar)):
        thickness, nx = unpack_simpar(simPar, i)
        splits = convert_DA_times(DA_time_subs, len(times[i][1:]))
        
        subdivided_times = np.split(times[i][1:], splits)
        subdivided_times[0] = np.insert(subdivided_times[0], 0, 0)
        for j in range(1, num_time_subs):
            subdivided_times[j] = np.insert(subdivided_times[j], 0, subdivided_times[j-1][-1])
            
        subdivided_vals = np.split(vals[i][1:], splits)
        subdivided_vals[0] = np.insert(subdivided_vals[0], 0, 0)
        for j in range(1, num_time_subs):
            subdivided_vals[j] = np.insert(subdivided_vals[j], 0, subdivided_vals[j-1][-1])
            
        next_init_condition = iniPar[i]
        for j in range(num_time_subs):
            sol, next_init_condition = do_simulation(prev_p, thickness, nx, next_init_condition, subdivided_times[j])
            prev_p.likelihood[i, j] -= np.sum((np.log10(sol[1:]) - subdivided_vals[j][1:])**2)

    prev_p.likelihood /= tf
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
                p.likelihood = np.zeros((len(iniPar), num_time_subs))
                for i in range(len(iniPar)):
                    thickness, nx = unpack_simpar(simPar, i)
                    splits = convert_DA_times(DA_time_subs, len(times[i][1:]))
                    
                    subdivided_times = np.split(times[i][1:], splits)
                    subdivided_times[0] = np.insert(subdivided_times[0], 0, 0)
                    for j in range(1, num_time_subs):
                        subdivided_times[j] = np.insert(subdivided_times[j], 0, subdivided_times[j-1][-1])
                        
                    subdivided_vals = np.split(vals[i][1:], splits)
                    subdivided_vals[0] = np.insert(subdivided_vals[0], 0, 0)
                    for j in range(1, num_time_subs):
                        subdivided_vals[j] = np.insert(subdivided_vals[j], 0, subdivided_vals[j-1][-1])
                        
                    next_init_condition = iniPar[i]
                    for j in range(num_time_subs):
                        sol, next_init_condition = do_simulation(p, thickness, nx, next_init_condition, subdivided_times[j])
                        p.likelihood[i, j] -= np.sum((np.log10(sol[1:]) - subdivided_vals[j][1:])**2)
                        p.likelihood[i, j] /= tf
                
                    # Compare with prior likelihood
                        logratio = p.likelihood[i, j] - prev_p.likelihood[i, j]
                        print("Partial Ratio: {}".format(10 ** logratio))
                    
                        accepted = roll_acceptance(logratio)
                        if not accepted:
                            break
                    if not accepted:
                        break
                    
                if accepted:
                    prev_p.likelihood = p.likelihood
                    
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