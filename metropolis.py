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
import signal
from functools import partial

from forward_solver import dydt
from sim_utils import MetroState, Grid, Solution
import pickle

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
def t_rad(B, p0):
    # B [nm^3 / ns]
    # p0 [nm^-3]
    if B == 0 or p0 == 0:
        return np.inf
    
    else:
        return 1 / (B*p0)

def t_auger(CP, p0):
    if CP == 0 or p0 == 0:
        return np.inf
    
    else:
        return 1 / (CP*p0**2)

def LI_tau_eff(B, p0, tau_n, Sf, Sb, CP, thickness, mu):
    # S [nm/ns]
    # B [nm^3 / ns]
    # p0 [nm^-3]
    # tau_n [ns]
    # thickness [nm]
    kb = 0.0257 #[ev]
    q = 1
    
    D = mu * kb / q # [nm^2/ns]
    if Sf+Sb == 0 or D == 0:
        tau_surf = np.inf
    else:
        tau_surf = (thickness / ((Sf+Sb))) + (thickness**2 / (np.pi ** 2 * D))
    t_r = t_rad(B, p0)
    t_aug = t_auger(CP, p0)
    return (t_r**-1 + t_aug**-1 + tau_surf**-1 + tau_n**-1)**-1

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

def check_approved_param(new_p, param_info):
    #return True
    order = param_info['names']
    ucs = param_info.get('unit_conversions', {})
    # mu_n and mu_p between 1e-6 and 1e6; a reasonable range for most materials
    if 'mu_n' in order:
        mu_n_size = (np.abs(new_p[order.index('mu_n')] - np.log10(ucs.get('mu_n', 1))) < 6)
    else:
        mu_n_size = True

    if 'mu_p' in order:
        mu_p_size = (np.abs(new_p[order.index('mu_p')] - np.log10(ucs.get('mu_p', 1))) < 6)
    else:
        mu_p_size = True

    if 'Sf' in order:
        sf_size = (np.abs(new_p[order.index('Sf')] - np.log10(ucs.get('Sf', 1))) < 7)
    else:
        sf_size = True

    if 'Sb' in order:
        sb_size = (np.abs(new_p[order.index('Sb')] - np.log10(ucs.get('Sb', 1))) < 7)
    else:
        sb_size = True

    # tau_n and tau_p must be *close* (within 3 OM) for a reasonable midgap SRH
    if 'tauN' in order and 'tauP' in order:
        tn_tp_constraint = (np.abs(new_p[order.index('tauN')] - new_p[order.index('tauP')]) <= 3)
    else:
        tn_tp_constraint = True

    return tn_tp_constraint and mu_n_size and mu_p_size and sf_size and sb_size

def select_from_box(p, means, variances, param_info, logger=None):
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
        new_p = np.zeros_like(mean)
        for i, param in enumerate(names):
            new_p[i] = np.random.uniform(mean[i] - cov[i,i], mean[i] + cov[i,i])
        
        approved = check_approved_param(new_p, param_info)
        attempts += 1
        if attempts > 100: break
    
    if logger is not None:
        logger.info(f"Found suitable parameters in {attempts} attempts")
        
    for i, param in enumerate(names):
        if is_active[param]:
            if do_log[param]:
                setattr(p, param, 10 ** new_p[i])
            else:
                setattr(p, param, new_p[i])
    return
    


def select_next_params(p, means, variances, param_info, logger=None):
    is_active = param_info["active"]
    do_log = param_info["do_log"]
    names = param_info["names"]
    mean = means.to_array(param_info)
    for i, param in enumerate(names):        
        if do_log[param]:
            mean[i] = np.log10(mean[i])
            
    cov = variances.cov

    try:
        assert np.all(cov >= 0)
        new_p = np.random.multivariate_normal(mean, cov)
    except Exception:
        if logger is not None:
            logger.error("multivar_norm failed: mean {}, cov {}".format(mean, cov))
        new_p = mean

    for i, param in enumerate(names):
        if is_active[param]:
            if do_log[param]:
                setattr(p, param, 10 ** new_p[i])
            else:
                setattr(p, param, new_p[i])

    return

def update_covariance(variances, picked_param):
    # Induce a univariate gaussian if doing one-param-at-a-time
    if picked_param is None:
        variances.cov = variances.little_sigma * variances.big_sigma
    else:
        i = picked_param[1]
        variances.cov = np.zeros_like(variances.cov)
        variances.cov[i,i] = variances.little_sigma[i] * variances.big_sigma[i,i]

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

def det_hmax(g, p):
    teff = LI_tau_eff(p.ks, p.p0, p.tauN, p.Sf, p.Sb, 1e-99, g.thickness, p.mu_n)
    if teff < g.time / 100:
        g.hmax = g.nt
    else:
    	g.hmax = 4
        
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
    
    det_hmax(g, p)
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

def anneal(t, anneal_mode, anneal_params):
    if anneal_mode is None or anneal_mode == "None":
        return anneal_params[0]

    elif anneal_mode == "exp":
        return anneal_params[0] * np.exp(-t / anneal_params[1])

    elif anneal_mode == "log":
        return (anneal_params[0] * np.log(2)) / (np.log(t / anneal_params[1] + 2))

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

def run_iteration(p, simPar, iniPar, times, vals, sim_flags, verbose, logger, prev_p=None, t=0):
    # Calculates likelihood of a new proposed parameter set
    accepted = True
    logratio = 0 # acceptance ratio = 1
    p.likelihood = np.zeros(len(iniPar))
    for i in range(len(iniPar)):
        thickness, nx = unpack_simpar(simPar, i)
        next_init_condition = iniPar[i]
        sol, next_init_condition = do_simulation(p, thickness, nx, next_init_condition, times[i])
        try:
            if verbose and logger is not None: 
                logger.info("Simulation complete; final t {}".format(times[i][len(sol)-1]))
            
            sol, fail = detect_sim_fail(sol, vals[i])
            if fail and logger is not None:
                logger.warning(f"{i}: Simulation terminated early!")

            sol, fail = detect_sim_depleted(sol)
            if fail and logger is not None:
                logger.warning(f"{i}: Carriers depleted!")

            p.likelihood[i] -= np.sum((np.log10(sol) - vals[i])**2)
            # TRPL must be positive! Any simulation which results in depleted carrier is clearly incorrect
            if np.isnan(p.likelihood[i]): raise ValueError(f"{i}: Simulation failed!")
        except ValueError as e:
            logger.warning(e)
            p.likelihood[i] = -np.inf
            
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

# def start_metro_controller(simPar, iniPar, e_data, sim_flags, param_info, initial_guess_list, initial_variance, logger):
#     #num_cpus = 2
#     num_cpus = min(os.cpu_count(), sim_flags["num_initial_guesses"])
#     logger.info(f"{num_cpus} CPUs marshalled")
#     logger.info(f"{len(initial_guess_list)} MC chains needed")
#     with Pool(num_cpus) as pool:
#         histories = pool.map(partial(metro, simPar, iniPar, e_data, sim_flags, param_info, initial_variance, False, logger), initial_guess_list)
        
#     history_list = HistoryList(histories, param_info)
    
#     return history_list

def all_signal_handler(func):
    for s in signal.Signals:
        try:
            signal.signal(s, func)
        except ValueError:
            continue
    return
        
def metro(simPar, iniPar, e_data, sim_flags, param_info, initial_variance, verbose, logger, initial_guess):
    # Setup
    #np.random.seed(42)
    logger.info("PID: {}".format(os.getpid()))
    all_signal_handler(kill_from_cl)

    num_iters = sim_flags["num_iters"]
    DA_mode = sim_flags.get("delayed_acceptance", "off")
    checkpoint_freq = sim_flags["checkpoint_freq"]
    load_checkpoint = sim_flags["load_checkpoint"]

    times, vals, uncs = e_data
    # As init temperature we take cN, where c is specified in main and N is number of observations
    sim_flags["anneal_params"][0] *= sum([len(time)-1 for time in times])

    if load_checkpoint is not None:
        with open(os.path.join(sim_flags["checkpoint_dirname"], load_checkpoint), 'rb') as ifstream:
            MS = pickle.load(ifstream)
            np.random.set_state(MS.random_state)
            starting_iter = int(load_checkpoint[load_checkpoint.find("_")+1:load_checkpoint.rfind(".pik")])+1
    else:
        MS = MetroState(param_info, initial_guess, initial_variance, num_iters)
        starting_iter = 1
    
        # Calculate likelihood of initial guess
        run_iteration(MS.prev_p, simPar, iniPar, times, vals, sim_flags, verbose, logger)
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
                update_covariance(MS.variances, picked_param)

            if sim_flags["proposal_function"] == "gauss":
                select_next_params(MS.p, MS.means, MS.variances, param_info, logger)
            elif sim_flags["proposal_function"] == "box":
                select_from_box(MS.p, MS.means, MS.variances, param_info, logger)

            if verbose: print_status(MS.p, MS.means, param_info, logger)
            # Calculate new likelihood?
                    
            if DA_mode == "off":
                accepted = run_iteration(MS.p, simPar, iniPar, times, vals, sim_flags, verbose, logger, prev_p=MS.prev_p, t=k)
                
            elif DA_mode == 'DEBUG':
                accepted = False
                
            if verbose and not accepted:
                logger.info("Rejected!")
                
            if accepted:
                update_means(MS.p, MS.means, param_info)
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
