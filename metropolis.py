# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 22:13:26 2022

@author: cfai2
"""
import numpy as np
from scipy.integrate import solve_ivp

from forward_solver import dydt
from utils import Grid, Solution, Parameters, History

def model(init_dN, g, p):
    N = init_dN + p.n0
    P = init_dN + p.p0
    E_field = np.zeros(len(N)+1)
    
    init_condition = np.concatenate([N, P, E_field], axis=None)
    args = (g,p)
    sol = solve_ivp(dydt, [0,g.time], init_condition, args=args, t_eval=g.tSteps, method='BDF', max_step=g.hmax)
    data = sol.y.T
    s = Solution()
    s.N, s.P, E_field = np.split(data, [g.nx, 2*g.nx], axis=1)
    s.calculate_PL(g, p)
    return s.PL

def select_next_params(p, means, variances, param_info):
    is_active = param_info["active"]
    do_log = param_info["do_log"]
    for param in param_info["names"]:
        if is_active.get(param, 0):
            mean = getattr(means, param)
            var = getattr(variances, param)
            
            if do_log.get(param, 0):    
                setattr(p, param, 10 ** np.random.normal(loc=mean, scale=var))
                
            else:
                setattr(p, param, np.random.normal(loc=mean, scale=var))
                while getattr(p, param) <= 0:
                    # Negative values not allowed - choose again
                    setattr(p, param, np.random.normal(loc=mean, scale=var))
        
    return

def update_means(p, means, param_info):
    do_log = param_info["do_log"]
    for param in param_info['names']:
        if do_log.get(param, 0):
            setattr(means, param, np.log10(getattr(p, param)))
        else:
            setattr(means, param, getattr(p, param))
    return

def print_status(p, means, param_info):
    is_active = param_info['active']
    do_log = param_info["do_log"]
    ucs = param_info["unit_conversions"]
    for param in param_info['names']:
        if is_active.get(param, 0):
            if do_log.get(param, 0):
                print("Next {}: {} from mean {}".format(param, getattr(p, param) / ucs.get(param, 1), 10 ** getattr(means, param) / ucs.get(param, 1)))
            else:
                print("Next {}: {} from mean {}".format(param, getattr(p, param) / ucs.get(param, 1), getattr(means, param) / ucs.get(param, 1)))
            
    return

def update_history(H, k, p, means, param_info):
    do_log = param_info["do_log"]
    for param in param_info['names']:
        h = getattr(H, param)
        h[k] = getattr(p, param)
        h_mean = getattr(H, f"mean_{param}")
        
        if do_log.get(param, 0):
            h_mean[k] = 10 ** getattr(means, param)
        else:
            h_mean[k] = getattr(means, param)
    
def metro(simPar, iniPar, e_data, param_info, num_iters=5):
    # Setup
    np.random.seed(42)
    times, vals, uncs = e_data    
    tf = sum([len(time)-1 for time in times]) * (1/2000)
    p = Parameters(param_info)
    p.apply_unit_conversions(param_info)
    
    H = History(num_iters, param_info)
    
    prev_p = Parameters(param_info)
    prev_p.apply_unit_conversions(param_info)
    
    means = Parameters(param_info)
    means.apply_unit_conversions(param_info)
    means.make_log(param_info)
    
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
    prev_p.likelihood = np.zeros(len(iniPar))
    for i in range(len(iniPar)):
        g = Grid()
        g.thickness = simPar[0][i] if isinstance(simPar[0], list) else simPar[0]
        g.nx = simPar[2]
        g.dx = g.thickness / g.nx
        g.xSteps = np.linspace(g.dx / 2, g.thickness - g.dx/2, g.nx)
        
        g.time = times[i][-1]
        g.nt = len(times[i]) - 1
        g.dt = g.time / g.nt
        g.hmax = 4
        g.tSteps = times[i]
        
        sol = model(iniPar[i], g, p)
        
        prev_p.likelihood[i] -= np.sum((np.log10(sol) - vals[i])**2)
        #sol.plot_PL(g)
        
    prev_p.likelihood /= tf
    update_history(H, 0, prev_p, means, param_info)

    for k in range(1, num_iters):
        try:
            # Select next sample from distribution
            
            select_next_params(p, means, variances, param_info)
    
            print_status(p, means, param_info)
            # Calculate new likelihood?
            p.likelihood = np.zeros(len(iniPar))
            for i in range(len(iniPar)):
                g = Grid()
                g.thickness = simPar[0][i] if isinstance(simPar[0], list) else simPar[0]
                g.nx = simPar[2]
                g.dx = g.thickness / g.nx
                g.xSteps = np.linspace(g.dx / 2, g.thickness - g.dx/2, g.nx)
                
                g.time = times[i][-1]
                g.nt = len(times[i]) - 1
                g.dt = g.time / g.nt
                g.hmax = 4
                g.tSteps = times[i]
                
                sol = model(iniPar[i], g, p)
                
                p.likelihood[i] -= np.sum((np.log10(sol) - vals[i])**2)
                #sol.plot_PL(g)
                
                p.likelihood[i] /= tf
            
                # Compare with prior likelihood

                logratio = p.likelihood[i] - prev_p.likelihood[i]
            
                print("Partial Ratio: {}".format(10 ** logratio))
            
                accepted = False
                if logratio >= 0:
                    # Continue
                    accepted = True
                    
                else:
                    accept = np.random.random()
                    if accept < 10 ** logratio:
                        # Continue
                        accepted = True
                        
                if not accepted:
                    print("Rejected!")
                    break
                
            print("Iter {}".format(k))
            print("#####")
            if accepted:
                update_means(p, means, param_info)
                H.accept[k] = 1
                #H.ratio[k] = 10 ** logratio
                prev_p.likelihood = p.likelihood
                
            
            update_history(H, k, p, means, param_info)
        except KeyboardInterrupt:
            print("Terminating with k={} iters completed:".format(k-1))
            H.truncate(k, param_info)
            break
        
    return H