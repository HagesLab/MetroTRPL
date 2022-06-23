# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:53:02 2022

@author: cfai2
"""
import numpy as np
import os
from secondary_parameters import mu_eff, LI_tau_eff
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
importr('coda')
calc_ESS = robjects.r['effectiveSize']

def recommend_logscale(which_param, do_log):
    if which_param == "Sf+Sb":
        recommend = (do_log["Sf"] or do_log["Sb"])
        
    elif which_param == "tau_eff":
        recommend = (do_log["tauN"] or (do_log["Sf"] or do_log["Sb"]))
        
    elif which_param == "mu_eff":
        recommend = (do_log["mu_n"] or do_log["mu_p"])
        
    elif which_param in do_log:
        recommend = do_log[which_param]
        
    else:
        recommend = False

    return recommend

def load_all_accepted(path, names, is_active, do_log):
    means = []
    for i, param in enumerate(names):
        if not is_active.get(param, 0):
            continue
        
        a = np.load(os.path.join(path, f"mean_{param}.npy"))
        
        if do_log[param]:
            a = np.log10(a)
            
        means.append(a)
    return np.array(means)

def ASJD(means, window):   
    diffs = np.diff(means[:,:,window[0]:window[1]], axis=2)
    diffs = diffs ** 2
    diffs = np.sum(diffs, axis=0) # [chain, iter]
    
    diffs = np.mean(diffs, axis=1) # [chain]
    return diffs

def ESS(means, window, do_log, verbose=True):
    means = means[:, window[0]:window[1]]
    
    if do_log:
        means = np.log10(means)
        
    if means.ndim == 1:
        chains = [means]
    elif means.ndim == 2:
        chains = []
        for chain in means:
            chains.append(chain)
            
    ess = np.zeros(len(chains))
    for c_id, chain in enumerate(chains):
        actual_N = len(chain)
        chain = robjects.FloatVector(chain)
        e = calc_ESS(chain)
        if verbose:
            print("Chain # {}".format(c_id))
            print("Actual N: {}".format(actual_N))
            print("ESS: {}".format(e[0]))
        ess[c_id] = e[0]
    
    avg_ess = np.mean(ess[ess != 0])
    return avg_ess

def fetch(path, which_param):
    raw_fetched = {}
    mean_fetched = {}
    if which_param == "Sf+Sb":
        params = ["Sf", "Sb"]
        
    elif which_param == "tau_eff":
        params = ["Sf", "Sb", "tauN", "mu_n", "mu_p", "ks", "p0"]
        
    elif which_param == "mu_eff":
        params = ["mu_n", "mu_p"]
        
    else:
        raise KeyError
        
    for param in params:
        raw_fetched[param] = np.load(os.path.join(path, f"{param}.npy"))
        mean_fetched[param] = np.load(os.path.join(path, f"mean_{param}.npy"))

    return raw_fetched, mean_fetched

def fetch_param(raw_fetched, mean_fetched, which_param, **kwargs):
    if which_param == "Sf+Sb":
        proposed = raw_fetched["Sf"] + raw_fetched["Sb"]
        accepted = mean_fetched["Sf"] + mean_fetched["Sb"]
        
    elif which_param == "tau_eff":
        mu_a = mu_eff(raw_fetched["mu_n"], raw_fetched["mu_p"])
        mean_mu_a = mu_eff(mean_fetched["mu_n"], mean_fetched["mu_p"])
        
        thickness = kwargs.get("thickness", 2000)
        
        proposed = LI_tau_eff(raw_fetched["ks"], raw_fetched["p0"], raw_fetched["tauN"], 
                              raw_fetched["Sf"], raw_fetched["Sb"], thickness, mu_a)
        accepted = LI_tau_eff(mean_fetched["ks"], mean_fetched["p0"], mean_fetched["tauN"], 
                              mean_fetched["Sf"], mean_fetched["Sb"], thickness, mean_mu_a)
        
    elif which_param == "mu_eff":
        proposed = mu_eff(raw_fetched["mu_n"], raw_fetched["mu_p"])
        accepted = mu_eff(mean_fetched["mu_n"], mean_fetched["mu_p"])
        
    return proposed, accepted

def calc_contours(x_accepted, y_accepted, clevels, which_params, size=1000, do_logx=False, do_logy=False):
    minx = min(x_accepted.flatten())
    miny = min(y_accepted.flatten())
    maxx = max(x_accepted.flatten())
    maxy = max(y_accepted.flatten())
    if do_logx:
        cx = np.geomspace(minx, maxx, size)
    else:
        cx = np.linspace(minx, maxx, size)
        
    if do_logy:
        cy = np.geomspace(miny, maxy, size)
    else:
        cy = np.linspace(miny, maxy, size)

    cx, cy = np.meshgrid(cx, cy)

    if "Sf" in which_params and "Sb" in which_params:
        # Sf+Sb
        cZ = cx+cy
        
    elif "mu_n" in which_params and "mu_p" in which_params:
        cZ = mu_eff(cx, cy)
        
    else:
        raise NotImplementedError
        
    return (cx, cy, cZ, clevels)