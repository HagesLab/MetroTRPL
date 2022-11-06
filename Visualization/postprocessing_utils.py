# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:53:02 2022

@author: cfai2
"""
import numpy as np
import os
from secondary_parameters import mu_eff, LI_tau_eff, HI_tau_srh, LI_tau_srh
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
importr('coda')
calc_ESS = robjects.r['effectiveSize']

def recommend_logscale(which_param, do_log):
    if which_param == "Sf+Sb":
        recommend = (do_log["Sf"] or do_log["Sb"])
        
    elif which_param == "tauN+tauP":
        recommend = (do_log["tauN"] or do_log["tauP"])
        
    elif which_param == "Cn+Cp":
        recommend = (do_log["Cn"] or do_log["Cp"])
        
    elif which_param == "tau_eff" or which_param == "tau_srh":
        recommend = (do_log["tauN"] or (do_log["Sf"] or do_log["Sb"]))
        
    elif which_param == "HI_tau_srh":
        recommend = (do_log["tauN"] or do_log["tauP"] or (do_log["Sf"] or do_log["Sb"]))
        
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

def binned_stderr(vals, bins):
    """
    Calculate a standard error for a MC chain with states denoted by vals[] by
    computing variance of subsample means.
    
    vals[] must be evenly subdivisible when bins are applied.

    Parameters
    ----------
    vals : 1D array
        List of states or parameter values visited by MC chain.
    bins : 1D array
        List of indices to divide vals[] using numpy.split.

    Returns
    -------
    sub_means : 1D array
        List of means from each subdivided portion of vals[].
    stderr : float
        Standard error computed by sqrt(var(sub_means)). This should be divided
        by sqrt(ESS) to get a sample stderr.

    """
    
    accepted_subs = np.split(vals, bins)
    
    lengths = np.array(list(map(len, accepted_subs)))
    if not np.all(lengths == lengths[0]):
        raise ValueError("Uneven binning")
        
    num_bins = len(accepted_subs)
    sub_means = np.zeros(num_bins)
    for s, sub in enumerate(accepted_subs):
        sub_means[s] = np.mean(sub)
        
    stderr = np.std(sub_means, ddof=1)# / np.sqrt(num_bins)
    
    return sub_means, stderr

def ASJD(means, window):   
    diffs = np.diff(means[:,:,window[0]:window[1]], axis=2)
    diffs = diffs ** 2
    diffs = np.sum(diffs, axis=0) # [chain, iter]
    
    diffs = np.mean(diffs, axis=1) # [chain]
    return diffs

def ESS(means, do_log, verbose=True):
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
        
    elif which_param == "tauN+tauP":
        params = ["tauN", "tauP"]
        
    elif which_param == "tau_eff":
        params = ["Sf", "Sb", "tauN", "mu_n", "mu_p", "ks", "p0", "Cp"]
        
    elif which_param == "tau_srh":
        params = ["Sf", "Sb", "tauN", "mu_n", "mu_p"]
        
    elif which_param == "HI_tau_srh":
        params = ["Sf", "Sb", "tauN", "tauP", "mu_n", "mu_p"]
        
    elif which_param == "mu_eff":
        params = ["mu_n", "mu_p"]
        
    elif which_param == "Cn+Cp":
        params = ["Cn", "Cp"]
        
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
        
    elif which_param == "tauN+tauP":
        proposed = raw_fetched["tauN"] + raw_fetched["tauP"]
        accepted = mean_fetched["tauN"] + mean_fetched["tauP"]
        
    elif which_param == "Cn+Cp":
        proposed = raw_fetched["Cn"] + raw_fetched["Cp"]
        accepted = mean_fetched["Cn"] + mean_fetched["Cp"]
        
    elif which_param == "tau_eff":
        mu_a = mu_eff(raw_fetched["mu_n"], raw_fetched["mu_p"])
        mean_mu_a = mu_eff(mean_fetched["mu_n"], mean_fetched["mu_p"])
        
        thickness = kwargs.get("thickness", 2000)
        
        proposed = LI_tau_eff(raw_fetched["ks"], raw_fetched["p0"], raw_fetched["tauN"], 
                              raw_fetched["Sf"], raw_fetched["Sb"], raw_fetched["Cp"], 
                              thickness, mu_a)
        accepted = LI_tau_eff(mean_fetched["ks"], mean_fetched["p0"], mean_fetched["tauN"], 
                              mean_fetched["Sf"], mean_fetched["Sb"], mean_fetched["Cp"],
                              thickness, mean_mu_a)
        
    elif which_param == "tau_srh":
        mu_a = mu_eff(raw_fetched["mu_n"], raw_fetched["mu_p"])
        mean_mu_a = mu_eff(mean_fetched["mu_n"], mean_fetched["mu_p"])
        
        thickness = kwargs.get("thickness", 2000)
        
        proposed = LI_tau_srh(raw_fetched["tauN"], 
                              raw_fetched["Sf"], raw_fetched["Sb"], 
                              thickness, mu_a)
        accepted = LI_tau_srh(mean_fetched["tauN"], 
                              mean_fetched["Sf"], mean_fetched["Sb"],
                              thickness, mean_mu_a)
        
    elif which_param == "mu_eff":
        proposed = mu_eff(raw_fetched["mu_n"], raw_fetched["mu_p"])
        accepted = mu_eff(mean_fetched["mu_n"], mean_fetched["mu_p"])
        
    elif which_param == "HI_tau_srh":
        mu_a = mu_eff(raw_fetched["mu_n"], raw_fetched["mu_p"])
        mean_mu_a = mu_eff(mean_fetched["mu_n"], mean_fetched["mu_p"])
        
        thickness = kwargs.get("thickness", 2000)
        proposed = HI_tau_srh(raw_fetched["tauN"], raw_fetched["tauP"],
                              raw_fetched["Sf"], raw_fetched["Sb"], thickness, mu_a)
        accepted = HI_tau_srh(mean_fetched["tauN"], mean_fetched["tauP"],
                              mean_fetched["Sf"], mean_fetched["Sb"], thickness, mean_mu_a)
        
        
    return proposed, accepted

def calc_contours(x_accepted, y_accepted, clevels, which_params, size=1000, do_logx=False, do_logy=False,xrange=None,yrange=None):
    if xrange is not None:
        minx = min(xrange)
        maxx = max(xrange)
    else:
        minx = min(x_accepted.flatten())
        maxx = max(x_accepted.flatten())
    if yrange is not None:
        miny = min(yrange)
        maxy = max(yrange)
    else:
        miny = min(y_accepted.flatten())
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
        
    elif "Sf+Sb" in which_params and "tauN" in which_params:
        kb = 0.0257 #[ev]
        q = 1
        
        D = 20 * kb / q * 10**14 / 10**9
        tau_surf = (2000 / ((cx)*0.01)) + (2000**2 / (np.pi ** 2 * D))

        cZ = (tau_surf**-1 + cy**-1)**-1
        
    elif "Sf+Sb" in which_params and "tauN+tauP" in which_params:
        kb = 0.0257 #[ev]
        q = 1
        
        D = 20 * kb / q * 10**14 / 10**9
        tau_surf = 2*(311 / ((cx)*0.01)) + (311**2 / (np.pi ** 2 * D))

        cZ = (tau_surf**-1 + cy**-1)**-1
    else:
        raise NotImplementedError
        
    return (cx, cy, cZ, clevels)