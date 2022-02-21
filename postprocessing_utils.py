# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:53:02 2022

@author: cfai2
"""
import numpy as np
import os
from secondary_parameters import mu_eff, LI_tau_eff

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

def fetch(path, which_param):
    raw_fetched = {}
    mean_fetched = {}
    if which_param == "Sf+Sb":
        raw_fetched["Sf"] = np.load(os.path.join(path, "Sf.npy"))
        mean_fetched["Sf"] = np.load(os.path.join(path, "mean_Sf.npy"))
        raw_fetched["Sb"] = np.load(os.path.join(path, "Sb.npy"))
        mean_fetched["Sb"] = np.load(os.path.join(path, "mean_Sb.npy"))
        
    elif which_param == "tau_eff":
        raw_fetched["Sf"] = np.load(os.path.join(path, "Sf.npy"))
        mean_fetched["Sf"] = np.load(os.path.join(path, "mean_Sf.npy"))
        raw_fetched["Sb"] = np.load(os.path.join(path, "Sb.npy"))
        mean_fetched["Sb"] = np.load(os.path.join(path, "mean_Sb.npy"))
        raw_fetched["tauN"] = np.load(os.path.join(path, "tauN.npy"))
        mean_fetched["tauN"] = np.load(os.path.join(path, "mean_tauN.npy"))
        raw_fetched["mu_n"] = np.load(os.path.join(path, "mu_n.npy"))
        mean_fetched["mu_n"] = np.load(os.path.join(path, "mean_mu_n.npy"))
        raw_fetched["mu_p"] = np.load(os.path.join(path, "mu_p.npy"))
        mean_fetched["mu_p"] = np.load(os.path.join(path, "mean_mu_p.npy"))
        raw_fetched["B"] = np.load(os.path.join(path, "B.npy"))
        mean_fetched["B"] = np.load(os.path.join(path, "mean_B.npy"))
        raw_fetched["p0"] = np.load(os.path.join(path, "p0.npy"))
        mean_fetched["p0"] = np.load(os.path.join(path, "mean_p0.npy"))
        
    elif which_param == "mu_eff":
        raw_fetched["mu_n"] = np.load(os.path.join(path, "mu_n.npy"))
        mean_fetched["mu_n"] = np.load(os.path.join(path, "mean_mu_n.npy"))
        raw_fetched["mu_p"] = np.load(os.path.join(path, "mu_p.npy"))
        mean_fetched["mu_p"] = np.load(os.path.join(path, "mean_mu_p.npy"))
        
    return raw_fetched, mean_fetched

def fetch_param(raw_fetched, mean_fetched, which_param, **kwargs):
    if which_param == "Sf+Sb":
        proposed = raw_fetched["Sf"] + raw_fetched["Sb"]
        accepted = mean_fetched["Sf"] + mean_fetched["Sb"]
        
    elif which_param == "tau_eff":
        mu_a = mu_eff(raw_fetched["mu_n"], raw_fetched["mu_p"])
        mean_mu_a = mu_eff(mean_fetched["mu_n"], mean_fetched["mu_p"])
        
        thickness = kwargs.get("thickness", 2000)
        
        proposed = LI_tau_eff(raw_fetched["B"], raw_fetched["p0"], raw_fetched["tauN"], 
                              raw_fetched["Sf"], raw_fetched["Sb"], thickness, mu_a)
        accepted = LI_tau_eff(mean_fetched["B"], mean_fetched["p0"], mean_fetched["tauN"], 
                              mean_fetched["Sf"], mean_fetched["Sb"], thickness, mean_mu_a)
        
    elif which_param == "mu_eff":
        proposed = mu_eff(raw_fetched["mu_n"], raw_fetched["mu_p"])
        accepted = mu_eff(mean_fetched["mu_n"], mean_fetched["mu_p"])
        
    return proposed, accepted

def calc_contours(x_accepted, y_accepted, clevels, which_params, size=1000, do_logx=False, do_logy=False):
    minx = min(x_accepted)
    miny = min(y_accepted)
    maxx = max(x_accepted)
    maxy = max(y_accepted)
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
    
    