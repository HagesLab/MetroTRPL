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

def fetch_param(path, which_param, **kwargs):
    if which_param == "Sf+Sb":
        raw_sf = np.load(os.path.join(path, "Sf.npy"))
        mean_sf = np.load(os.path.join(path, "mean_Sf.npy"))
        raw_sb = np.load(os.path.join(path, "Sb.npy"))
        mean_sb = np.load(os.path.join(path, "mean_Sb.npy"))
        
        proposed = raw_sf + raw_sb
        accepted = mean_sf + mean_sb
        
    elif which_param == "tau_eff":
        raw_sf = np.load(os.path.join(path, "Sf.npy"))
        mean_sf = np.load(os.path.join(path, "mean_Sf.npy"))
        raw_sb = np.load(os.path.join(path, "Sb.npy"))
        mean_sb = np.load(os.path.join(path, "mean_Sb.npy"))
        raw_tau_n = np.load(os.path.join(path, "tauN.npy"))
        mean_tau_n = np.load(os.path.join(path, "mean_tauN.npy"))
        raw_mu_n = np.load(os.path.join(path, "mu_n.npy"))
        mean_mu_n = np.load(os.path.join(path, "mean_mu_n.npy"))
        raw_mu_p = np.load(os.path.join(path, "mu_p.npy"))
        mean_mu_p = np.load(os.path.join(path, "mean_mu_p.npy"))
        raw_B = np.load(os.path.join(path, "B.npy"))
        mean_B = np.load(os.path.join(path, "mean_B.npy"))
        raw_p0 = np.load(os.path.join(path, "p0.npy"))
        mean_p0 = np.load(os.path.join(path, "mean_p0.npy"))
        
        mu_a = mu_eff(raw_mu_n, raw_mu_p)
        mean_mu_a = mu_eff(mean_mu_n, mean_mu_p)
        
        thickness = kwargs.get("thickness", 2000)
        
        proposed = LI_tau_eff(raw_B, raw_p0, raw_tau_n, raw_sf, raw_sb, thickness, mu_a)
        accepted = LI_tau_eff(mean_B, mean_p0, mean_tau_n, mean_sf, mean_sb, thickness, mean_mu_a)
        
    elif which_param == "mu_eff":
        raw_mu_n = np.load(os.path.join(path, "mu_n.npy"))
        mean_mu_n = np.load(os.path.join(path, "mean_mu_n.npy"))
        raw_mu_p = np.load(os.path.join(path, "mu_p.npy"))
        mean_mu_p = np.load(os.path.join(path, "mean_mu_p.npy"))
        
        proposed = mu_eff(raw_mu_n, raw_mu_p)
        accepted = mu_eff(mean_mu_n, mean_mu_p)
        
    return proposed, accepted

def calc_contours(x_accepted, y_accepted, clevels, which_params):
    minx = min(x_accepted)
    miny = min(y_accepted)
    maxx = max(x_accepted)
    maxy = max(y_accepted)
    cx = np.linspace(minx, maxx, 100)
    cy = np.linspace(miny, maxy, 100)

    cx, cy = np.meshgrid(cx, cy)

    if "Sf" in which_params and "Sb" in which_params:
        # Sf+Sb
        cZ = cx+cy
        
    return (cx, cy, cZ, clevels)
    
    