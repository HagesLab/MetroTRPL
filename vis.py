# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 19:11:50 2022

@author: cfai2
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from secondary_parameters import mu_eff, LI_tau_eff
true_vals = {"mu_n":20,
             "mu_p":20,
             "p0":3e15,
             "B":4.8e-11,
             "Sf":10,
             "Sb":10,
             "tauN":511,
             "tauP":871,}

adds = {"Sf+Sb": 20,}
pairs = [("Sf", "Sb")]
#adds = {"tau_eff": 454}
#adds = {"mu_eff":20}
#pairs = [("mu_n", "mu_p")]
#pairs = []

if __name__ == "__main__":
    path = "twothick on"
    
    path = os.path.join("bay_outputs", path)
    with open(os.path.join(path, "param_info.pik"), "rb") as ifstream:
        param_info = pickle.load(ifstream)
    
    names = param_info["names"]
    is_active = param_info["active"]
    do_log = param_info["do_log"]
    
    first = True
    for i, param in enumerate(names):
        if not is_active.get(param, 0):
            continue
        
        m = np.load(os.path.join(path, f"{param}.npy"))
        mm = np.load(os.path.join(path, f"mean_{param}.npy"))

        plt.figure(i, dpi=200, figsize=(3.5,2.7))
        plt.plot(m, label="Proposed samples")
        plt.plot(mm, label="Running mean")
        plt.axhline(true_vals[param], color='r', linestyle='dashed', label="Actual value")
        plt.ylabel(param)
        plt.xlabel("Sample #")
        if do_log.get(param, 0): plt.yscale('log')
        
        if first:
            plt.legend()
            first = False
            
    if "Sf+Sb" in adds:
        raw_sf = np.load(os.path.join(path, "Sf.npy"))
        mean_sf = np.load(os.path.join(path, "mean_Sf.npy"))
        raw_sb = np.load(os.path.join(path, "Sb.npy"))
        mean_sb = np.load(os.path.join(path, "mean_Sb.npy"))
        
        m = raw_sf + raw_sb
        mm = mean_sf + mean_sb
        
        plt.figure(1000, dpi=200, figsize=(3.5,2.7))
        plt.plot(m, label="Proposed samples")
        plt.plot(mm, label="Running mean")
        plt.axhline(adds["Sf+Sb"], color='r', linestyle='dashed', label="Actual value")
        plt.ylabel("Sf+Sb")
        plt.xlabel("Sample #")
        plt.yscale('log')

    if "tau_eff" in adds:
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
        
        m = LI_tau_eff(raw_B, raw_p0, raw_tau_n, raw_sf, raw_sb, 2000, mu_a)
        mm = LI_tau_eff(mean_B, mean_p0, mean_tau_n, mean_sf, mean_sb, 2000, mean_mu_a)
        
        plt.figure(2000, dpi=200, figsize=(3.5,2.7))
        plt.plot(m, label="Proposed samples")
        plt.plot(mm, label="Running mean")
        plt.axhline(adds["tau_eff"], color='r', linestyle='dashed', label="Actual value")
        plt.ylabel("tau_eff")
        plt.xlabel("Sample #")
        
    if "mu_eff" in adds:
        raw_mu_n = np.load(os.path.join(path, "mu_n.npy"))
        mean_mu_n = np.load(os.path.join(path, "mean_mu_n.npy"))
        raw_mu_p = np.load(os.path.join(path, "mu_p.npy"))
        mean_mu_p = np.load(os.path.join(path, "mean_mu_p.npy"))
        
        m = mu_eff(raw_mu_n, raw_mu_p)
        mm = mu_eff(mean_mu_n, mean_mu_p)
        
        plt.figure(3000, dpi=200, figsize=(3.5,2.7))
        plt.plot(m, label="Proposed samples")
        plt.plot(mm, label="Running mean")
        plt.axhline(adds["mu_eff"], color='r', linestyle='dashed', label="Actual value")
        plt.ylabel("mu_eff")
        plt.xlabel("Sample #")
        
    for p in pairs:
        fig, ax2d = plt.subplots(1,1,figsize=(3.5*1.1,3.5), dpi=120)
        from matplotlib.cm import ScalarMappable
        paramx = p[0]
        paramy = p[1]
        mmx = np.load(os.path.join(path, f"mean_{paramx}.npy"))
        mmy = np.load(os.path.join(path, f"mean_{paramy}.npy"))
        color_grad = np.linspace(0, 1, len(mmx))
        
        minx = min(mmx)
        miny = min(mmy)
        maxx = max(mmx)
        maxy = max(mmy)
        
        bin_ctx = 96
        bin_cty = 96
        
        bins_x = np.arange(bin_ctx+1)
        bins_y = np.arange(bin_cty+1)

        bins_x   = minx + (maxx-minx)*(bins_x)/bin_ctx    # Get params
        bins_y   = miny + (maxy-miny)*(bins_y)/bin_cty
        im = np.histogram2d(mmx, mmy, bins=[bins_x,bins_y], density=True)
        
        marP_corr = im[0] * 1
        Y_corr, X_corr = np.meshgrid(bins_x, bins_y)
        ax2d.scatter(mmx, mmy, c=color_grad)
        #### S Specific ####
        sf = np.linspace(minx, maxx, 100)
        sb = np.linspace(miny, maxy, 100)

        sf, sb = np.meshgrid(sf, sb)

        Z = sf+sb

        levels = [20]
        cwg = ax2d.contour(sf, sb, Z, levels=levels, colors='black', zorder=9999)
        ax2d.clabel(cwg)        
        ####################
        
        #### MU SPECIFIC ####
        # mu_n = np.linspace(minx, maxx, 100)
        # mu_p = np.linspace(miny, maxy, 100)

        # mu_n, mu_p = np.meshgrid(mu_n, mu_p)

        # Z = 2 / (mu_n**-1 + mu_p**-1)
        
        
        # levels = [20]
        # cwg = ax2d.contour(mu_n, mu_p, Z, levels=levels, colors='black')
        # ax2d.clabel(cwg)    
        #####################
        ax2d.set_xlabel(paramx)
        ax2d.set_ylabel(paramy)
        # ax2d.set_yscale('log')
        # ax2d.set_xscale('log')
        cbar = fig.colorbar(ScalarMappable(), ax=ax2d, ticks=[0,1])
        cbar.ax.set_yticklabels(["Initial", "Final"])
        # ax2d[1].pcolormesh(Y_corr, X_corr, marP_corr.T, cmap='Blues')
        # ax2d[1].set_xlabel(paramx)
        # ax2d[1].set_ylabel(paramy)
        # ax2d[1].set_yscale('log')
        # ax2d[1].set_xscale('log')
        fig.tight_layout()