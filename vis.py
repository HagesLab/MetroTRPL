# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 19:11:50 2022

@author: cfai2
"""

import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams.update({'font.family':'STIXGeneral'})
matplotlib.rcParams.update({'mathtext.fontset':'stix'})
import os
import pickle

from metro_plot_lib import make_1D_tracker, make_2D_tracker, make_1D_histo, make_2D_histo
from postprocessing_utils import fetch_param, recommend_logscale, calc_contours


true_vals = {"mu_n":20,
             "mu_p":20,
             "p0":3e15,
             "B":4.8e-11,
             "Sf":10,
             "Sb":10,
             "tauN":511,
             "tauP":871,}

do = {"1D_trackers":0,
      "2D_trackers":1,
      "1D_histos":0,
      "2D_histos":1}

burn = 250
adds = {"Sf+Sb": 20, "tau_eff": 454}
pairs = [("Sf", "Sb", 20)]
#adds = {"tau_eff": 454}
adds = {"mu_eff":20}
#pairs = [("mu_n", "mu_p")]
#pairs = []

if __name__ == "__main__":
    path = "earlymu cuDA"
    
    path = os.path.join("bay_outputs", path)
    with open(os.path.join(path, "param_info.pik"), "rb") as ifstream:
        param_info = pickle.load(ifstream)
    
    names = param_info["names"]
    is_active = param_info["active"]
    do_log = param_info["do_log"]
    
    if do["1D_trackers"]:
        first = True
        for i, param in enumerate(names):
            if not is_active.get(param, 0):
                continue
            
            proposed = np.load(os.path.join(path, f"{param}.npy"))
            accepted = np.load(os.path.join(path, f"mean_{param}.npy"))
            make_1D_tracker(proposed, accepted, mark_value=true_vals[param], ylabel=param,
                            show_legend=first, do_log=do_log[param])
            first = False
             
        for add_param in adds:
            proposed, accepted = fetch_param(path, add_param, thickness=2000)
            recommended_log = recommend_logscale(add_param, do_log)
            make_1D_tracker(proposed, accepted, mark_value=adds[add_param], ylabel=add_param,
                            show_legend=first, do_log=recommended_log)
            
    if do["1D_histos"]:
        for i, param in enumerate(names):
            if not is_active.get(param, 0):
                continue
            
            accepted = np.load(os.path.join(path, f"mean_{param}.npy"))
            accepted = accepted[burn:]
            
            make_1D_histo(accepted, mark_value=true_vals[param], xlabel=param,
                          do_log=do_log[param], bin_count=24)
            
            print(param, " mean ", np.mean(accepted))
            print(param, " std ", np.std(accepted))
            
        for add_param in adds:
            proposed, accepted = fetch_param(path, add_param, thickness=2000)
            recommended_log = recommend_logscale(add_param, do_log)
            make_1D_histo(accepted, mark_value=adds[add_param], xlabel=add_param,
                          do_log=recommended_log, bin_count=24)
            
            print(add_param, " mean ", np.mean(accepted))
            print(add_param, " std ", np.std(accepted))
        
        
    if do["2D_trackers"] or do["2D_histos"]:
        for p in pairs:
            paramx = p[0]
            paramy = p[1]
            try:
                clevels = p[2]
                if isinstance(clevels, (float, int)):
                    clevels = [clevels]
            except IndexError:
                clevels = None
            x_accepted = np.load(os.path.join(path, f"mean_{paramx}.npy"))
            y_accepted = np.load(os.path.join(path, f"mean_{paramy}.npy"))
            if clevels is not None:
                contour_info = calc_contours(x_accepted, y_accepted, clevels, (paramx, paramy))
            else:
                contour_info = None
            if do["2D_trackers"]:
                make_2D_tracker(x_accepted, y_accepted, markx=true_vals[paramx], marky=true_vals[paramy],
                                do_logx=do_log[paramx], do_logy=do_log[paramy],xlabel=paramx,ylabel=paramy,
                                contours=contour_info)
                
            if do["2D_histos"]:
                make_2D_histo(x_accepted, y_accepted, markx=true_vals[paramx], marky=true_vals[paramy],
                              do_logx=do_log[paramx], do_logy=do_log[paramy],xlabel=paramx,ylabel=paramy,
                              contours=contour_info)

                
            