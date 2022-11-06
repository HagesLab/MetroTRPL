# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 19:11:50 2022

@author: cfai2
"""

import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'font.family':'STIXGeneral'})
matplotlib.rcParams.update({'mathtext.fontset':'stix'})
import os
import pickle

from metro_plot_lib import make_1D_tracker, make_2D_tracker, make_1D_histo, make_2D_histo
from metro_plot_lib import plot_history
from postprocessing_utils import fetch, fetch_param, recommend_logscale, calc_contours
from postprocessing_utils import load_all_accepted, ASJD, ESS, binned_stderr

stylized_names = {"p0":r"$p_0$", "ks":r"$k^*$", "mu_n":r"$\mu_n$","mu_p":r"$\mu_p$", "mu_eff":r"$\mu\prime$",
                  "Sf+Sb":r"$S_F+S_B$", "Sf":r"$S_F$", "Sb":r"$S_B$",
                  "tauN":r"$\tau_n$", "tauP":r"$\tau_p$", "tau_srh":r"$\tau_{SRH}$",
                  "Cn+Cp":r"$C_n+C_p$", "tauN+tauP":r"$\tau_n+\tau_p$"}

true_vals = {"mu_n":320,
              "mu_p":80,
              "mu_eff":40,
              "p0":3e15,
              "ks":4.8e-11,
              "Sf":10,
              "Sb":10,
              "Cn":4.4e-29,
              "Cp":4.4e-29,
              "Cn+Cp":8.8e-29,
              "tauN":511,
              "tauP":871,
              "tau_eff": 454,
              "HI_tau_srh": 1290,
              "Sf+Sb":20}
true_vals = {}

do = {"1D_trackers":0,
      "2D_trackers":0,
      "1D_histos":1,
      "2D_histos":0,
      "ASJD":0,
      "ESS":0,
      "binning":1}

axis_overrides_2D = {"mu_n":(1e0,1e2),
                  "mu_p":(1e0,1e2),
                  "mu_eff":(1e-4,1e4),
                  "p0":(2e14, 3e16),
                  "ks":(1e-11, 3e-10),
                  "Sf":(1e0, 3e1),
                  "Sb":(1e0, 3e1),
                  "tauN":(1e2, 1e4),
                  "tauP":(1e-4, 1e4),
                  "tau_eff":(1e-4, 1e4),
                  "Sf+Sb":(1e1, 1e3),
                  "tauN+tauP":(1e2,1e4)}
thickness = 3000 #[nm]
axis_overrides_1D = None

window = [0,1282]
window = [20000, 50000]
adds = {}
pairs = [('p0', 'ks')]
pairs = [('mu_n', 'mu_p', 20)]
#pairs = [("Sf+Sb", "tauN"), ]#('p0', 'B'), ('p0', 'tauN'), ('p0', 'Sf'), ('p0', 'Sb'), ('p0', 'tau_eff')]
#adds = {"mu_eff":None}
adds = {"tau_srh":None}
#adds = {"HI_tau_srh":None, "Cn+Cp":None, "mu_eff":None}
pairs = [("Sf+Sb", "tauN", 485)]
pairs = [("Sf", "Sb", 20)]


if __name__ == "__main__":
    #path = "1T_step_all_0/1T_step_all_0-0
    path = os.path.join("2A1FSGS_TRPL", str(0))
    #path = os.path.join("2T_auger_definitive_withauger", str(2))

    #path = "DEBUG/DEBUG-0"
    
    path = os.path.join("..", "bay_outputs", path)
    with open(os.path.join(path, "param_info.pik"), "rb") as ifstream:
        param_info = pickle.load(ifstream)
        
    with open(os.path.join(path, "sim_flags.pik"), "rb") as ifstream:
        sim_flags = pickle.load(ifstream)
        
    
    
    
    did_multicore = sim_flags.get("joined", False)
    
    names = param_info["names"]
    is_active = param_info["active"]
    do_log = param_info["do_log"]
    binning = do["binning"]
    
    accept_history = np.load(os.path.join(path, "accept.npy"))
    accept_history = accept_history[window[0]:window[1]]
    
    m_binning = 100 # Mandatory binning to be able to track acceptance%
    bins = np.arange(0, len(accept_history), int(m_binning))
    accept_sub_means, accept_stderr = binned_stderr(accept_history, bins[1:])
    
    plot_history(bins + 0.5*m_binning, accept_sub_means, 
                 xlabel = "Sample #", ylabel = "Acceptance rate")
    
    if do["1D_trackers"]:
        first = False
        import matplotlib.pyplot as plt
        #fig, ax = plt.subplots(2,5, figsize=(3.5*5,2.7*2), dpi=200)
        plot_counter = 0
        for i, param in enumerate(names):
            if not is_active.get(param, 0):
                continue
            fig, ax = plt.subplots(1,1, figsize=(3.5,2.7), dpi=200, squeeze=False)
            
            # j = plot_counter // 5
            # k = plot_counter % 5
            j = 0
            k = 0
            
            proposed = np.load(os.path.join(path, f"{param}.npy"))
            accepted = np.load(os.path.join(path, f"mean_{param}.npy"))
            
            proposed = proposed[window[0]:window[1]]
            accepted = accepted[window[0]:window[1]]
            
            if binning is not None and binning > 0:
                bins = np.arange(0, len(accepted), int(binning))
                if do_log: accepted = np.log10(accepted)
                accepted, stderr = binned_stderr(accepted, bins[1:])
                accepted = 10 ** accepted
                x = bins + 0.5 * binning
                print(param)
                print(stderr)
            else:
                x = np.arange(len(accepted))
            
            make_1D_tracker(x, accepted, mark_value=true_vals.get(param, None), ylabel=stylized_names.get(param, param),
                            show_legend=first, do_log=do_log[param], did_multicore=did_multicore,
                            fig=fig, ax=ax[j,k])

            plot_counter += 1
             
        for ii, add_param in enumerate(adds):

            # j = (plot_counter) // 5
            # k = (plot_counter) % 5
            fig, ax = plt.subplots(1,1, figsize=(3.5,2.7), dpi=200, squeeze=False)
            j = 0
            k = 0
            
            raw_fetched, mean_fetched = fetch(path, add_param)
            for p in raw_fetched:
                raw_fetched[p] = raw_fetched[p][window[0]:window[1]]
                mean_fetched[p] = mean_fetched[p][window[0]:window[1]]
            proposed, accepted = fetch_param(raw_fetched, mean_fetched, add_param, thickness=thickness)
            
            recommended_log = recommend_logscale(add_param, do_log)
            
            if binning is not None and binning > 0:
                bins = np.arange(0, len(accepted), int(binning))
                if do_log: accepted = np.log10(accepted)
                accepted, stderr = binned_stderr(accepted, bins[1:])
                accepted = 10 ** accepted
                x = bins + 0.5 * binning
                print(add_param)
                print(stderr)
                
            else:
                x = np.arange(len(accepted))
            make_1D_tracker(x, accepted, mark_value=adds.get(add_param, None), ylabel=stylized_names.get(add_param, add_param),
                            show_legend=first, do_log=recommended_log, did_multicore=did_multicore,
                            fig=fig, ax=ax[j,k])
            plot_counter += 1
            
        fig.tight_layout()

    if do["1D_histos"]:
        for i, param in enumerate(names):
            if not is_active.get(param, 0):
                continue
            
            accepted = np.load(os.path.join(path, f"mean_{param}.npy"))
            if did_multicore:
                accepted = accepted[:, window[0]:window[1]]
            else:
                accepted = accepted[window[0]:window[1]]
            
            make_1D_histo(accepted, mark_value=true_vals.get(param, None), xlabel=stylized_names.get(param, param),
                          do_log=0, bin_count=96, did_multicore=did_multicore, axis_overrides=axis_overrides_1D,
                          show_legend=False, size=(3,3))
            
            print(param, " mean ", np.mean(accepted))
            print(param, " std ", np.std(accepted))
            
        for add_param in adds:
            raw_fetched, mean_fetched = fetch(path, add_param)
            for p in raw_fetched:
                raw_fetched[p] = raw_fetched[p][window[0]:window[1]]
                mean_fetched[p] = mean_fetched[p][window[0]:window[1]]
            proposed, accepted = fetch_param(raw_fetched, mean_fetched, add_param, thickness=thickness)
            # if did_multicore:
            #     accepted = accepted[:, window[0]:window[1]]
            # else:
            #     accepted = accepted[window[0]:window[1]]
            
            recommended_log = recommend_logscale(add_param, do_log)
            make_1D_histo(accepted, mark_value=adds.get(add_param, None), xlabel=stylized_names.get(add_param, add_param),
                          do_log=0, bin_count=96, did_multicore=did_multicore, axis_overrides=axis_overrides_1D, 
                          show_legend=False, size=(3,3))
            
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
                
            if paramx in adds:
                do_log[paramx] = True
                raw_fetched, mean_fetched = fetch(path, paramx)
                for p in raw_fetched:
                    raw_fetched[p] = raw_fetched[p][window[0]:window[1]]
                    mean_fetched[p] = mean_fetched[p][window[0]:window[1]]
                proposed, x_accepted = fetch_param(raw_fetched, mean_fetched, paramx, thickness=thickness)
                do_log[paramx] = recommend_logscale(paramx, do_log)
                
            else:
                x_accepted = np.load(os.path.join(path, f"mean_{paramx}.npy"))
                x_accepted = x_accepted[window[0]:window[1]]
                #x_accepted = x_accepted[:, [0, -1]]
                
            if paramy in adds:
                do_log[paramy] = True
                raw_fetched, mean_fetched = fetch(path, paramy)
                for p in raw_fetched:
                    raw_fetched[p] = raw_fetched[p][window[0]:window[1]]
                    mean_fetched[p] = mean_fetched[p][window[0]:window[1]]
                proposed, y_accepted = fetch_param(raw_fetched, mean_fetched, paramy, thickness=thickness)
            else:
                y_accepted = np.load(os.path.join(path, f"mean_{paramy}.npy"))
                y_accepted = y_accepted[window[0]:window[1]]
                #y_accepted = y_accepted[:, [0, -1]]
                
            # if did_multicore:
            #     x_accepted = x_accepted[:,window[0]:window[1]]
            #     y_accepted = y_accepted[:,window[0]:window[1]]
                
            # else:
            #     x_accepted = x_accepted[window[0]:window[1]]
            #     y_accepted = y_accepted[window[0]:window[1]]
                
            if clevels is not None:
                contour_info = calc_contours(x_accepted, y_accepted, clevels, (paramx, paramy),
                                             do_logx=do_log[paramx], do_logy=do_log[paramy],
                                             xrange=axis_overrides_2D[paramx], yrange=axis_overrides_2D[paramy])
            else:
                contour_info = None
                
                
            if do["2D_trackers"]:
                make_2D_tracker(x_accepted, y_accepted, markx=true_vals.get(paramx, None), marky=true_vals.get(paramy, None),
                                do_logx=do_log.get(paramx, None), do_logy=do_log.get(paramy, None),xlabel=stylized_names.get(paramx, None),ylabel=stylized_names.get(paramy, None),
                                contours=contour_info, did_multicore=did_multicore, axis_overrides=(axis_overrides_2D.get(paramx, None), axis_overrides_2D.get(paramy, None)),
                                )
                
            if do["2D_histos"]:
                make_2D_histo(x_accepted, y_accepted, markx=true_vals.get(paramx, None), marky=true_vals.get(paramy, None),
                              do_logx=do_log[paramx], do_logy=do_log[paramy],xlabel=paramx,ylabel=paramy,
                              contours=contour_info, did_multicore=did_multicore)

    if do["ASJD"]:
        means = load_all_accepted(path, names, is_active, do_log) # [param, chain, iter]
        
        if means.ndim == 2: means = np.expand_dims(means, 1)
        diffs = ASJD(means, window)
        print(diffs)
        print("Avg across chains: {}".format(np.mean(diffs[diffs != 0])))
        
        
    if do["ESS"]:
        min_ess = np.inf
        for i, param in enumerate(names):
            if not is_active.get(param, 0):
                continue
            
            print(f"#### {param} ####")
            accepted = np.load(os.path.join(path, f"mean_{param}.npy")) #[chain, iter]
            accepted = accepted[window[0]:window[1]]
            avg_ess = ESS(accepted,do_log[param])

            print("Average ESS: {}".format(avg_ess))
            min_ess = min(min_ess, avg_ess)
        print("Minimum ESS across params: {}".format(min_ess))
        
        for add_param in adds:
            print(f"#### {add_param} ####")
            raw_fetched, mean_fetched = fetch(path, add_param)
            for p in raw_fetched:
                raw_fetched[p] = raw_fetched[p][window[0]:window[1]]
                mean_fetched[p] = mean_fetched[p][window[0]:window[1]]
            proposed, accepted = fetch_param(raw_fetched, mean_fetched, add_param, thickness=thickness)
            
            avg_ess = ESS(accepted, recommend_logscale(add_param, do_log))
            print("Average ESS: {}".format(avg_ess))
            
            