# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 19:11:50 2022

@author: cfai2
"""

from stitch_arrayjob_outputs import join_chains
from metro_plot_lib import make_1D_tracker, make_2D_tracker, make_1D_histo, make_2D_histo
from metro_plot_lib import plot_history
from postprocessing_utils import fetch_param, recommend_logscale, calc_contours
from postprocessing_utils import package_all_accepted, ASJD, ESS, binned_stderr, geweke, gelman
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
sys.path.append("..")


matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams.update({'font.family': 'sans-serif'})
matplotlib.rcParams.update({'mathtext.fontset': 'stixsans'})


stylized_names = {"p0": r"$p_\mathrm{o}$", "ks": r"$k^*$",
                  "mu_n": r"$\mu_n$", "mu_p": r"$\mu_p$",
                  "mu_eff": r"$\mu\prime$",
                  "Sf+Sb": r"$S_\mathrm{F}+S_\mathrm{B}$", "Sf": r"$S_\mathrm{F}$",
                  "Sb": r"$S_\mathrm{B}$", "tauN": r"$\tau_n$",
                  "tauP": r"$\tau_p$", "tau_srh": r"$\tau_{SRH}$",
                  "HI_tau_srh": r"$\tau_{SRH}$",
                  "Cn+Cp": r"$C_n+C_p$",
                  "tauN+tauP": r"$\tau_n+\tau_p$"}

units = {"p0": r"$\mathrm{cm^{-3}}$", "ks": r"$\mathrm{cm^{3}\ s^{-1}}$",
         "mu_n": r"$\mathrm{cm^{2}\ V^{-1}\ s^{-1}}$",
         "mu_p": r"$\mathrm{cm^{2}\ V^{-1}\ s^{-1}}$",
         "mu_eff": r"$\mathrm{cm^{2}\ V^{-1}\ s^{-1}}$",
         "Cn": r"$\mathrm{cm^{6}\ s^{-1}}$", "Cp": r"$\mathrm{cm^{6}\ s^{-1}}$",
         "Sf+Sb": r"$\mathrm{cm\ s^{-1}}$", "Sf": r"$\mathrm{cm\ s^{-1}}$",
         "Sb": r"$\mathrm{cm\ s^{-1}}$",
         "tauN": r"$\mathrm{ns}$", "tauP": r"$\mathrm{ns}$",
         "tau_srh": r"$\mathrm{ns}$",
         "HI_tau_srh": r"$\mathrm{ns}$",
         "Cn+Cp": r"$\mathrm{cm^{6}\ s^{-1}}$", "tauN+tauP": r"$\mathrm{ns}$"}

true_vals = {"mu_n": 90,  # 320,
             "mu_p": 45,  # 40,
             "mu_eff": 20,  # 71.1,
             "p0": 3e15,  # 1e13,
             "ks": 4.8e-11,  # 2e-10,
             "Sf": 1e5,  # 100,
             "Sb": 1e3,  # 1e4,
             "Cn": 4.4e-29,
             "Cp": 4.4e-29,
             "Cn+Cp": 8.8e-29,
             "tauN": 511,  # 1,
             "tauP": 871,  # 1,
             "tau_eff": 454,
             "HI_tau_srh": 1290,
             "Sf+Sb": 20}
true_vals = {}

do = {"big_grid": 0,
      "1D_trackers": 1,
      "2D_trackers": 0,
      "1D_histos": 1,
      "2D_histos": 0,
      "ASJD": 0,
      "ESS": 0,
      "diffusion": 0,
      "Geweke": 0,
      "Gelman": 0,
      "binning": 1}

axis_overrides_2D = {"mu_n": (1e0, 1e3),
                     "mu_p": (1e0, 1e3),
                     "mu_eff": (1e0, 1e3),
                     "p0": (3e14, 3e16),
                     "ks": (4.8e-12, 4.8e-10),
                     "Sf": (1e4, 1e6),
                     "Sb": (1e0, 1e2),
                     "tauN": (1e2, 2e4),
                     "tauP": (1e2, 2e4),
                     "tau_srh": (1e0, 1e3),
                     "Sf+Sb": (1e0, 1e2),
                     "tauN+tauP": (1e3, 1e5),
                     "Cn": (1e-30, 1e-28),
                     "Cp": (1e-30, 1e-27),
                     "Cn+Cp": (1e-29, 1e-27)}

thickness = 2000  # [nm]
axis_overrides_1D = {"mu_n": (6e0, 6e2),
                     "mu_p": (6e0, 6e2),
                     "p0": (1e14, 1e16),
                     "ks": (1e-11, 2e-10),
                     "Cn": (1e-29, 1e-27),
                     "Cp": (1e-29, 1e-27),
                     "Sf": (1e0, 1e2),
                     "Sb": (1e0, 1e2),
                     "tauN": (480, 540),
                     "tauP": (1e2, 3e3),
                     "Sf+Sb": (15, 25),
                     "HI_tau_SRH": (1e2, 1e4),
                     "tau_srh": (5e1, 1.2e3),
                     }

axis_overrides_1D = {}

window = [0000, 8000]
adds = {}
pairs = [('mu_n', 'mu_p', 130), ('mu_eff', 'Sf', None)]
# pairs = [('tauN', 'tauP', None)]
#pairs = [("Sf+Sb", "tauN"), ]
adds = {"Cn+Cp": None, "HI_tau_srh": None}
adds = {"Sf+Sb": 20, "Cn+Cp": 8.8e-29}
#adds = {"tau_srh": None, "Cn+Cp": 16.6e-29, "HI_tau_srh": None}
#adds = {"Cn+Cp": 8.8e-29, "mu_eff": 20}
pairs = [('Cn', 'Cp', 16.6e-29), ("Sf+Sb", "tauN", 485), ("Sf+Sb", "tauN+tauP", 1311)]
#pairs = [("Sf", "Sb", 20)]


if __name__ == "__main__":
    path = os.path.join(
        "..", r"bay_outputs\table5_but_early")

    exclude = [1, 2, 3]
    join_chains(path, exclude=exclude)

    with open(os.path.join(path, "Ffinal.pik"), 'rb') as ifstream:
        MS = pickle.load(ifstream)

    param_info = MS.param_info
    sim_flags = MS.sim_flags

    # This is deprecated and should come from MS
    # with open(os.path.join(path, "sim_flags.pik"), "rb") as ifstream:
    #     sim_flags = pickle.load(ifstream)

    # with open(os.path.join(path, "Ffinal.pik"), 'rb') as ifstream:
    #     MS2 = pickle.load(ifstream)

    # sim_flags = MS2.sim_flags

    names = param_info["names"]
    is_active = param_info["active"]
    do_log = param_info["do_log"]
    binning = do["binning"]

    accept_history = MS.H.accept
    if accept_history.ndim == 1:
        accept_history = np.expand_dims(accept_history, 0)
    accept_history = accept_history[:, window[0]:window[1]]

    loglikelihood = MS.H.loglikelihood
    if loglikelihood.ndim == 1:
        loglikelihood = np.expand_dims(loglikelihood, 0)
    loglikelihood = loglikelihood[:, window[0]:window[1]]

    m_binning = 100  # Mandatory binning to be able to track acceptance%
    bins = np.arange(0, len(accept_history[0]), int(m_binning))
    accept_sub_means, accept_stderr = binned_stderr(accept_history, bins[1:])
    loglikelihood_sub_means, loglikelihood_stderr = binned_stderr(
        loglikelihood, bins[1:])

    
    # Number the remaining chains
    labels = []
    l_ct = 0
    while len(labels) != len(accept_sub_means):
        if l_ct not in exclude:
            labels.append(l_ct)
        l_ct += 1

    #fig, ax = plt.subplots(1, 1, figsize=(3, 3*2.417/2.185), dpi=600)
    #fig, ax = plt.subplots(1, 1, figsize=(1.875, 1.875*1.763/1.548*1.763/1.696), dpi=600)
    fig, ax = plt.subplots(1, 1, figsize=(2.8, 2.5), dpi=600)
    do_l = abs(np.log10(min(-loglikelihood_sub_means.flatten())) -
               np.log10(max(-loglikelihood_sub_means.flatten()))) > 1
    
    # For LL plot Figure 2
    # plot_history(bins + 0.5*m_binning, loglikelihood_sub_means * 10,
    #               xlabel=r"$n$", ylabel=r"ln $L(\Omega_n)$", do_log=False,
    #               fig=fig, ax=ax, break_region = (-7e4, -5200), ylim=(-200000, -5170),
    #               yticks = [-100000, -5190, -5180], yticklabels= [r'-10$^5$', -5190, -5180])
    plot_history((bins + 0.5*m_binning) / 1e3, loglikelihood_sub_means,
                  xlabel=r"$N\ /\ 10^3$", ylabel=r"ln $L(\Omega_n)$", do_log=do_l,
                  fig=fig, ax=ax)
    # colors=['k', '#1f77ba', 'r', '#2ca02c']
    plot_history(bins + 0.5*m_binning, accept_sub_means,
                 xlabel=r"$n$", ylabel="Acceptance rate")

    # Apply window
    for i, param in enumerate(names):
        x = getattr(MS.H, param)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        setattr(MS.H, param, x[:, window[0]:window[1]])

        x = getattr(MS.H, f"mean_{param}")
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        setattr(MS.H, f"mean_{param}", x[:, window[0]:window[1]])
        
    if do["diffusion"]:
        num_active = sum(is_active.values())    
        num_samples = len(accept_history[0])
        diffusion_coef = np.zeros(len(accept_history))
        
        for i, param in enumerate(names):
            if not is_active[param]:
                continue
            
            step_delta = np.array([[1e-1, 5e-2, 4e-2, 3e-2, 2e-2, 1e-2,
                                    5e-3, 2e-3, 1e-3, 5e-4, 1e-4, 1e-5]]).T
            #step_delta = param_info["init_variance"][param]
            print(step_delta)
            x = getattr(MS.H, f"mean_{param}")
            if do_log[param]: 
                x = np.log10(x)
            x = np.diff(x, axis=1)
            x = x ** 2
            diffusion_coef += np.sum(x, axis=1)
    
        diffusion_coef /= (num_samples * num_active)
        print("Diffusion coef:", diffusion_coef)


    if do["1D_trackers"]:
        first = False
        plot_counter = 0
        for i, param in enumerate(names):
            if not is_active.get(param, 0):
                continue

            fig, ax = plt.subplots(1, 1, figsize=(
                3, 3), dpi=600, squeeze=False)

            # j = plot_counter // 5
            # k = plot_counter % 5
            j = 0
            k = 0

            proposed = getattr(MS.H, param)
            accepted = getattr(MS.H, f"mean_{param}")
            # if do_log[param]:
            #     accepted = np.log10(accepted)

            # nt = len(accepted[0])
            # trunc_nt = int(nt * 0.9)

            # autoc = np.zeros((len(accepted), trunc_nt))
            # var = np.var(accepted, axis=1)
            # mean = np.mean(accepted, axis=1)
            # for t in range(trunc_nt):
            #     autoc[:, t] = np.mean((np.roll(accepted, -t, axis=1)[:, :nt-t] - mean) * (accepted[:, :nt-t] - mean))
            # autoc /= var
            # autoc = np.abs(autoc)
            # #autoc = np.cumsum(autoc, axis=1)

            # if do_log[param]:
            #     accepted = 10 ** accepted

            # binnings = [1,2,4,8,16,32,64,80,100,125,250,500,1000]
            # xxxx = []
            # yyyy = []

            # #print("\nParam | # bins | (log) Mean | var of bins | var / # bins")
            # if do_log[param]: accepted = np.log10(accepted)
            # for binning in binnings:
            #     if binning is not None and binning > 0:
            #         accepted_n = accepted[0, ::binning]
            #         stderr = np.std(accepted_n, ddof=1)
            #         xxxx.append(len(accepted_n))
            #         yyyy.append(stderr**2 / len(accepted_n))

            #         #bins = np.arange(0, len(accepted[0]), int(binning))
            #         #accepted_n, stderr = binned_stderr(accepted, bins[1:])
            #         #x = bins + 0.5 * binning
            #         #num_bins = len(accepted[0]) // binning
            #         #np.savetxt("{}-{}bins.csv".format(param, num_bins), np.vstack((np.arange(len(accepted_n[0])), accepted_n[0])).T, delimiter=",")
            #         #print(param, num_bins, "bins", np.mean(accepted_n), stderr[0]**2, stderr[0]**2 / num_bins)
            #         #xxxx.append(num_bins)
            #         #yyyy.append(stderr[0]**2)
            #     else:
            #         x = np.arange(len(accepted))

            # plt.figure(1000*(i+1))
            # plt.scatter(xxxx, yyyy)

            # plt.title(param)
            # plt.ylabel("var of sampled states ")
            # plt.xlabel("# sampled states")
            # plt.xscale("log")

            if binning is not None and binning > 0:
                bins = np.arange(0, len(accepted[0]), int(binning))
                if do_log[param]:
                    accepted = np.log10(accepted)
                accepted, stderr = binned_stderr(accepted, bins[1:])
                accepted = 10 ** accepted
                x = bins + 0.5 * binning
            else:
                x = np.arange(len(accepted))

            # make_1D_tracker(np.arange(trunc_nt), autoc, mark_value=None,
            #                 ylabel="{} (acf)".format(stylized_names.get(
            #                     param, param)),
            #                 show_legend=first, do_log=False,
            #                 ylim=None,
            #                 fig=fig, ax=ax[j, k])

            make_1D_tracker(x, accepted, mark_value=true_vals.get(param, None),
                            ylabel="{} [{}]".format(stylized_names.get(
                                param, param), units.get(param, "a.u.")),
                            show_legend=first, do_log=do_log[param],
                            ylim=axis_overrides_1D.get(param, None),
                            fig=fig, ax=ax[j, k])

            plot_counter += 1
            fig.tight_layout()

        for ii, add_param in enumerate(adds):

            # j = (plot_counter) // 5
            # k = (plot_counter) % 5
            fig, ax = plt.subplots(1, 1, figsize=(
                3, 3.0), dpi=600, squeeze=False)
            j = 0
            k = 0

            proposed, accepted = fetch_param(MS, add_param, thickness=thickness)

            recommended_log = recommend_logscale(add_param, do_log)

            # if recommended_log:
            #     accepted = np.log10(accepted)

            # nt = len(accepted[0])
            # trunc_nt = int(nt * 0.9)

            # autoc = np.zeros((len(accepted), trunc_nt))
            # var = np.var(accepted, axis=1)
            # mean = np.mean(accepted, axis=1)
            # for t in range(trunc_nt):
            #     autoc[:, t] = np.mean((np.roll(accepted, -t, axis=1)[:, :nt-t] - mean) * (accepted[:, :nt-t] - mean))
            # autoc /= var
            # autoc = np.abs(autoc)
            # #autoc = np.cumsum(autoc, axis=1)

            # if recommended_log:
            #     accepted = 10 ** accepted

            # binnings = [1,2,4,8,16,32,64,80,100,125,250,500,1000]
            # xxxx = []
            # yyyy = []

            # print("\nParam | # bins | (log) Mean | var of bins | var / # bins")
            # if recommended_log: accepted = np.log10(accepted)
            # for binning in binnings:
            #     if binning is not None and binning > 0:
            #         bins = np.arange(0, len(accepted[0]), int(binning))
            #         accepted_n, stderr = binned_stderr(accepted, bins[1:])
            #         x = bins + 0.5 * binning
            #         num_bins = len(accepted[0]) // binning
            #         np.savetxt("{}-{}bins.csv".format(add_param, num_bins), np.vstack((np.arange(len(accepted_n[0])), accepted_n[0])).T, delimiter=",")
            #         print(add_param, num_bins, "bins", np.mean(accepted_n), stderr[0]**2, stderr[0]**2 / num_bins)
            #         xxxx.append(num_bins)
            #         yyyy.append(stderr[0]**2)
            #     else:
            #         x = np.arange(len(accepted))

            # plt.figure(1000*(i+1))
            # plt.scatter(xxxx, yyyy)
            # plt.title(add_param)
            # plt.ylabel("var of bin means")
            # plt.xlabel("# bins")
            # plt.xscale("log")

            if binning is not None and binning > 0:
                bins = np.arange(0, len(accepted[0]), int(binning))
                if do_log:
                    accepted = np.log10(accepted)
                accepted, stderr = binned_stderr(accepted, bins[1:])
                accepted = 10 ** accepted
                x = bins + 0.5 * binning

            else:
                x = np.arange(len(accepted))

            # make_1D_tracker(np.arange(trunc_nt), autoc, mark_value=None,
            #                 ylabel="{} (acf)".format(stylized_names.get(
            #                     add_param, add_param)),
            #                 show_legend=first, do_log=False,
            #                 ylim=None,
            #                 fig=fig, ax=ax[j, k])

            make_1D_tracker(x, accepted, mark_value=adds.get(add_param, None), ylabel=stylized_names.get(add_param, add_param),
                            show_legend=first, do_log=recommended_log,
                            ylim=axis_overrides_1D.get(add_param, None),
                            fig=fig, ax=ax[j, k])
            plot_counter += 1
            fig.tight_layout()

    if do["1D_histos"]:
        for i, param in enumerate(names):
            if not is_active.get(param, 0):
                continue
            
            try:
                accepted = getattr(MS.H, f"mean_{param}")

                # make_1D_histo(accepted, mark_value=true_vals.get(param, None),
                #               xlabel="{} [{}]".format(stylized_names.get(
                #                   param, param), units.get(param, "a.u.")),
                #               ylabel="P({})".format(stylized_names.get(param, param)),
                #               do_log=0, bin_count=256, axis_overrides=axis_overrides_1D.get(param, None),
                #               show_legend=False, size=(7.5/4, 7.5/4))
            except Exception as e:
                print(f"Histo {param}", e)
                continue

            print(param, " mean ", np.mean(accepted))
            print(param, " std ", np.std(accepted))

        for add_param in adds:
            proposed, accepted = fetch_param(MS, add_param, thickness=thickness)

            recommended_log = recommend_logscale(add_param, do_log)
            try:
                # make_1D_histo(accepted, mark_value=adds.get(add_param, None),
                #               xlabel="{} [{}]".format(stylized_names.get(
                #                   add_param, add_param), units.get(add_param, "a.u.")),
                #               ylabel="P({})".format(
                #                   stylized_names.get(add_param, add_param)),
                #               do_log=0, bin_count=256, axis_overrides=axis_overrides_1D.get(add_param, None),
                #               show_legend=False, size=(7.5/4, 7.5/4))
                pass
            except Exception as e:
                print(f"Histo {add_param}", e)
                continue

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

            try:
                x_accepted = getattr(MS.H, f"mean_{paramx}")

            except AttributeError:
                do_log[paramx] = True
                proposed, x_accepted = fetch_param(
                    MS, paramx, thickness=thickness)
                do_log[paramx] = recommend_logscale(paramx, do_log)

            try:
                y_accepted = getattr(MS.H, f"mean_{paramy}")

            except AttributeError:
                do_log[paramy] = True
                proposed, y_accepted = fetch_param(
                    MS, paramy, thickness=thickness)

            if clevels is not None:
                contour_info = calc_contours(x_accepted, y_accepted, clevels, (paramx, paramy),
                                             do_logx=do_log[paramx], do_logy=do_log[paramy],
                                             xrange=axis_overrides_2D.get(paramx, None), yrange=axis_overrides_2D.get(paramy, None))
            else:
                contour_info = None

            if do["2D_trackers"]:
                make_2D_tracker(x_accepted, y_accepted, markx=true_vals.get(paramx, None), marky=true_vals.get(paramy, None),
                                do_logx=do_log.get(paramx, None), do_logy=do_log.get(paramy, None),
                                xlabel="{} [{}]".format(stylized_names.get(
                                    paramx, paramx), units.get(paramx, "a.u.")),
                                ylabel="{} [{}]".format(stylized_names.get(
                                    paramy, paramy), units.get(paramy, "a.u.")),
                                contours=contour_info, axis_overrides=(axis_overrides_2D.get(
                                    paramx, None), axis_overrides_2D.get(paramy, None)),
                                size=(2, 2)
                                )

            if do["2D_histos"]:
                make_2D_histo(x_accepted, y_accepted, markx=true_vals.get(paramx, None), marky=true_vals.get(paramy, None),
                              do_logx=do_log[paramx], do_logy=do_log[paramy], xlabel=paramx, ylabel=paramy,
                              contours=contour_info, axis_overrides=(axis_overrides_2D.get(paramx, None), axis_overrides_2D.get(paramy, None)))

    if do["big_grid"]:
        active = [i for i in is_active if is_active[i]]
        active += list(adds.keys())
        bigfig, bigax = plt.subplots(len(active), len(
            active), figsize=(2*len(active), 2*len(active)), dpi=400)
        for i, py in enumerate(active):
            for j, px in enumerate(active):
                if i < j:
                    bigax[i, j].axis('off')

                elif i == j:
                    if py in adds:
                        proposed, accepted = fetch_param(
                            MS, py, thickness=thickness)

                        recommended_log = recommend_logscale(py, do_log)
                        make_1D_histo(accepted, mark_value=adds.get(py, None),
                                      xlabel="{} [{}]".format(stylized_names.get(
                                          py, py), units.get(py, "a.u.")),
                                      ylabel="P({})".format(
                                          stylized_names.get(py, py)),
                                      do_log=1, bin_count=96, axis_overrides=axis_overrides_1D.get(py, None),
                                      fig=bigfig, ax=bigax[i, j],
                                      show_legend=False, size=(2, 2))

                    else:
                        accepted = getattr(MS.H, f"mean_{py}")
                        make_1D_histo(accepted, mark_value=true_vals.get(py, None),
                                      xlabel="{} [{}]".format(stylized_names.get(
                                          py, py), units.get(py, "a.u.")),
                                      ylabel="P({})".format(
                                          stylized_names.get(py, py)),
                                      do_log=1, bin_count=96, axis_overrides=axis_overrides_1D.get(py, None),
                                      fig=bigfig, ax=bigax[i, j],
                                      show_legend=False, size=(2, 2))

                else:

                    if px in adds:
                        proposed, x_accepted = fetch_param(
                            MS, px, thickness=thickness)
                        do_log[px] = recommend_logscale(px, do_log)
                    else:
                        x_accepted = getattr(MS.H, f"mean_{px}")

                    if py in adds:
                        proposed, y_accepted = fetch_param(
                            MS, py, thickness=thickness)
                        do_log[py] = recommend_logscale(py, do_log)
                    else:
                        y_accepted = getattr(MS.H, f"mean_{py}")

                    contour_info = None

                    make_2D_tracker(x_accepted, y_accepted, markx=true_vals.get(px, None), marky=true_vals.get(py, None),
                                    do_logx=do_log.get(px, None), do_logy=do_log.get(py, None),
                                    xlabel="{} [{}]".format(stylized_names.get(
                                        px, px), units.get(px, "a.u.")),
                                    ylabel="{} [{}]".format(stylized_names.get(
                                        py, py), units.get(py, "a.u.")),
                                    contours=contour_info, axis_overrides=(
                                        axis_overrides_2D.get(px, None), axis_overrides_2D.get(py, None)),
                                    fig=bigfig, ax=bigax[i, j]
                                    )

        bigfig.tight_layout(pad=0.1)

    if do["ASJD"]:
        means = package_all_accepted(
            MS, names, is_active, do_log)  # [param, chain, iter]
        diffs = ASJD(means)
        print(diffs)
        print("Avg across chains: {}".format(np.mean(diffs[diffs != 0])))

    if do["ESS"]:
        min_ess = np.inf
        for i, param in enumerate(names):
            if not is_active.get(param, 0):
                continue

            print(f"#### {param} ####")
            accepted = getattr(MS.H, f"mean_{param}")  # [chain, iter]
            total_ess = ESS(accepted, do_log[param])

            print("ESS: {}".format(total_ess))

        for add_param in adds:
            print(f"#### {add_param} ####")
            proposed, accepted = fetch_param(MS, add_param, thickness=thickness)

            total_ess = ESS(accepted, recommend_logscale(add_param, do_log))
            print("ESS: {}".format(total_ess))

    if do["Geweke"]:

        for j in range(sim_flags["num_chains"]):
            print(f"Chain {j}")
            for i, param in enumerate(names):
                if not is_active.get(param, 0):
                    continue

                print(f"#### {param} ####")
                accepted = getattr(MS.H, f"mean_{param}")  # [chain, iter]
                # TODO: Handle 2D instead of downcasting

                try:
                    g = geweke(accepted[j], do_log[param])
                except Exception as e:
                    print(e)
                    g = np.inf

                print("Geweke z-score: {}".format(g))

            for add_param in adds:
                print(f"#### {add_param} ####")
                proposed, accepted = fetch_param(
                    MS, add_param, thickness=thickness)
                try:
                    g = geweke(accepted[j], recommend_logscale(
                        add_param, do_log))
                except Exception as e:
                    print(e)
                    g = np.inf

                print("Geweke z-score: {}".format(g))

    if do["Gelman"]:

        for i, param in enumerate(names):
            if not is_active.get(param, 0):
                continue

            print(f"#### {param} ####")
            accepted = getattr(MS.H, f"mean_{param}")  # [chain, iter]

            try:
                g = gelman(accepted, do_log[param])
            except Exception as e:
                print(e)
                g = np.inf

            print("Gelman-Rubin score: {}".format(g))

        for add_param in adds:
            print(f"#### {add_param} ####")
            proposed, accepted = fetch_param(MS, add_param, thickness=thickness)
            try:
                g = gelman(accepted, recommend_logscale(add_param, do_log))
            except Exception as e:
                print(e)
                g = np.inf

            print("Gelman-Rubin score: {}".format(g))

    plt.show()
