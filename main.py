# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 22:03:46 2022

@author: cfai2
"""
import numpy as np
import os
import pickle
import sys
import logging

from bayes_io import get_data, get_initpoints
from metropolis import metro, start_metro_controller
from time import perf_counter


if __name__ == "__main__":
    
    # Set space and time grid options
    Length = [311,2000,311,2000, 311, 2000]
    #Length  = 2000                            # Length (nm)
    L   = 2 ** 7                                # Spatial points
    plT = 1                                  # Set PL interval (dt)
    pT  = (0,1,3,10,30,100)                   # Set plot intervals (%)
    tol = 7                                   # Convergence tolerance
    MAX = 10000                                  # Max iterations
    
    simPar = [Length, -1, L, -1, plT, pT, tol, MAX]
    
    # This code follows a strict order of parameters:
    # matPar = [N0, P0, DN, DP, rate, sr0, srL, tauN, tauP, Lambda, mag_offset]
    # Set the parameter ranges/sample space
    param_names = ["n0", "p0", "mu_n", "mu_p", "B", 
                   "Sf", "Sb", "tauN", "tauP", "eps", "m"]
    unit_conversions = {"n0":((1e-7) ** 3), "p0":((1e-7) ** 3), 
                        "mu_n":((1e7) ** 2) / (1e9), "mu_p":((1e7) ** 2) / (1e9), 
                        "B":((1e7) ** 3) / (1e9), "Sf":1e-2, "Sb":1e-2}
    do_log = {"n0":1, "p0":1,"mu_n":0,"mu_p":0,"B":1,
              "Sf":1,"Sb":1,"tauN":0,"tauP":0,"eps":1,
              "m":0}
    
    num_initial_guesses = 8
    initial_guesses = {"n0":1e8, 
                        "p0":3e15, 
                        "mu_n":20, 
                        "mu_p":20, 
                        "B":4.8e-11, 
                        "Sf":np.logspace(-1, 2, 8), 
                        "Sb":np.logspace(-1, 2, 8), 
                        "tauN":np.linspace(100, 1000, 8), 
                        "tauP":np.linspace(100, 1000, 8), 
                        "eps":10, 
                        "m":0}
    
    active_params = {"n0":0, 
                     "p0":0, 
                     "mu_n":0, 
                     "mu_p":0, 
                     "B":0, 
                     "Sf":1, 
                     "Sb":1, 
                     "tauN":1, 
                     "tauP":1, 
                     "eps":0, 
                     "m":0}
    # Other options
    ic_flags = {"time_cutoff":None,
                "select_obs_sets": None,
                "noise_level":1e14}
    
    gpu_info = {"sims_per_gpu": 2 ** 13,
                "num_gpus": 8}
    
    # TODO: Validation
    sim_flags = {"num_iters": 50,
                 "delayed_acceptance": 'on', # "off", "on", "cumulative"
                 "DA time subdivisions": [10, 50],
                 "override_equal_mu":False,
                 "override_equal_s":False,
                 "log_pl":True,
                 "self_normalize":False,
                 "do_multicore":False
                 }
    
    if sim_flags.get("do_multicore", False):
        param_is_iterable = {param:isinstance(initial_guesses[param], (list, tuple, np.ndarray)) for param in initial_guesses}
        param_infos = []
        for ig in range(num_initial_guesses):
            initial_guess = {}
            for param in initial_guesses:
                if param_is_iterable[param]:
                    initial_guess[param] = initial_guesses[param][ig]
                else:
                    initial_guess[param] = initial_guesses[param]
                    
            param_info = {"names":param_names,
                          "active":active_params,
                          "unit_conversions":unit_conversions,
                          "do_log":do_log,
                          "initial_guess":initial_guess}
        
            param_infos.append(param_info)
    
    else:
        param_is_iterable = {param:isinstance(initial_guesses[param], (list, tuple, np.ndarray)) for param in initial_guesses}
        if any(param_is_iterable.values()):
            logging.warning("Multiple initial guesses detected without do_multicore, taking only first guess "
                         "- did you mean to enable do_multicore?")
        initial_guess = {}
        for param in initial_guesses:
            if param_is_iterable[param]:
                initial_guess[param] = initial_guesses[param][0]
            else:
                initial_guess[param] = initial_guesses[param]
                
        param_infos = {"names":param_names,
                      "active":active_params,
                      "unit_conversions":unit_conversions,
                      "do_log":do_log,
                      "initial_guess":initial_guess}

    
    # Collect filenames
    try:
        on_hpg = sys.argv[1]
    except IndexError:
        on_hpg = False

    if on_hpg:
        init_dir = r"/blue/c.hages/cfai2304/Metro_in"
        out_dir = r"/blue/c.hages/cfai2304/Metro_out"
    else:
        init_dir = r"bay_inputs"
        out_dir = r"bay_outputs"

    init_fname = "staub_MAPI_threepower_twothick_input.csv"
    exp_fname = "staub_MAPI_threepower_twothick_nonoise.csv"
    out_fname = "DEBUG2"
    init_pathname = os.path.join(init_dir, init_fname)
    experimental_data_pathname = os.path.join(init_dir, exp_fname)
    out_pathname = os.path.join(out_dir, out_fname)
    
    if not os.path.isdir(out_pathname):
        os.mkdir(out_pathname)
    
    with open(os.path.join(out_pathname, "param_info.pik"), "wb+") as ofstream:
        pickle.dump(param_infos, ofstream)
    
    # Get observations and initial condition
    iniPar = get_initpoints(init_pathname, ic_flags)
    e_data = get_data(experimental_data_pathname, ic_flags, sim_flags, scale_f=1e-23)
    clock0 = perf_counter()
    if sim_flags.get("do_multicore", False):
        history = start_metro_controller(simPar, iniPar, e_data, sim_flags, param_infos)
    else:
        history = metro(simPar, iniPar, e_data, sim_flags, param_infos)
    
    final_t = perf_counter() - clock0
    print("Metro took {} s".format(final_t))
    print("Avg: {} s per iter".format(final_t / sim_flags["num_iters"]))
    print("Acceptance rate:", np.sum(history.accept) / len(history.accept.flatten()))
    
    history.export(param_infos, out_pathname)
