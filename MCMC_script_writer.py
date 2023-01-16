# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 16:09:36 2023

Example script to generate MCMC configuration/scripting files.
Note the format of each field and whether it requires lists, dicts, or floats
Run this to generate a script that run the MCMC with specific settings.

@author: cfai2
"""
import numpy as np
import os
import sys
from bayes_io import generate_config_script_file

if __name__ == "__main__":
    # Just some HiperGator-specific stuff
    on_hpg = 0
    try:
        jobid = int(sys.argv[1])
    except IndexError:
        jobid = 0

    if on_hpg:
        init_dir = r"/blue/c.hages/cfai2304/Metro_in"
        out_dir = r"/blue/c.hages/cfai2304/Metro_out"

    else:
        init_dir = r"trts_inputs"
        out_dir = r"trts_outputs"

    # Filenames
    init_fname = "3B1FSGS_TRTS_input_new.csv"
    exp_fname = "3B1FSGS_TRTS_NEW.csv"
    out_fname = "DEBUG"

    # Save this script to...
    script_path = f"mcmc{jobid}.txt"
    
    # Info for each measurement's corresponding simulation
    num_measurements = 4
    Length = [311,2000,311,2000, 311, 2000]
    Length  = [3000, 3000, 3000, 3000]             # Length (nm)
    L   = 128                                # Spatial points
    measurement_types = ["TRTS", "TRTS", "TRTS", "TRTS"]
    simPar = {"lengths": Length, 
              "nx": L,
              "meas_types": measurement_types,
              "num_meas": num_measurements}

    # Info regarding the parameters
    # Here the global scale factor 'm' is also defined,
    # which will shift the simulation output by x10**m before calculating
    # likelihood vs measurement
    param_names = ["n0", "p0", "mu_n", "mu_p", "ks", "Cn", "Cp",
                   "Sf", "Sb", "tauN", "tauP", "eps", "Tm", "m"]
    
    unit_conversions = {"n0":((1e-7) ** 3), "p0":((1e-7) ** 3), 
                        "mu_n":((1e7) ** 2) / (1e9), "mu_p":((1e7) ** 2) / (1e9), 
                        "ks":((1e7) ** 3) / (1e9), 
                        "Cn":((1e7) ** 6) / (1e9), "Cp":((1e7) ** 6) / (1e9),
                        "Sf":1e-2, "Sb":1e-2, "Tm":1}
    do_log = {"n0":1, "p0":1,"mu_n":1,"mu_p":1,"ks":1, "Cn":1, "Cp":1,
              "Sf":1,"Sb":1,"tauN":1,"tauP":1,"eps":1,"Tm":1,
              "m":1}

    initial_guesses = {"n0":1e8,
                        "p0": 1e14,
                        "mu_n": 20,
                        "mu_p": 20,
                        "ks": 5.958e-11,
                        "Cn": 1e-29,
                        "Cp": 1e-29,
                        "Sf":2.1e2,
                        "Sb": 2.665e2,
                        "tauN": 4.708e2,
                        "tauP": 1.961e2,
                        "eps":10,
                        "Tm":300,
                        "m":1}

    active_params = {"n0":0,
                     "p0":1,
                     "mu_n":0,
                     "mu_p":0,
                     "ks":1,
                     "Cn":1,
                     "Cp":1,
                     "Sf":1,
                     "Sb":1,
                     "tauN":1,
                     "tauP":1,
                     "eps":0,
                     "Tm":0,
                     "m":0}

    # Proposal function search widths
    initial_variance = {"n0":1e-2,
                     "p0":1e-2,
                     "mu_n":1e-2,
                     "mu_p":1e-2,
                     "ks":1e-2,
                     "Cn":1e-2,
                     "Cp":1e-2,
                     "Sf":1e-2,
                     "Sb":1e-2,
                     "tauN":1e-2,
                     "tauP":1e-2,
                     "eps":1e-2,
                     "Tm":1e-2,
                     "m":1e-2}

    param_info = {"names":param_names,
                  "active":active_params,
                  "unit_conversions":unit_conversions,
                  "do_log":do_log,
                  "init_guess":initial_guesses,
                  "init_variance":initial_variance}

    # Measurement preprocessing options
    meas_fields = {"time_cutoff":[0, np.inf],
                "select_obs_sets": None, #[0,1,2],
                "noise_level":None}

    # Other MCMC control potions
    # TODO: Decide what to do with these. Some unusued, some outdated. Should they be mandatory?
    output_path = os.path.join(out_dir, out_fname)
    MCMC_fields = {"init_cond_path": os.path.join(init_dir, init_fname),
                 "measurement_path": os.path.join(init_dir, exp_fname),
                 "output_path": output_path,
                 "num_iters": 10,
                 "solver": "solveivp",
                 "anneal_params": [1/2500*100, 1e3, 1/2500*0.1], # [Unused, unused, initial_T]
                 "override_equal_mu":0,
                 "override_equal_s":0,
                 "log_pl":1,
                 "self_normalize":1,
                 "proposal_function":"box", # box or gauss; anything else disables new proposals
                 "one_param_at_a_time":0,
                 "checkpoint_dirname": os.path.join(output_path, "Checkpoints"),
                 "checkpoint_header": f"CPU{jobid}",
                 "checkpoint_freq":5, # Save a checkpoint every #this many iterations#
                 "load_checkpoint": None,
                 }
    
    generate_config_script_file(script_path, simPar, param_info, meas_fields, MCMC_fields, verbose=False)
    from bayes_io import read_config_script_file
    print(read_config_script_file("mcmc0.txt"))