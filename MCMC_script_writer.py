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
    on_hpg = 1
    try:
        jobid = int(sys.argv[1])
        script_head = sys.argv[2]
    except IndexError:
        jobid = 0
        script_head = "mcmc"

    if on_hpg:
        init_dir = r"/blue/c.hages/cfai2304/Metro_in"
        out_dir = r"/blue/c.hages/cfai2304/Metro_out"

    else:
        init_dir = r"trts_inputs"
        out_dir = r"trts_outputs"

    # Filenames
    init_fname = "staub_MAPI_power_thick_input.csv"
    exp_fname = "staub_MAPI_power_thick_withauger.csv"
    out_fname = "staub_pscan_with1OM_btwn_tntp_equalized"

    # Save this script to...
    script_path = f"{script_head}{jobid}.txt"

    np.random.seed(10000000*(jobid+1))
    
    # Info for each measurement's corresponding simulation
    num_measurements = 3
    #Length = [311,2000,311,2000, 311, 2000]
    Length  = [2000]*3             # Length (nm)
    L   = 128                                # Spatial points
    measurement_types = ["TRPL"]*3
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
                       "p0": 1.8 * 10 ** np.random.uniform(14, 16),
                       "mu_n": 20, #1.56 * 10 ** np.random.uniform(0, 2),
                       "mu_p": 20, #1.56 * 10 ** np.random.uniform(0, 2),
                       "ks": 6.1 * 10 ** np.random.uniform(-12, -10),
                       "Cn": 1.1 * 10 ** np.random.uniform(-29, -27),
                       "Cp": 1.1 * 10 ** np.random.uniform(-29, -27),
                       "Sf": 2.2 * 10 ** np.random.uniform(1, 3),
                       "Sb": 2.2 * 10 ** np.random.uniform(1, 3),
                       "tauN": 4.56 * 10 ** np.random.uniform(1,3),
                       "tauP": 4.56 * 10 ** np.random.uniform(1,3),
                       "eps":10,
                       "Tm":300,
                       "m":1}
    initial_guesses["tauP"] = initial_guesses["tauN"]

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
    meas_fields = {"time_cutoff":[0, 2000],
                   "select_obs_sets": None, #[0,1,2],
                   "noise_level":1e14}

    # Other MCMC control potions
    output_path = os.path.join(out_dir, out_fname)
    MCMC_fields = {"init_cond_path": os.path.join(init_dir, init_fname),
                   "measurement_path": os.path.join(init_dir, exp_fname),
                   "output_path": output_path,
                   "num_iters": 32000,
                   "solver": "solveivp",
                   "model_uncertainty": 1/2500*0.1,
                   "log_pl":1,
                   "self_normalize":0,
                   "proposal_function":"box", # box or gauss; anything else disables new proposals
                   "one_param_at_a_time":0,
                   "checkpoint_dirname": os.path.join(output_path, "Checkpoints"),
                   "checkpoint_header": f"CPU{jobid}",
                   "checkpoint_freq":15000, # Save a checkpoint every #this many iterations#
                   "load_checkpoint": None, #f"checkpointCPU{jobid}_30000.pik",
                   }
    
    generate_config_script_file(script_path, simPar, param_info, meas_fields, MCMC_fields, verbose=False)
