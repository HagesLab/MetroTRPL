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
    # Set up jobid, script_head, init_dir, out_dir, etc... depending on your computer
    on_hpg = 0
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
        init_dir = r"Inputs"
        out_dir = r"bay_outputs"

    # Filenames
    init_fname = "staub_MAPI_threepower_twothick_fluences.csv"
    exp_fname = "staub_MAPI_threepower_twothick_nonoise.csv"
    out_fname = f"PT_staubnn_8T_10swap_scaled_mpi"

    # Save this script to...
    script_path = f"{script_head}{jobid}.txt"

    np.random.seed(100000000*(jobid+1))

    # Info for each measurement's corresponding simulation
    num_measurements = 6
    Length = [311, 2000, 311, 2000, 311, 2000] # in nm
    L = [128] * 6                         # Spatial points
    measurement_types = ["TRPL"] * 6
    simPar = {"lengths": Length,
              "nx": L,
              "meas_types": measurement_types,
              "num_meas": num_measurements}

    # Info regarding the parameters
    param_names = ["n0", "p0", "mu_n", "mu_p", "ks", "Cn", "Cp",
                   "Sf", "Sb", "tauN", "tauP", "eps", "Tm"]

    unit_conversions = {"n0": ((1e-7) ** 3), "p0": ((1e-7) ** 3),
                        "mu_n": ((1e7) ** 2) / (1e9),
                        "mu_p": ((1e7) ** 2) / (1e9),
                        "ks": ((1e7) ** 3) / (1e9),
                        "Cn": ((1e7) ** 6) / (1e9),
                        "Cp": ((1e7) ** 6) / (1e9),
                        "Sf": 1e-2, "Sb": 1e-2, "Tm": 1,
                        }

    do_log = {"n0": 1, "p0": 1, "mu_n": 1, "mu_p": 1, "ks": 1, "Cn": 1, "Cp": 1,
              "Sf": 1, "Sb": 1, "tauN": 1, "tauP": 1, "eps": 1, "Tm": 1
              }

    prior_dist = {"n0": (0, np.inf),
                  "p0": (1e14, 1e16),
                  "mu_n": (1e0, 1e2),
                  "mu_p": (1e0, 1e2),
                  "ks": (1e-11, 1e-9),
                  "Cn": (1e-29, 1e-27),
                  "Cp": (1e-29, 1e-27),
                  "Sf": (1e-1, 1e4),
                  "Sb": (1e-1, 1e4),
                  "tauN": (1, 2e3),
                  "tauP": (1, 1e4),
                  "eps": (0, np.inf),
                  "Tm": (0, np.inf),
                  }

    initial_guesses = {"n0": 1e8,
                       "p0": 3e15,
                       "mu_n": 20,
                       "mu_p": 20,
                       "ks": 4.8e-11,
                       "Cn": 4.4e-29,
                       "Cp": 4.4e-29,
                       "Sf": 10,
                       "Sb": 10,
                       "tauN": 511,
                       "tauP": 871,
                       "eps": 10,
                       "Tm": 300,
                       }

    active_params = {"n0": 0,
                     "p0": 1,
                     "mu_n": 1,
                     "mu_p": 1,
                     "ks": 1,
                     "Cn": 1,
                     "Cp": 1,
                     "Sf": 1,
                     "Sb": 1,
                     "tauN": 1,
                     "tauP": 1,
                     "eps": 0,
                     "Tm": 0,
                     }
    # Proposal function search widths
    trial_move = {param: 0.003 for param in param_names}

    # Randomize the initial guess a little
    for name in param_names:
        if active_params[name]:
            initial_guesses[name] *= 10 ** np.random.uniform(-0.5, 0.5)

    param_info = {"names": param_names,
                  "active": active_params,
                  "unit_conversions": unit_conversions,
                  "do_log": do_log,
                  "prior_dist": prior_dist,
                  "init_guess": initial_guesses,
                  "trial_move": trial_move}

    # Measurement preprocessing options
    meas_fields = {"time_cutoff": [0, 2000],
                   "select_obs_sets": None,  # e.g. [0, 2, 4] to select only 311nm curves
                   }

    # Other MCMC control potions
    output_path = os.path.join(out_dir, out_fname)
    MCMC_fields = {"init_cond_path": os.path.join(init_dir, init_fname),
                   "measurement_path": os.path.join(init_dir, exp_fname),
                   "output_path": output_path,
                   "num_iters": 25,
                   "solver": ("solveivp",),
                   "model": "std",
                   "likel2move_ratio": 333.33,
                   "log_y": 1,
                   "scale_factor": None,
                   "irf_convolution": None,
                   "parallel_tempering": [1, 2, 4, 10, 16, 32],
                   "temper_freq": 10,
                   "hard_bounds": 1,
                   "force_min_y": 0,
                   "checkpoint_freq": 2000,
                   "load_checkpoint": None,
                   }

    generate_config_script_file(script_path, simPar, param_info,
                                meas_fields, MCMC_fields, verbose=True)
