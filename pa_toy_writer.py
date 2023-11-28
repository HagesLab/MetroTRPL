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
        init_dir = r"Inputs"
        out_dir = r"bay_outputs"

    else:
        init_dir = r"Inputs"
        out_dir = r"bay_outputs"

    # Filenames
    init_fname = "staub_MAPI_threepower_twothick_fluences.csv"
    exp_fname = "staub_MAPI_threepower_twothick_nonoise.csv"
    out_fname = "PA_highT"

    # Save this script to...
    script_path = f"{script_head}{jobid}.txt"

    np.random.seed(100000000*(jobid+1))

    # Info for each measurement's corresponding simulation
    num_measurements = 1
    Length = [1] # in nm
    L = [1] * 1                         # Spatial points
    measurement_types = ["pa"] * 1
    simPar = {"lengths": Length,
              "nx": L,
              "meas_types": measurement_types,
              "num_meas": num_measurements}

    # Info regarding the parameters
    # Here the global scale factor 'm' is also defined,
    # which will shift the simulation output by x10**m before calculating
    # likelihood vs measurement
    param_names = ["x"]

    unit_conversions = {}

    do_log = {"x": 0,
              }

    prior_dist = {"x": (-np.inf, np.inf),
                  }

    initial_guesses = {"x": [-1.9, -1.5, -1, -0.5, 0, 0.5, 1.5][jobid],
                       }

    active_params = {"x": 1,
                     }
    # Proposal function search widths
    initial_variance = {param: 0.1 for param in param_names}



    param_info = {"names": param_names,
                  "active": active_params,
                  "unit_conversions": unit_conversions,
                  "do_log": do_log,
                  "prior_dist": prior_dist,
                  "init_guess": initial_guesses,
                  "init_variance": initial_variance}

    # Measurement preprocessing options
    meas_fields = {"time_cutoff": [0, 2000],
                   "select_obs_sets": None,  # e.g. [0, 2, 4] to select only 311nm curves
                   }

    # Other MCMC control potions
    output_path = os.path.join(out_dir, out_fname)
    MCMC_fields = {"init_cond_path": os.path.join(init_dir, init_fname),
                   "measurement_path": os.path.join(init_dir, exp_fname),
                   "output_path": output_path,
                   "num_iters": 100000,
                   "solver": ("solveivp",),
                   "model": "pa",
                   "likel2variance_ratio": 50,
                   "log_pl": 0,
                   "self_normalize": None,
                   "scale_factor": None,
                   "fittable_fluences": None,
                   "irf_convolution": None,
                   "proposal_function": "box",
                   "one_param_at_a_time": 0,
                   "hard_bounds": 1,
                   "force_min_y": 0,
                   "checkpoint_freq": 12000,
                   "load_checkpoint": None,
                   }

    generate_config_script_file(script_path, simPar, param_info,
                                meas_fields, MCMC_fields, verbose=True)
