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
from datetime import datetime

# Collect filenames
try:
    on_hpg = int(sys.argv[1])
except IndexError:
    on_hpg = False
except ValueError:
    print("First CL arg should be 1 for HPG and 0 for desktop")
    sys.exit(1)

if on_hpg:
    jobid = os.getenv('SLURM_ARRAY_TASK_ID')
    assert jobid is not None, "Error: use a job array script"
    jobid = int(jobid)
    
else:
    try:
        jobid = int(sys.argv[2])
    except IndexError:
        jobid = 3

if on_hpg:
    init_dir = r"/blue/c.hages/cfai2304/Metro_in"
    out_dir = r"/blue/c.hages/cfai2304/Metro_out"

    init_fname = sys.argv[4]
    exp_fname = sys.argv[3]
    out_fname = sys.argv[2]

else:
    init_dir = r"trts_inputs"
    out_dir = r"trts_outputs"
    #init_dir = r"bay_inputs"
    #out_dir = r"bay_outputs"
    init_fname = "mocktrts_cdte_800nm_input.csv"
    exp_fname = "mocktrts_cdte_800nm.csv"
    #exp_fname = "abrupt_p0.csv"
    #init_fname = "2A1FSGS_input.csv"
    #exp_fname = "2A1FSGS.csv"
    #init_fnames = {0:"2A1FSGS_TRPL_input.csv",
    #               1:"3B1FSGS_TRPL_input.csv",
    #               2:"staub_MAPI_threepower_twothick_input.csv",
    #               3:"staub_MAPI_power_input.csv"}
    #init_fname = init_fnames[jobid]

    #exp_fnames = {0:"2A1FSGS_TRPL.csv",
    #              1:"3B1FSGS_TRPL.csv",
    #              2:"staub_MAPI_threepower_twothick_withauger.csv",
    #              3:"real_staub_aug_corr.csv"}
    #exp_fname = exp_fnames[jobid]

    out_fnames = {0:"THEORY_TAUONLY",
                  1:"THEORY_KP0",
                  2:"THEORY_MUONLY",
                  3:"THEORY_SONLY"}
    out_fname = out_fnames[jobid]

init_pathname = os.path.join(init_dir, init_fname)
experimental_data_pathname = os.path.join(init_dir, exp_fname)
out_pathname = os.path.join(out_dir, out_fname)
if not os.path.isdir(out_pathname):
    try:
        os.makedirs(out_pathname, exist_ok=True)
    except FileExistsError:
        print(f"{out_pathname} already exists")

        
out_pathname = os.path.join(out_pathname, f"{jobid}")

def start_logging(log_dir="Logs"):

    if not os.path.isdir(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except FileExistsError:
            pass

    tstamp = str(datetime.now()).replace(":", "-")
    #logging.basicConfig(filename=os.path.join(log_dir, f"{tstamp}.log"), filemode='a', level=logging.DEBUG)
    logger = logging.getLogger("Metro Logger Main")
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(os.path.join(log_dir, f"{tstamp}.log"))
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
            fmt='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
            )

    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger, handler

def stop(logger, handler, err=0):
    if err:
        logger.error(f"Termining with error code {err}")

    # Spyder needs explicit handler handling for some reason
    logger.removeHandler(handler)
    logging.shutdown()
    return

from bayes_io import get_data, get_initpoints
from metropolis import metro
from time import perf_counter

if __name__ == "__main__":
    logger, handler = start_logging(log_dir=os.path.join(out_pathname))
    # Set space and time grid options
    #Length = [311,2000,311,2000, 311, 2000]
    Length  = 3000                           # Length (nm)
    L   = 300                                # Spatial points
    plT = 1                                  # Set PL interval (dt)
    pT  = (0,1,3,10,30,100)                   # Set plot intervals (%)
    tol = 7                                   # Convergence tolerance
    MAX = 10000                                  # Max iterations

    simPar = [Length, -1, L, -1, plT, pT, tol, MAX]

    # This code follows a strict order of parameters:
    # matPar = [N0, P0, DN, DP, rate, Cn, Cp, sr0, srL, tauN, tauP, Lambda, mag_offset]
    # Set the parameter ranges/sample space
    param_names = ["n0", "p0", "mu_n", "mu_p", "ks", "Cn", "Cp",
                   "Sf", "Sb", "tauN", "tauP", "eps", "Tm", "m"]
    unit_conversions = {"n0":((1e-7) ** 3), "p0":((1e-7) ** 3), 
                        "mu_n":((1e7) ** 2) / (1e9), "mu_p":((1e7) ** 2) / (1e9), 
                        "ks":((1e7) ** 3) / (1e9), 
                        "Cn":((1e7) ** 6) / (1e9), "Cp":((1e7) ** 6) / (1e9),
                        "Sf":1e-2, "Sb":1e-2, "Tm":1}
    do_log = {"n0":1, "p0":1,"mu_n":1,"mu_p":1,"ks":1, "Cn":1, "Cp":1,
              "Sf":1,"Sb":1,"tauN":1,"tauP":1,"eps":1,"Tm":0,
              "m":0}

    initial_guesses = {"n0":1e8,
                        "p0": 1e14 if jobid == 1 else 1e13,
                        "mu_n": 100 if jobid == 2 else 320,
                        "mu_p": 100 if jobid == 2 else 40,
                        "ks": 4.8e-11 if jobid == 1 else 2e-10,
                        "Cn": 1e-99,
                        "Cp": 1e-99,
                        "Sf":1000 if jobid == 3 else 100,
                        "Sb": 100 if jobid == 3 else 1e4,
                        "tauN": 10 if jobid == 0 else 1,
                        "tauP": 10 if jobid == 0 else 1,
                        "eps":9.4,
                        "Tm":300,
                        "m":0}

    active_params = {"n0":0,
                     "p0":1 if jobid == 1 else 0,
                     "mu_n":1 if jobid == 2 else 0,
                     "mu_p":1 if jobid == 2 else 0,
                     "ks":1 if jobid == 1 else 0,
                     "Cn":0,
                     "Cp":0,
                     "Sf":1 if jobid == 3 else 0,
                     "Sb":1 if jobid == 3 else 0,
                     "tauN":1 if jobid == 0 else 0,
                     "tauP":1 if jobid == 0 else 0,
                     "eps":0,
                     "Tm":0,
                     "m":0}

    # Other options
    initial_variance = {"n0":0,
                     "p0":1,
                     "mu_n":0.1,
                     "mu_p":0.1,
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

    initial_variance = 1e-1

    param_info = {"names":param_names,
                  "active":active_params,
                  "unit_conversions":unit_conversions,
                  "do_log":do_log,}

    ic_flags = {"time_cutoff":2000,
                "select_obs_sets": None,
                "noise_level":None}

    # TODO: Validation
    sim_flags = {"measurement":"TRTS",
                 "num_iters": 50000,
                 "solver": "solveivp",
                 "use_multi_cpus":False,
                 "rtol":1e-7,
                 "atol":1e-10,
                 "hmax":0.1,
                 "verify_hmax":False,
                 "anneal_params": [1/2500*100, 1e3, 1/2500*0.1], # [Unused, unused, initial_T]
                 "override_equal_mu":False,
                 "override_equal_s":False,
                 "log_pl":True,
                 "self_normalize":False,
                 "proposal_function":"box", # box or gauss; anything else disables new proposals
                 "one_param_at_a_time":False,
                 "checkpoint_dirname": os.path.join(out_pathname, "Checkpoints"),
                 "checkpoint_freq":10000, # Save a checkpoint every #this many iterations#
                 "load_checkpoint": None,
                 }

    if not os.path.isdir(sim_flags["checkpoint_dirname"]):
        os.makedirs(sim_flags["checkpoint_dirname"], exist_ok=True)

    # Reset (clear) checkpoints
    if sim_flags["load_checkpoint"] is None:
        for chpt in os.listdir(sim_flags["checkpoint_dirname"]):
            os.remove(os.path.join(sim_flags["checkpoint_dirname"], chpt))

    if on_hpg:
        logger.info("Array job detected, ID={}".format(jobid))
    else:
        logger.info(f"Not array job, using ID={jobid}")


    np.random.seed(0)

    if not os.path.isdir(out_pathname):
        os.makedirs(out_pathname, exist_ok=True)

    with open(os.path.join(out_pathname, "param_info.pik"), "wb+") as ofstream:
        pickle.dump(param_info, ofstream)

    with open(os.path.join(out_pathname, "sim_flags.pik"), "wb+") as ofstream:
        pickle.dump(sim_flags, ofstream)

    logger.info("Length: {}".format(Length))
    logger.info("Init_fname: {}".format(init_fname))
    logger.info("Exp_fname: {}".format(exp_fname))
    logger.info("Out_fname: {}".format(out_pathname))
    logger.info("IC flags: {}".format(ic_flags))
    logger.info("Param infos: {}".format(param_info))
    logger.info("Sim flags: {}".format(sim_flags))
    logger.info("Initial variances: {}".format(initial_variance))

    # Get observations and initial condition
    iniPar = get_initpoints(init_pathname, ic_flags)
    if sim_flags["measurement"] == "TRPL":
        scale_f = 1e-23
    elif sim_flags["measurement"] == "TRTS":
        scale_f = 1e-9
    else:
        raise NotImplementedError("No scale_f for measurements other than TRPL and TRTS")
    e_data = get_data(experimental_data_pathname, ic_flags, sim_flags, scale_f=scale_f)
    clock0 = perf_counter()

    logger.info("Initial guess: {}".format(initial_guesses))
    history = metro(simPar, iniPar, e_data, sim_flags, param_info, initial_variance, True, logger, initial_guesses)

    final_t = perf_counter() - clock0

    logger.info("Metro took {} s".format(final_t))
    logger.info("Avg: {} s per iter".format(final_t / sim_flags["num_iters"]))
    logger.info("Acceptance rate: {}".format(np.sum(history.accept) / len(history.accept.flatten())))

    logger.info("Exporting to {}".format(out_pathname))
    history.export(param_info, out_pathname)

    stop(logger, handler, 0)
    print(f"{jobid} Finished")
