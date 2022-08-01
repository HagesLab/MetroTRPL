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
    init_dir = r"/blue/c.hages/cfai2304/Metro_in"
    out_dir = r"/blue/c.hages/cfai2304/Metro_out"

    init_fname = sys.argv[4]
    exp_fname = sys.argv[3]
    out_fname = sys.argv[2]

else:
    init_dir = r"bay_inputs"
    out_dir = r"bay_outputs"

    init_fname = "staub_MAPI_power_thick_input.csv"
    exp_fname = "staub_MAPI_power_thick.csv"
    #exp_fname = "abrupt_p0.csv"
    out_fname = "DEBUG2"


init_pathname = os.path.join(init_dir, init_fname)
experimental_data_pathname = os.path.join(init_dir, exp_fname)
out_pathname = os.path.join(out_dir, out_fname)
if not os.path.isdir(out_pathname):
    try:
        os.mkdir(out_pathname)
    except FileExistsError:
        print(f"{out_pathname} already exists")

if on_hpg:
    jobid = os.getenv('SLURM_ARRAY_TASK_ID')
    assert jobid is not None, "Error: use a job array script"
    jobid = int(jobid)
    out_pathname = os.path.join(out_pathname, f"{jobid}")
else:
    try:
        jobid = int(sys.argv[2])
    except IndexError:
        jobid = 0

def start_logging(log_dir="Logs"):

    if not os.path.isdir(log_dir):
        try:
            os.mkdir(log_dir)
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
from metropolis import metro, draw_initial_guesses
from time import perf_counter

if __name__ == "__main__":
    logger, handler = start_logging(log_dir=os.path.join(out_pathname))
    # Set space and time grid options
    #Length = [311,2000,311,2000, 311, 2000]
    Length  = 2000                            # Length (nm)
    L   = 2 ** 7                                # Spatial points
    plT = 1                                  # Set PL interval (dt)
    pT  = (0,1,3,10,30,100)                   # Set plot intervals (%)
    tol = 7                                   # Convergence tolerance
    MAX = 10000                                  # Max iterations

    simPar = [Length, -1, L, -1, plT, pT, tol, MAX]

    # This code follows a strict order of parameters:
    # matPar = [N0, P0, DN, DP, rate, sr0, srL, tauN, tauP, Lambda, mag_offset]
    # Set the parameter ranges/sample space
    param_names = ["n0", "p0", "mu_n", "mu_p", "ks", 
                   "Sf", "Sb", "tauN", "tauP", "eps", "m"]
    unit_conversions = {"n0":((1e-7) ** 3), "p0":((1e-7) ** 3), 
                        "mu_n":((1e7) ** 2) / (1e9), "mu_p":((1e7) ** 2) / (1e9), 
                        "ks":((1e7) ** 3) / (1e9), "Sf":1e-2, "Sb":1e-2}
    do_log = {"n0":1, "p0":1,"mu_n":1,"mu_p":1,"ks":1,
              "Sf":1,"Sb":1,"tauN":1,"tauP":1,"eps":1,
              "m":0}

    initial_guesses = {"n0":1e8,
                        "p0": 3e15,
                        "mu_n": 20,
                        "mu_p": 20,
                        "ks": 4.8e-11,
                        "Sf":10,
                        "Sb": 10,
                        "tauN": 511,
                        "tauP": 871,
                        "eps":10,
                        "m":0}
    mods = [{"p0":3e14}, {"p0":3e16}, {"ks":4.8e-12}, {"ks":4.8e-10},
            {"mu_n":2}, {"mu_n":200}, {"mu_p":2}, {"mu_p":200},
            {"Sf":1}, {"Sf":100}, {"Sb":1}, {"Sb":100},
            {"tauN":51.1}, {"tauN":5110}, {"tauP":87.1}, {"tauP":8710}]
    for p, v in mods[jobid].items():
        initial_guesses[p] = v

    active_params = {"n0":0,
                     "p0":1,
                     "mu_n":1,
                     "mu_p":1,
                     "ks":1,
                     "Sf":1,
                     "Sb":1,
                     "tauN":1,
                     "tauP":1,
                     "eps":0,
                     "m":0}

    #initial_guesses = {"n0":1e8,
    #                 "p0":[1e10,1e12,1e14,1e16],
    #                 "mu_n":20,
    #                 "mu_p":20,
    #                 "ks":1.585e-12,
    #                 "Sf":1.585e-1,
    #                 "Sb":1e-1,
    #                 "tauN":1e4,
    #                 "tauP":3.415,
    #                 "eps":10,
    #                 "m":0}
    # Other options
    initial_variance = {"n0":0,
                     "p0":1,
                     "mu_n":0.1,
                     "mu_p":0.1,
                     "ks":1,
                     "Sf":1,
                     "Sb":1,
                     "tauN":1,
                     "tauP":1,
                     "eps":0,
                     "m":0}

    initial_variance = 1
    param_info = {"names":param_names,
                  "active":active_params,
                  "unit_conversions":unit_conversions,
                  "do_log":do_log,}

    ic_flags = {"time_cutoff":2000,
                "select_obs_sets": None,
                "noise_level":None}

    # TODO: Validation
    sim_flags = {"num_iters": 100000,
                 "anneal_mode": "exp", # None, "exp", "log"
                 "anneal_params": [1/2500*100, 1e4, 1/2500*0.1], # [Initial T; time constant (exp decreases by 63% when t=, log decreases by 50% when 2t=; minT]
                 "delayed_acceptance": 'off', # "off", "on", "cumulative", "DEBUG"
                 "DA time subdivisions": 1,
                 "override_equal_mu":False,
                 "override_equal_s":False,
                 "log_pl":True,
                 "self_normalize":False,
                 "num_initial_guesses":4,
                 "proposal_function":"box", # box or gauss; anything else disables new proposals
                 "adaptive_covariance":"None", #AM for Harrio Adaptive, LAP for Shaby Log-Adaptive
                 "AM_activation_time":0,
                 "one_param_at_a_time":True,
                 "LAP_params":(1,0.8,0.234),
                 "checkpoint_dirname": os.path.join(out_pathname, "Checkpoints"),
                 "checkpoint_freq":10000, # Save a checkpoint every #this many iterations#
                 "load_checkpoint": None,
                 }

    if not os.path.isdir(sim_flags["checkpoint_dirname"]):
        os.mkdir(sim_flags["checkpoint_dirname"])

    # Reset (clear) checkpoints
    if sim_flags["load_checkpoint"] is None:
        for chpt in os.listdir(sim_flags["checkpoint_dirname"]):
            os.remove(os.path.join(sim_flags["checkpoint_dirname"], chpt))

    if on_hpg:
        logger.info("Array job detected, ID={}".format(jobid))
    else:
        logger.info(f"Not array job, using ID={jobid}")


    np.random.seed(jobid)
    #param_is_iterable = {param:isinstance(initial_guesses[param], (list, tuple, np.ndarray)) for param in initial_guesses}
    #for param in initial_guesses:
    #    if param_is_iterable[param]:
    #        np.random.shuffle(initial_guesses[param])

    #if not sim_flags.get("do_multicore", False) and any(param_is_iterable.values()):
    #    logger.warning("Multiple initial guesses detected without do_multicore - doing only first guess"
    #                    "- did you mean to enable do_multicore?")

    #initial_guess_list = draw_initial_guesses(initial_guesses, sim_flags["num_initial_guesses"])

    if not os.path.isdir(out_pathname):
        os.mkdir(out_pathname)

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
    e_data = get_data(experimental_data_pathname, ic_flags, sim_flags, scale_f=1e-23)
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
