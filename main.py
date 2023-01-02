# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 22:03:46 2022

@author: cfai2
"""
import numpy as np
import os
import sys
from time import perf_counter

from mcmc_logging import start_logging, stop_logging
from bayes_io import get_data, get_initpoints, read_config_script_file
from bayes_io import validate_grid
from metropolis import metro

if __name__ == "__main__":
    # Some HiperGator specific stuff
    on_hpg = 0
    try:
        jobid = int(sys.argv[1])
    except IndexError:
        jobid = 0
    
    script_path = "mcmc0.txt"

    
    try:
        simPar, param_info, ic_flags, sim_flags = read_config_script_file(script_path)
    except Exception as e:
        print(e)
        sys.exit()
        
    logger, handler = start_logging(log_dir=sim_flags["output_path"])
    
    
    if not os.path.isdir(sim_flags["checkpoint_dirname"]):
        os.makedirs(sim_flags["checkpoint_dirname"], exist_ok=True)

    # Reset (clear) checkpoints
    if sim_flags["load_checkpoint"] is None:
        for chpt in os.listdir(sim_flags["checkpoint_dirname"]):
            os.remove(os.path.join(sim_flags["checkpoint_dirname"], chpt))

    np.random.seed(0)

    if not os.path.isdir(sim_flags["output_path"]):
        os.makedirs(sim_flags["output_path"], exist_ok=True)

    logger.info("Sim info: {}".format(simPar))
    logger.info("Measurement handling fields: {}".format(ic_flags))
    logger.info("Param infos: {}".format(param_info))
    logger.info("MCMC fields: {}".format(sim_flags))
    
    
    # Get observations and initial condition
    iniPar = get_initpoints(sim_flags["init_cond_path"], ic_flags)
    
    measurement_types = simPar["meas_types"]
    scale_f = np.ones(len(measurement_types))
    for i, mt in enumerate(measurement_types):
        if mt == "TRPL":
            scale_f[i] = 1e-23
        elif mt == "TRTS":
            scale_f[i] = 1e-9
        else:
            raise NotImplementedError("No scale_f for measurements other than TRPL and TRTS")
    e_data = get_data(sim_flags["measurement_path"], ic_flags, sim_flags, scale_f=scale_f)
    
    clock0 = perf_counter()

    MS = metro(simPar, iniPar, e_data, sim_flags, param_info, True, logger)

    final_t = perf_counter() - clock0

    logger.info("Metro took {} s".format(final_t))
    logger.info("Avg: {} s per iter".format(final_t / sim_flags["num_iters"]))
    logger.info("Acceptance rate: {}".format(np.sum(MS.H.accept) / len(MS.H.accept.flatten())))

    logger.info("Exporting to {}".format(sim_flags["output_path"]))
    MS.checkpoint(os.path.join(sim_flags["output_path"], f"CPU{jobid}-final.pik"))

    stop_logging(logger, handler, 0)
    output_path = sim_flags["output_path"]
    print(f"{jobid} Finished - {output_path}")
