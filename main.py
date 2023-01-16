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
from metropolis import metro

if __name__ == "__main__":
    # Some HiperGator specific stuff
    on_hpg = 1
    if on_hpg:
        jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        script_head = sys.argv[1]
    else:
        jobid = 0
        script_head = "mcmc"
        
    script_path = f"{script_head}{jobid}.txt"

    
    try:
        sim_info, param_info, meas_fields, MCMC_fields = read_config_script_file(script_path)
    except Exception as e:
        print(e)
        sys.exit()
        
    logger, handler = start_logging(log_dir=MCMC_fields["output_path"], name=f"CPU{jobid}")
    
    
    if not os.path.isdir(MCMC_fields["checkpoint_dirname"]):
        os.makedirs(MCMC_fields["checkpoint_dirname"], exist_ok=True)

    # Reset (clear) checkpoints
    if MCMC_fields["load_checkpoint"] is None:
        for chpt in os.listdir(MCMC_fields["checkpoint_dirname"]):
            os.remove(os.path.join(MCMC_fields["checkpoint_dirname"], chpt))

    np.random.seed(jobid)

    if not os.path.isdir(MCMC_fields["output_path"]):
        os.makedirs(MCMC_fields["output_path"], exist_ok=True)

    logger.info("Sim info: {}".format(sim_info))
    logger.info("Measurement handling fields: {}".format(meas_fields))
    logger.info("Param infos: {}".format(param_info))
    logger.info("MCMC fields: {}".format(MCMC_fields))
    
    
    # Get observations and initial condition
    iniPar = get_initpoints(MCMC_fields["init_cond_path"], meas_fields)
    
    measurement_types = sim_info["meas_types"]
    scale_f = np.ones(len(measurement_types))
    for i, mt in enumerate(measurement_types):
        if mt == "TRPL":
            scale_f[i] = 1e-23
        elif mt == "TRTS":
            scale_f[i] = 1e-9
        else:
            raise NotImplementedError("No scale_f for measurements other than TRPL and TRTS")
    e_data = get_data(MCMC_fields["measurement_path"], meas_fields, MCMC_fields, scale_f=scale_f)
    
    clock0 = perf_counter()

    MS = metro(sim_info, iniPar, e_data, MCMC_fields, param_info, True, logger)

    final_t = perf_counter() - clock0

    logger.info("Metro took {} s".format(final_t))
    logger.info("Avg: {} s per iter".format(final_t / MCMC_fields["num_iters"]))
    logger.info("Acceptance rate: {}".format(np.sum(MS.H.accept) / len(MS.H.accept.flatten())))

    logger.info("Exporting to {}".format(MCMC_fields["output_path"]))
    MS.checkpoint(os.path.join(MCMC_fields["output_path"], f"CPU{jobid}-final.pik"))

    stop_logging(logger, handler, 0)
    output_path = MCMC_fields["output_path"]
    print(f"{jobid} Finished - {output_path}")
