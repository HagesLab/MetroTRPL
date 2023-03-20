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
        sim_info, param_info, meas_fields, MCMC_fields = read_config_script_file(
            script_path)
    except Exception as e:
        print(e)
        sys.exit()
    np.random.seed(jobid)

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
            raise NotImplementedError(
                "No scale_f for measurements other than TRPL and TRTS")
    e_data = get_data(MCMC_fields["measurement_path"],
                      meas_fields, MCMC_fields, scale_f=scale_f)

    # Make simulation info consistent with actual number of selected measurements
    if meas_fields.get("select_obs_sets", None) is not None:
        sim_info["meas_types"] = [sim_info["meas_types"][i]
                                  for i in meas_fields["select_obs_sets"]]
        sim_info["lengths"] = [sim_info["lengths"][i]
                               for i in meas_fields["select_obs_sets"]]
        sim_info["num_measurements"] = len(meas_fields["select_obs_sets"])

    logger, handler = start_logging(
        log_dir=MCMC_fields["output_path"], name=f"CPU{jobid}")
    logger.info("Measurement handling fields: {}".format(meas_fields))
    logger.info("E data: {}".format(e_data))
    logger.info("Initial condition: {}".format(
        ["[{}...{}]".format(iniPar[i][0], iniPar[i][-1]) for i in range(len(iniPar))]))

    export_path = f"CPU{jobid}-final.pik"

    clock0 = perf_counter()
    MS = metro(sim_info, iniPar, e_data, MCMC_fields, param_info, verbose=True,
               export_path=export_path, logger=logger)

    final_t = perf_counter() - clock0

    logger.info("Metro took {} s ({} hr)".format(final_t, final_t / 3600))
    logger.info("Avg: {} s per iter".format(final_t / MCMC_fields["num_iters"]))
    logger.info("Acceptance rate: {}".format(
        np.sum(MS.H.accept) / len(MS.H.accept.flatten())))

    # Successful completion - checkpoints not needed anymore
    chpt_header = MS.MCMC_fields["checkpoint_header"]
    for chpt in os.listdir(MS.MCMC_fields["checkpoint_dirname"]):
        if f"checkpoint{chpt_header}" in chpt:
            os.remove(os.path.join(MS.MCMC_fields["checkpoint_dirname"], chpt))
    # os.rmdir(MS.MCMC_fields["checkpoint_dirname"])

    stop_logging(logger, handler, 0)

    output_path = MCMC_fields["output_path"]
    print(f"{jobid} Finished - {output_path}")
