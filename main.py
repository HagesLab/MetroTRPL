# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 22:03:46 2022

@author: cfai2
"""
import os
import sys

import numpy as np

from bayes_io import get_data, get_initpoints, read_config_script_file
from metropolis import metro

if __name__ == "__main__":
    # Some HiperGator specific stuff
    # Replace as needed for your setup
    on_hpg = 0
    if on_hpg:
        jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        script_head = sys.argv[1]
    else:
        jobid = 0
        script_head = "mcmc"

    logger_name = f"Ensemble{jobid}"
    script_path = f"{script_head}{jobid}.txt"
    export_path = f"CPU{jobid}-final.pik"

    try:
        sim_info, param_info, meas_fields, MCMC_fields = read_config_script_file(
            script_path)
    except Exception as e:
        print(e)
        sys.exit()
    np.random.seed(jobid)

    # Get observations and initial condition
    iniPar = get_initpoints(MCMC_fields["init_cond_path"], meas_fields)

    e_data = get_data(MCMC_fields["measurement_path"],
                      meas_fields, MCMC_fields)

    MS_list = metro(sim_info, iniPar, e_data, MCMC_fields, param_info, verbose=False,
               export_path=export_path, logger_name=logger_name)

    # Successful completion - remove all non-final checkpoints
    if "checkpoint_header" in MS_list.ensemble_fields:
        chpt_header = MS_list.ensemble_fields["checkpoint_header"]
        for chpt in os.listdir(MS_list.ensemble_fields["checkpoint_dirname"]):
            if (chpt.startswith(chpt_header)
                and not chpt.endswith("final.pik")
                and not chpt.endswith(".log")):
                os.remove(os.path.join(MS_list.ensemble_fields["checkpoint_dirname"], chpt))
        if len(os.listdir(MS_list.ensemble_fields["checkpoint_dirname"])) == 0:
            os.rmdir(MS_list.ensemble_fields["checkpoint_dirname"])

    output_path = MS_list.ensemble_fields["output_path"]
    print(f"{jobid} Finished - {output_path}")
