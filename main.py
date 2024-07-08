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

    # Get observations and initial condition
    iniPar = get_initpoints(MCMC_fields["init_cond_path"], meas_fields)

    e_data = get_data(MCMC_fields["measurement_path"],
                      meas_fields, MCMC_fields)

    metro(sim_info, iniPar, e_data, MCMC_fields, param_info, verbose=False,
          export_path=export_path, logger_name=logger_name,
          serial_fallback=True,
          )

    print(f"{jobid} Finished - {export_path}")
