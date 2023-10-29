from time import perf_counter
import sys
import os
sys.path.append(os.path.join(".", "Dense_Sample"))

import numpy as np

from mcmc_logging import start_logging, stop_logging
from bayes_io import get_data, get_initpoints, read_config_script_file
from bayes_io import insert_param
from dense_sampling import bayes, export

if __name__ == "__main__":
    on_hpg = 0
    if on_hpg:
        jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        script_head = sys.argv[1]
    else:
        jobid = 0
        script_head = "mcmc"

    script_path = os.path.join(f"{script_head}{jobid}.txt")

    try:
        sim_info, param_info, meas_fields, MCMC_fields = read_config_script_file(
            script_path)
    except Exception as e:
        print(e)
        sys.exit()
    np.random.seed(jobid)

    logger, handler = start_logging(
        log_dir=MCMC_fields["output_path"], name=f"CPU{jobid}")

    # Get observations and initial condition
    iniPar = get_initpoints(MCMC_fields["init_cond_path"], meas_fields)

    measurement_types = sim_info["meas_types"]

    e_data = get_data(MCMC_fields["measurement_path"], measurement_types,
                      meas_fields, MCMC_fields)

    # If fittable fluences, use the initial condition to setup fittable fluence parameters
    # (Only if the initial condition supplies fluences instead of the entire profile)
    if MCMC_fields.get("fittable_fluences", None) is not None:
        if len(iniPar[0]) != sim_info["nx"][0]:
            insert_param(param_info, MCMC_fields, mode="fluences")
        else:
            logger.warning("No fluences found in Input file - fittable_fluences ignored!")
            MCMC_fields["fittable_fluences"] = None

    if MCMC_fields.get("fittable_absps", None) is not None:
        if len(iniPar[0]) != sim_info["nx"][0]:
            insert_param(param_info, MCMC_fields, mode="absorptions")
        else:
            logger.warning("No absorptions found in Input file - fittable_absps ignored!")
            MCMC_fields["fittable_absps"] = None

    # Make simulation info consistent with actual number of selected measurements
    if meas_fields.get("select_obs_sets", None) is not None:
        sim_info["meas_types"] = [sim_info["meas_types"][i]
                                  for i in meas_fields["select_obs_sets"]]
        sim_info["lengths"] = [sim_info["lengths"][i]
                               for i in meas_fields["select_obs_sets"]]
        sim_info["num_meas"] = len(meas_fields["select_obs_sets"])
        if MCMC_fields.get("irf_convolution", None) is not None:
            MCMC_fields["irf_convolution"] = [MCMC_fields["irf_convolution"][i]
                                              for i in meas_fields["select_obs_sets"]]

    logger.info("Measurement handling fields: {}".format(meas_fields))
    logger.info("E data: {}".format(
        ["[{}...{}]".format(e_data[1][i][0], e_data[1][i][-1]) for i in range(len(e_data[1]))]))
    logger.info("Initial condition: {}".format(
        ["[{}...{}]".format(iniPar[i][0], iniPar[i][-1]) for i in range(len(iniPar))]))

    N = np.array([0])
    P = None
    clock0 = perf_counter()
    N, P, X = bayes(N, P, iniPar, sim_info, e_data, MCMC_fields, param_info, logger=logger)
    logger.info("Bayesim took {} s".format(perf_counter() - clock0))

    clock0 = perf_counter()
    export(os.path.join(MCMC_fields["output_path"], f"CPU{jobid}"), P, X, logger=logger)
    logger.info("Export took {}".format(perf_counter() - clock0))
    stop_logging(logger, handler, 0)

    output_path = MCMC_fields["output_path"]
    print(f"{jobid} Finished - {output_path}")