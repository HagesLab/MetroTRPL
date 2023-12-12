# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 22:13:26 2022

@author: cfai2
"""
import os
import signal
import pickle
from time import perf_counter
import numpy as np

from mcmc_logging import start_logging
from sim_utils import Ensemble
from bayes_io import make_dir
from laplace import make_I_tables

# Constants
MSG_FREQ = 100
MSG_COOLDOWN = 3 # Log first few states regardless of verbose


def roll_acceptance(rng : np.random.Generator, logratio):
    if isinstance(logratio, np.ndarray):
        return rng.random(len(logratio)) < np.exp(logratio)
    else:
        return rng.random() < np.exp(logratio)


def almost_equal(x, x0, threshold=1e-10):
    if x.shape != x0.shape:
        return False

    return np.abs(np.nanmax((x - x0) / x0)) < threshold


def main_metro_loop(MS_list : Ensemble, starting_iter, num_iters,
                    need_initial_state=True, verbose=False):
    """
    Run the Metropolis loop for each chain in an Ensemble()
    over a specified number of iterations,
    storing all info in MetroState() objects and saving the Ensemble()
    as occasional checkpoints.

    Parameters
    ----------
    MS_list : Ensemble() object

    starting_iter : int
        Index of first iter. 1 if starting from scratch, checkpoint-dependent
        otherwise.
    num_iters : int
        Index of final iter.
    need_initial_state : bool, optional
        Whether we're starting from scratch, and thus need to calculate the
        likelihood of the initial state before moving around.
        The default is True.
    logger : logging object, optional
        Stream to write status messages. The default is None.
    verbose : bool, optional
        Print more detailed status messages. The default is False.

    Returns
    -------
    None. Ensemble() and MetroStates is updated throughout.

    """
    checkpoint_freq = MS_list.ensemble_fields["checkpoint_freq"]

    if need_initial_state:
        MS_list.logger.info("Simulating initial state:")
        # Calculate likelihood of initial guess
        for m, MS in enumerate(MS_list.MS):
            logll, ll_funcs = MS_list.eval_trial_move(MS_list.H.states[m, :, 0], MS.MCMC_fields)
            MS_list.H.loglikelihood[m, 0] = logll
            MS_list.ll_funcs[m] = ll_funcs

    for k in range(starting_iter, num_iters):
        try:
            if k % MSG_FREQ == 0 or k < starting_iter + MSG_COOLDOWN:
                MS_list.logger.info("#####")
                MS_list.logger.info(f"Iter {k}")

            for m, MS in enumerate(MS_list.MS):
                if k % MSG_FREQ == 0 or k < starting_iter + MSG_COOLDOWN:
                    MS_list.logger.info(f"MetroState #{m}")

                # Trial displacement move
                new_state = MS_list.select_next_params(MS_list.H.states[m, :, k-1],
                                                       MS_list.MS[m].MCMC_fields["_T"] ** 0.5 * MS_list.ensemble_fields["base_trial_move"])

                logll, ll_funcs = MS_list.eval_trial_move(new_state, MS.MCMC_fields)

                MS_list.logger.debug(f"Log likelihood of proposed move: {logll}")
                logratio = (logll - MS_list.H.loglikelihood[m, k-1])
                if np.isnan(logratio):
                    logratio = -np.inf

                accepted = roll_acceptance(MS_list.RNG, logratio)

                if accepted:
                    MS_list.H.loglikelihood[m, k] = logll
                    MS_list.H.states[m, :, k] = new_state
                    MS_list.H.accept[m, k] = 1
                    MS_list.ll_funcs[m] = ll_funcs
                else:
                    MS_list.H.loglikelihood[m, k] = MS_list.H.loglikelihood[m, k-1]
                    MS_list.H.states[m, :, k] = MS_list.H.states[m, :, k-1]

            if MS_list.ensemble_fields["do_parallel_tempering"] and k % MS_list.ensemble_fields["temper_freq"] == 0:
                for _ in range(len(MS_list.MS) - 1):
                    # Select a pair (the ith and (i+1)th) of chains
                    i = MS_list.RNG.integers(0, len(MS_list.MS)-1)
                    # Do a tempering move between (swap the positions of) the ith and (i+1)th chains
                    if k % MSG_FREQ == 0 or k < starting_iter + MSG_COOLDOWN:
                        MS_list.logger.info(f"Tempering - swapping chains {i} and {i+1}")
                    T_j = MS_list.MS[i+1].MCMC_fields["_T"]
                    T_i = MS_list.MS[i].MCMC_fields["_T"]

                    bi_ui, bj_ui, bi_uj, bj_uj = 0, 0, 0, 0
                    for ss in range(MS_list.sim_info["num_meas"]):
                        bi_ui += MS_list.ll_funcs[i][ss](T_i)
                        bj_ui += MS_list.ll_funcs[i][ss](T_j)
                        bi_uj += MS_list.ll_funcs[i+1][ss](T_i)
                        bj_uj += MS_list.ll_funcs[i+1][ss](T_j)

                    logratio = bi_ui + bj_uj - bi_uj - bj_ui

                    accepted = roll_acceptance(MS_list.RNG, -logratio)

                    if accepted:
                        MS_list.H.loglikelihood[i, k] = bi_uj
                        MS_list.H.loglikelihood[i+1, k] = bj_ui
                        MS_list.H.states[i, :, k] = MS_list.H.states[i+1, :, k]
                        MS_list.H.states[i+1, :, k] = MS_list.H.states[i, :, k]
                        MS_list.ll_funcs[i+1], MS_list.ll_funcs[i] = MS_list.ll_funcs[i], MS_list.ll_funcs[i+1]

                    else:
                        MS_list.H.loglikelihood[i, k] = MS_list.H.loglikelihood[i, k]
                        MS_list.H.loglikelihood[i+1, k] = MS_list.H.loglikelihood[i+1, k]
                        MS_list.H.states[i, :, k] = MS_list.H.states[i, :, k]
                        MS_list.H.states[i+1, :, k] = MS_list.H.states[i+1, :, k]

            if verbose or k % MSG_FREQ == 0 or k < starting_iter + MSG_COOLDOWN:
                MS_list.print_status()
            MS_list.latest_iter = k

        except KeyboardInterrupt:
            MS_list.logger.warning(f"Terminating with k={k-1} iters completed:")
            for MS in MS_list.MS:
                MS_list.H.truncate(k)
            break

        if checkpoint_freq is not None and k % checkpoint_freq == 0:
            chpt_header = MS_list.ensemble_fields["checkpoint_header"]
            chpt_fname = os.path.join(MS_list.ensemble_fields["checkpoint_dirname"],
                                      f"{chpt_header}.pik")
            MS_list.logger.info(f"Saving checkpoint at k={k}; fname {chpt_fname}")
            MS_list.random_state = np.random.get_state()
            MS_list.checkpoint(chpt_fname)
    return


def kill_from_cl(signal_n, frame):
    raise KeyboardInterrupt("Terminate from command line")


def all_signal_handler(func):
    for s in signal.Signals:
        try:
            signal.signal(s, func)
        except (ValueError, OSError):
            continue
    return


def metro(sim_info, iniPar, e_data, MCMC_fields, param_info,
          verbose=False, export_path="", **kwargs):
    logger_name = kwargs.get("logger_name", "Ensemble0")
    
    clock0 = perf_counter()

    # Setup
    all_signal_handler(kill_from_cl)

    make_dir(MCMC_fields["checkpoint_dirname"])
    make_dir(MCMC_fields["output_path"])

    load_checkpoint = MCMC_fields["load_checkpoint"]
    num_iters = MCMC_fields["num_iters"]
    if load_checkpoint is None:
        MS_list = Ensemble(param_info, sim_info, MCMC_fields, num_iters, logger_name, verbose)
        MS_list.checkpoint(os.path.join(MS_list.ensemble_fields["output_path"], export_path))
        if MS_list.ensemble_fields.get("checkpoint_header", None) is None:
            MS_list.ensemble_fields["checkpoint_header"] = export_path[:export_path.find(".pik")]

        starting_iter = 1

        e_string = [f"[{e_data[1][i][0]}...{e_data[1][i][-1]}]" for i in range(len(e_data[1]))]
        MS_list.logger.info(f"E data: {e_string}")
        i_string = [f"[{iniPar[i][0]}...{iniPar[i][-1]}]" for i in range(len(iniPar))]
        MS_list.logger.info(f"Initial condition: {i_string}")
        # Just so MS saves a record of these
        MS_list.iniPar = iniPar
        MS_list.times, MS_list.vals, MS_list.uncs = e_data

        for i, unc in enumerate(MS_list.uncs):
            MS_list.logger.info(f"{i} exp unc max: {np.amax(unc)} avg: {np.mean(unc)}")

        if MCMC_fields.get("irf_convolution", None) is not None:
            irfs = {}
            for i in MCMC_fields["irf_convolution"]:
                if i > 0 and i not in irfs:
                    irfs[int(i)] = np.loadtxt(os.path.join("IRFs", f"irf_{int(i)}nm.csv"),
                                              delimiter=",")

            MS_list.IRF_tables = make_I_tables(irfs)
        else:
            MS_list.IRF_tables = {}

    else:
        with open(os.path.join(MCMC_fields["checkpoint_dirname"],
                               load_checkpoint), 'rb') as ifstream:
            MS_list : Ensemble = pickle.load(ifstream)
            np.random.set_state(MS_list.random_state)
            MS_list.ll_funcs = [None for _ in range(len(MS_list.MS))]
            MS_list.logger, MS_list.handler = start_logging(
                log_dir=MCMC_fields["output_path"], name=logger_name, verbose=verbose)
            if "starting_iter" in MCMC_fields and MCMC_fields["starting_iter"] < MS_list.latest_iter:
                starting_iter = MCMC_fields["starting_iter"]
                MS_list.H.extend(starting_iter)

            else:
                starting_iter = MS_list.latest_iter + 1
                MS_list.H.extend(num_iters)
                MS_list.ensemble_fields["num_iters"] = MCMC_fields["num_iters"]

    # From this point on, for consistency, work with ONLY the MetroState objects
    MS_list.logger.info(f"Sim info: {MS_list.sim_info}")
    MS_list.logger.info(f"Ensemble fields: {MS_list.ensemble_fields}")
    for i, MS in enumerate(MS_list.MS):
        MS_list.logger.info(f"Metrostate #{i}:")
        MS_list.logger.info(f"MCMC fields: {MS.MCMC_fields}")

    need_initial_state = (load_checkpoint is None)
    main_metro_loop(MS_list, starting_iter, num_iters,
                    need_initial_state=need_initial_state, verbose=verbose)

    MS_list.random_state = np.random.get_state()
    if export_path is not None:
        MS_list.logger.info(f"Exporting to {MS_list.ensemble_fields['output_path']}")
        MS_list.checkpoint(os.path.join(MS_list.ensemble_fields["output_path"], export_path))

    final_t = perf_counter() - clock0
    MS_list.logger.info(f"Metro took {final_t} s ({final_t / 3600} hr)")
    MS_list.logger.info(f"Avg: {final_t / MS_list.ensemble_fields['num_iters']} s per iter")
    for i, MS in enumerate(MS_list.MS):
        MS_list.logger.info(f"Metrostate #{i}:")
        MS_list.logger.info(f"Acceptance rate: {np.sum(MS_list.H.accept[i]) / len(MS_list.H.accept[i].flatten())}")

    MS_list.stop_logging(0)
    return MS_list
