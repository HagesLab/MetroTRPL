# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 22:13:26 2022

@author: cfai2
"""
import os
import signal
import pickle
import numpy as np

from sim_utils import Ensemble
from mcmc_logging import start_logging, stop_logging
from bayes_io import make_dir
from laplace import make_I_tables

# Constants
MAX_PROPOSALS = 100
MSG_FREQ = 100
MSG_COOLDOWN = 3 # Log first few states regardless of verbose


def check_approved_param(new_state, param_info, indexes, active, do_log):
    """ Raise a warning for non-physical or unrealistic proposed trial moves,
        or proposed moves that exceed the prior distribution.
    """
    order = param_info['names']
    checks = {}
    prior_dist = param_info["prior_dist"]

    # Ensure proposal stays within bounds of prior distribution
    
    diff = np.where(do_log, 10 ** new_state, new_state)
    for i, param in enumerate(order):
        if not active[i]:
            continue

        lb = prior_dist[param][0]
        ub = prior_dist[param][1]
        checks[f"{param}_size"] = (lb < diff[i] < ub)

    # TRPL specific checks:
    # p0 > n0 by definition of a p-doped material
    if 'p0' in order and 'n0' in order:
        checks["p0_greater"] = (new_state[indexes["p0"]]
                                > new_state[indexes["n0"]])
    else:
        checks["p0_greater"] = True

    # tau_n and tau_p must be *close* (within 2 OM) for a reasonable midgap SRH
    if 'tauN' in order and 'tauP' in order:
        # Compel logscale for this one - makes for easier check
        logtn = new_state[indexes['tauN']]
        if not do_log[indexes["tauN"]]:
            logtn = np.log10(logtn)

        logtp = new_state[indexes['tauP']]
        if not do_log[indexes["tauP"]]:
            logtp = np.log10(logtp)

        diff = np.abs(logtn - logtp)
        checks["tn_tp_close"] = (diff <= 2)

    else:
        checks["tn_tp_close"] = True

    failed_checks = [k for k in checks if not checks[k]]

    return failed_checks


def select_next_params(current_state, param_info, indexes, is_active, trial_move, do_log, RNG=None, coerce_hard_bounds=False, logger=None, verbose=False):
    """ 
    Trial move function: returns a new proposed state equal to the current_state plus a uniform random displacement
    """
    if RNG is None:
        RNG = np.random.default_rng(235817049752375780)

    _current_state = np.array(current_state, dtype=float)

    mu_constraint = param_info.get("do_mu_constraint", None)

    _current_state = np.where(do_log, np.log10(_current_state), _current_state)

    tries = 0

    # Try up to MAX_PROPOSALS times to come up with a proposal that stays within
    # the hard boundaries, if we ask
    if coerce_hard_bounds:
        max_tries = MAX_PROPOSALS
    else:
        max_tries = 1

    new_state = np.array(_current_state)
    while tries < max_tries:
        tries += 1

        new_state = np.where(is_active, RNG.uniform(_current_state-trial_move, _current_state+trial_move), _current_state)

        if mu_constraint is not None:
            ambi = mu_constraint[0]
            ambi_std = mu_constraint[1]
            if verbose and logger is not None:
                logger.debug(f"mu constraint: ambi {ambi} +/- {ambi_std}")
            new_muambi = np.random.uniform(ambi - ambi_std, ambi + ambi_std)
            new_state[indexes["mu_p"]] = np.log10(
                (2 / new_muambi - 1 / 10 ** new_state[indexes["mu_n"]])**-1)

        failed_checks = check_approved_param(new_state, param_info, indexes, is_active, do_log)
        success = len(failed_checks) == 0
        if success:
            if verbose and logger is not None:
                logger.info(f"Found params in {tries} tries")
            break

        if logger is not None and len(failed_checks) > 0:
            logger.warning(f"Failed checks: {failed_checks}")

    new_state = np.where(do_log, 10 ** new_state, new_state)
    return new_state


def roll_acceptance(logratio):
    accepted = False
    if logratio >= 0: # Automatic accept
        accepted = True

    else:
        accept = np.random.random()
        if accept < np.exp(logratio):
            accepted = True
    return accepted


def almost_equal(x, x0, threshold=1e-10):
    if x.shape != x0.shape:
        return False

    return np.abs(np.nanmax((x - x0) / x0)) < threshold


def main_metro_loop(MS_list : Ensemble, starting_iter, num_iters,
                    need_initial_state=True, logger=None, verbose=False):
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
        if logger is not None:
            logger.info("Simulating initial state:")
        # Calculate likelihood of initial guess
        for MS in MS_list.MS:
            logll = MS_list.eval_trial_move(MS.init_state, MS.MCMC_fields, logger, verbose)

            MS.H.loglikelihood[0, 0] = logll
            MS.H.states[:, 0] = MS.init_state
    for k in range(starting_iter, num_iters):
        try:
            if logger is not None:
                logger.info("#####")
                logger.info(f"Iter {k}")

            if MS_list.ensemble_fields["do_parallel_tempering"] and k % MS_list.ensemble_fields["temper_freq"] == 0:
                # Select a pair (the ith and (i+1)th) of chains
                i = np.random.choice(np.arange(len(MS_list.MS)-1))
            else:
                i = -1337

            for m, MS in enumerate(MS_list.MS):
                if logger is not None:
                    logger.info(f"MetroState #{m}")
                if m == i:
                    # Do a tempering move between (swap the positions of) the ith and (i+1)th chains
                    if logger is not None:
                        logger.info(f"Tempering - swapping chains {i} and {i+1}")
                    MS_I = MS_list.MS[i]
                    MS_J = MS_list.MS[i+1]
                    beta_j = MS_J.MCMC_fields["_beta"]
                    beta_i = MS_I.MCMC_fields["_beta"]
                    logratio = -(beta_j - beta_i) * (MS_J.H.loglikelihood[0, k-1] - MS_I.H.loglikelihood[0, k-1])

                    accepted = roll_acceptance(logratio)

                    if accepted:
                        MS_I.H.loglikelihood[0, k] = MS_J.H.loglikelihood[0, k-1]
                        MS_J.H.loglikelihood[0, k] = MS_I.H.loglikelihood[0, k-1]
                        MS_I.H.states[:, k] = MS_J.H.states[:, k-1]
                        MS_J.H.states[:, k] = MS_I.H.states[:, k-1]

                    else:
                        MS_I.H.loglikelihood[0, k] = MS_I.H.loglikelihood[0, k-1]
                        MS_J.H.loglikelihood[0, k] = MS_J.H.loglikelihood[0, k-1]
                        MS_I.H.states[:, k] = MS_I.H.states[:, k-1]
                        MS_J.H.states[:, k] = MS_J.H.states[:, k-1]

                elif m == i+1:
                    # Skip the (i+1)th chain if it was just swapped
                    continue

                else:
                    # Non-tempering move, or all other chains not selected for tempering

                    new_state = select_next_params(MS.H.states[:, k-1], MS.param_info, MS_list.param_indexes, MS_list.ensemble_fields["active"],
                                                   MS_list.ensemble_fields["trial_move"], MS_list.ensemble_fields["do_log"], MS_list.RNG,
                                                   MS.MCMC_fields.get("hard_bounds", 0), logger)

                    if (verbose or k % MSG_FREQ == 0 or k < starting_iter + MSG_COOLDOWN) and logger is not None:
                        MS.print_status(k - 1, MS_list.ensemble_fields["active"], new_state, logger)

                    logll = MS_list.eval_trial_move(new_state, MS.MCMC_fields, logger, verbose)
                    
                    if verbose and logger is not None:
                        logger.info(f"Log likelihood of proposed move: {MS.MCMC_fields.get('_beta', 1) * logll}")
                    logratio = MS.MCMC_fields.get('_beta', 1) * (logll - MS.H.loglikelihood[0, k-1])
                    if np.isnan(logratio):
                        logratio = -np.inf

                    accepted = roll_acceptance(logratio)

                    if accepted:
                        MS.H.loglikelihood[0, k] = logll
                        MS.H.states[:, k] = new_state
                        MS.H.accept[0, k] = 1
                    else:
                        MS.H.loglikelihood[0, k] = MS.H.loglikelihood[0, k-1]
                        MS.H.states[:, k] = MS.H.states[:, k-1]

            MS_list.latest_iter = k

        except KeyboardInterrupt:
            if logger is not None:
                logger.warning(f"Terminating with k={k-1} iters completed:")
            for MS in MS_list.MS:
                MS.H.truncate(k)
            break

        if checkpoint_freq is not None and k % checkpoint_freq == 0:
            chpt_header = MS_list.ensemble_fields["checkpoint_header"]
            chpt_fname = os.path.join(MS_list.ensemble_fields["checkpoint_dirname"],
                                      f"{chpt_header}.pik")
            if logger is not None:
                logger.info(f"Saving checkpoint at k={k}; fname {chpt_fname}")
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
          verbose=False, export_path="", logger=None):

    if logger is None:  # Require a logger
        logger, handler = start_logging(log_dir=MCMC_fields["output_path"],
                                        name="Log-")
        using_default_logger = True
    else:
        using_default_logger = False
        handler = None

    # Setup
    logger.info(f"PID: {os.getpid()}")
    all_signal_handler(kill_from_cl)

    make_dir(MCMC_fields["checkpoint_dirname"])
    make_dir(MCMC_fields["output_path"])

    load_checkpoint = MCMC_fields["load_checkpoint"]
    num_iters = MCMC_fields["num_iters"]
    if load_checkpoint is None:
        MS_list = Ensemble(param_info, sim_info, MCMC_fields, num_iters)
        MS_list.checkpoint(os.path.join(MS_list.ensemble_fields["output_path"], export_path))
        if "checkpoint_header" not in MS_list.ensemble_fields:
            MS_list.ensemble_fields["checkpoint_header"] = export_path[:export_path.find(".pik")]

        starting_iter = 1

        e_string = [f"[{e_data[1][i][0]}...{e_data[1][i][-1]}]" for i in range(len(e_data[1]))]
        logger.info(f"E data: {e_string}")
        i_string = [f"[{iniPar[i][0]}...{iniPar[i][-1]}]" for i in range(len(iniPar))]
        logger.info(f"Initial condition: {i_string}")
        # Just so MS saves a record of these
        MS_list.iniPar = iniPar
        MS_list.times, MS_list.vals, MS_list.uncs = e_data

        if logger is not None:
            for i, unc in enumerate(MS_list.uncs):
                logger.info(f"{i} exp unc max: {np.amax(unc)} avg: {np.mean(unc)}")

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
            if "starting_iter" in MCMC_fields and MCMC_fields["starting_iter"] < MS_list.latest_iter:
                starting_iter = MCMC_fields["starting_iter"]
                for MS in MS_list.MS:
                    MS.H.extend(starting_iter)

            else:
                starting_iter = MS_list.latest_iter + 1

                for MS in MS_list.MS:
                    MS.H.extend(num_iters)
                    MS.MCMC_fields["num_iters"] = MCMC_fields["num_iters"]

    # From this point on, for consistency, work with ONLY the MetroState objects
    logger.info(f"Sim info: {MS_list.sim_info}")
    logger.info(f"Ensemble fields: {MS_list.ensemble_fields}")
    for i, MS in enumerate(MS_list.MS):
        logger.info(f"Metrostate #{i}:")
        logger.info(f"Param infos: {MS.param_info}")
        logger.info(f"MCMC fields: {MS.MCMC_fields}")

    need_initial_state = (load_checkpoint is None)
    main_metro_loop(MS_list, starting_iter, num_iters,
                    need_initial_state=need_initial_state,
                    logger=logger, verbose=verbose)

    MS_list.random_state = np.random.get_state()
    if export_path is not None:
        logger.info(f"Exporting to {MS_list.ensemble_fields['output_path']}")
        MS_list.checkpoint(os.path.join(MS_list.ensemble_fields["output_path"], export_path))

    if using_default_logger:
        stop_logging(logger, handler, 0)
    return MS_list
