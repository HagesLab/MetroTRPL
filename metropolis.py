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

try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD   # Global communicator
except ImportError as e:
    print(f"ImportError: {e}")
    print("Failed to load MPI library!")
    print("To avoid COMM errors and use single-CPU fallback, pass serial_fallback=True in "
          "your metro() call (i.e. in main.py)")
    COMM = None

from mcmc_logging import start_logging, stop_logging
from sim_utils import Ensemble
from trial_move_evaluation import eval_trial_move
from trial_move_generation import make_trial_move
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


def trial_displacement_move(k, states, logll, accept, unique_fields, shared_fields, RNG, logger):
    new_state = make_trial_move(states[:, k-1],
                            unique_fields["_T"] ** 0.5 * shared_fields["base_trial_move"],
                            shared_fields,
                            RNG, logger)
    _logll, new_ll_func = eval_trial_move(new_state, unique_fields, shared_fields, logger)

    logratio = _logll - logll[k-1]
    if np.isnan(logratio):
        logratio = -np.inf

    accepted = roll_acceptance(RNG, logratio)

    if accepted:
        logll[k] = _logll
        states[:, k] = new_state
        accept[k] = 1
        return new_ll_func
    else:
        logll[k] = logll[k-1]
        states[:, k] = states[:, k-1]
        return None


def swap_move_serial(k, i, j, states, logll, ll_funcs, unique_fields, shared_fields, RNG, logger):
    # Select a pair (the ith and jth) of chains
    # Do a tempering move between (swap the positions of) the ith and jth chains
    T_j = shared_fields["_T"][j]
    T_i = shared_fields["_T"][i]

    bi_ui, bj_ui, bi_uj, bj_uj = 0, 0, 0, 0
    for ss in range(shared_fields["_sim_info"]["num_meas"]):
        bi_ui += ll_funcs[i][ss](T_i)
        bj_ui += ll_funcs[i][ss](T_j)
        bi_uj += ll_funcs[j][ss](T_i)
        bj_uj += ll_funcs[j][ss](T_j)

    logratio = bi_ui + bj_uj - bi_uj - bj_ui

    accepted = roll_acceptance(RNG, -logratio)

    if accepted:
        logll[i, k] = bi_uj
        logll[j, k] = bj_ui
        states[i, :, k], states[j, :, k] = states[j, :, k], states[i, :, k]
        _, ll_funcs[j] = eval_trial_move(states[j, :, k], unique_fields[j], shared_fields, logger)
        _, ll_funcs[i] = eval_trial_move(states[i, :, k], unique_fields[i], shared_fields, logger)

    return accepted


def main_metro_loop_serial(states, logll, accept, starting_iter, num_iters, shared_fields,
                           unique_fields, RNG,
                           logger, need_initial_state=True):
    """
    Serial version of main_metro_loop().

    """

    n_chains = shared_fields["_n_chains"]
    n_sigmas = shared_fields["_n_sigmas"]
    chains_per_sigma = shared_fields["chains_per_sigma"]
    swap_accept = np.zeros(n_chains, dtype=int)
    swap_attempts = np.zeros(n_chains, dtype=int)
    ll_funcs = [None for _ in range(n_chains)]
    if need_initial_state:
        logger.info("Simulating initial state:")
        # Calculate likelihood of initial guess
        for m in range(n_chains):
            _logll, ll_func = eval_trial_move(states[m, :, 0], unique_fields[m], shared_fields, logger)
            logll[m, 0] = _logll
            ll_funcs[m] = ll_func
        starting_iter += 1
    else:
        for m in range(n_chains):
            _, ll_func = eval_trial_move(states[m, :, starting_iter - 1], unique_fields[m], shared_fields, logger)
            ll_funcs[m] = ll_func

    for k in range(starting_iter, num_iters):
        if k % MSG_FREQ == 0 or k < starting_iter + MSG_COOLDOWN:
            for m in range(n_chains):
                logger.info(f"Iter {k} MetroState #{m} Current state: {states[m, :, k-1]} logll {logll[m, k-1]}")

        for m in range(n_chains):
            # Trial displacement move
            new_ll_func = trial_displacement_move(k, states[m], logll[m], accept[m], unique_fields[m], shared_fields, RNG, logger)
            if new_ll_func is not None:
                ll_funcs[m] = new_ll_func

        if shared_fields["do_parallel_tempering"] and k % shared_fields["temper_freq"] == 0:
            for _ in range(n_chains - 1):
                r_sigma = RNG.integers(0, n_sigmas - 1)
                offset_1 = RNG.integers(0, chains_per_sigma)
                offset_2 = RNG.integers(0, chains_per_sigma)

                i = r_sigma * chains_per_sigma + offset_1
                j = (r_sigma + 1) * chains_per_sigma + offset_2

                swap_attempts[i] += 1
                swap_success = swap_move_serial(k, i, j, states, logll, ll_funcs, unique_fields, shared_fields, RNG, logger)
                if swap_success:
                    swap_accept[i] += 1

    return states, logll, accept, swap_attempts, swap_accept


def main_metro_loop(m, states, logll, accept, starting_iter, num_iters, shared_fields,
                    unique_fields, RNG,
                    logger, need_initial_state=True):
    """
    Run the Metropolis loop for each chain in an Ensemble()
    over a specified number of iterations,
    storing all info in MetroState() objects and saving the Ensemble()
    as occasional checkpoints.

    Parameters
    ----------
    m : int
        Numerical index of chain in ensemble
    states : ndarray
        Array of states visited by the mth chain
    logll : ndarray
        Array of log likelihoods for each state visited by the mth chain
    accept : ndarray
        Array of whether each trial move by the mth chain was accepted
    starting_iter : int
        Index of first iter. 1 if starting from scratch, checkpoint-dependent
        otherwise.
    num_iters : int
        Index of final iter.
    shared_fields : dict
        Shared ensemble fields.
    unique_fields : dict
        Settings unique to the mth chain.
    RNG : np.random.Generator
        A random number Generator instance used to make the trial moves.
    logger : logging object, optional
        Stream to write status messages. The default is None.
    need_initial_state : bool, optional
        Whether we're starting from scratch, and thus need to calculate the
        likelihood of the initial state before moving around.
        The default is True.

    Returns
    -------
    None. Ensemble() and MetroStates is updated throughout.

    """
    swap_accept = 0
    swap_attempts = 0

    if need_initial_state:
        logger.info("Simulating initial state:")
        # Calculate likelihood of initial guess
        _logll, ll_func = eval_trial_move(states[:, 0], unique_fields, shared_fields, logger)
        logll[0] = _logll
        starting_iter += 1
    else:
        # Repeat most recent eval, to re-acquire non-pickleable ll_func
        _, ll_func = eval_trial_move(states[:, starting_iter - 1], unique_fields, shared_fields, logger)

    for k in range(starting_iter, num_iters):
        if k % MSG_FREQ == 0 or k < starting_iter + MSG_COOLDOWN:
            logger.info(f"Iter {k} MetroState #{m} Current state: {states[:, k-1]} logll {logll[k-1]}")

        # Trial displacement move
        new_ll_func = trial_displacement_move(k, states, logll, accept, unique_fields, shared_fields, RNG, logger)
        if new_ll_func is not None:
            ll_func = new_ll_func

        if shared_fields["do_parallel_tempering"] and k % shared_fields["temper_freq"] == 0:
            # TODO: Precalculate these, to avoid the bcast
            for _ in range(shared_fields["_n_chains"] - 1):
                i = None
                j = None
                if m == 0:
                    # Select a pair (the ith and jth) of chains
                    r_sigma = RNG.integers(0, shared_fields["_n_sigmas"] - 1)
                    offset_1 = RNG.integers(0, shared_fields["chains_per_sigma"])
                    offset_2 = RNG.integers(0, shared_fields["chains_per_sigma"])

                    i = r_sigma * shared_fields["chains_per_sigma"] + offset_1
                    j = (r_sigma + 1) * shared_fields["chains_per_sigma"] + offset_2

                i = COMM.bcast(i, root=0)
                j = COMM.bcast(j, root=0)

                # Do a tempering move between (swap the positions of) the ith and jth chains
                T_j = shared_fields["_T"][j]
                T_i = shared_fields["_T"][i]

                bi_ui, bj_ui = 0, 0
                for ss in range(shared_fields["_sim_info"]["num_meas"]):
                    bi_ui += ll_func[ss](T_i)
                    bj_ui += ll_func[ss](T_j)

                log_ri = bi_ui - bj_ui

                # Must get log_rj (log_ri from j) from other process
                log_rj = 0
                if m == i:
                    log_rj = COMM.recv(source=j)
                elif m == j:
                    COMM.send(-log_ri, dest=i)

                logratio = log_ri + log_rj

                accepted = None
                if m == i:
                    swap_attempts += 1
                    accepted = roll_acceptance(RNG, -logratio)
                    COMM.send(accepted, dest=j)
                elif m == j:
                    accepted = COMM.recv(source=i)

                if m == i and accepted:
                    swap_accept += 1
                    logll[k] = COMM.recv(source=j)
                    COMM.send(bj_ui, dest=j)

                    temp_states = COMM.recv(source=j)
                    COMM.send(states[:, k], dest=j)
                    states[:, k] = temp_states

                    _, ll_func = eval_trial_move(states[:, k], unique_fields, shared_fields, logger)

                elif m == j and accepted:
                    COMM.send(bi_ui, dest=i)
                    logll[k] = COMM.recv(source=i)

                    COMM.send(states[:, k], dest=i)
                    states[:, k] = COMM.recv(source=i)

                    _, ll_func = eval_trial_move(states[:, k], unique_fields, shared_fields, logger)

            COMM.Barrier()

        #if verbose or k % MSG_FREQ == 0 or k < starting_iter + MSG_COOLDOWN:
        #    MS_list.print_status()
        #MS_list.latest_iter = k

    return states, logll, accept, swap_attempts, swap_accept


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
    clock0 = perf_counter()

    # Setup
    serial_fallback = kwargs.get("serial_fallback", False)
    all_signal_handler(kill_from_cl)

    make_dir(MCMC_fields["output_path"])

    load_checkpoint = MCMC_fields.get("load_checkpoint", None)
    num_iters = MCMC_fields["num_iters"]
    checkpoint_freq = MCMC_fields.get("checkpoint_freq", num_iters)
    RNG = np.random.default_rng(MCMC_fields.get("random_seed", None))
    if serial_fallback:
        rank = 0
    else:
        rank = COMM.Get_rank()  # Process index
    logger_name = kwargs.get("logger_name", "Ensemble0")
    logger_name += f"-rank{rank}-"

    starting_iter = 0
    global_states = None
    global_logll = None
    global_accept = None
    state_dims = None
    shared_fields = None
    unique_fields = None
    RNG_state = None
    logger, handler = start_logging(
        log_dir=MCMC_fields["output_path"], name=logger_name, verbose=verbose)

    if rank == 0:
        if load_checkpoint is None:
            MS_list = Ensemble(param_info, sim_info, MCMC_fields, num_iters, verbose)

            spr = MS_list.ensemble_fields["random_spread"]
            init_randomize = 10 ** RNG.uniform(-spr, spr, size=MS_list.H.states[:, :, 0].shape)
            init_randomize[:, np.logical_not(MS_list.ensemble_fields["active"])] = 1

            MS_list.H.states[:, :, 0] *= init_randomize

            MS_list.checkpoint(os.path.join(MS_list.ensemble_fields["output_path"], export_path))

            e_string = [f"[{e_data[1][i][0]}...{e_data[1][i][-1]}]" for i in range(len(e_data[1]))]
            logger.info(f"E data: {e_string}")
            i_string = [f"[{iniPar[i][0]}...{iniPar[i][-1]}]" for i in range(len(iniPar))]
            logger.info(f"Initial condition: {i_string}")
            # Just so MS saves a record of these
            MS_list.ensemble_fields["_init_params"] = iniPar
            MS_list.ensemble_fields["_times"], MS_list.ensemble_fields["_vals"], MS_list.ensemble_fields["_uncs"] = e_data
            MS_list.random_state = RNG.bit_generator.state
            for i, unc in enumerate(MS_list.ensemble_fields["_uncs"]):
                logger.info(f"{i} exp unc max: {np.amax(unc)} avg: {np.mean(unc)}")

            if MS_list.ensemble_fields.get("irf_convolution", None) is not None:
                irfs = {}
                for i in MS_list.ensemble_fields["irf_convolution"]:
                    if i > 0 and i not in irfs:
                        irfs[int(i)] = np.loadtxt(os.path.join("IRFs", f"irf_{int(i)}nm.csv"),
                                                delimiter=",")

                MS_list.ensemble_fields["_IRF_tables"] = make_I_tables(irfs)
            else:
                MS_list.ensemble_fields["_IRF_tables"] = {}

        else:
            with open(os.path.join(MCMC_fields["output_path"],
                                load_checkpoint), 'rb') as ifstream:
                MS_list : Ensemble = pickle.load(ifstream)
                if "starting_iter" in MCMC_fields and MCMC_fields["starting_iter"] < MS_list.latest_iter:
                    starting_iter = MCMC_fields["starting_iter"]
                    MS_list.H.extend(starting_iter)

                else:
                    starting_iter = MS_list.latest_iter
                    MS_list.H.extend(num_iters)
                    MS_list.ensemble_fields["num_iters"] = MCMC_fields["num_iters"]

                # Compatibility with prior ensembles
                if "_n_sigmas" not in MS_list.ensemble_fields:
                    MS_list.ensemble_fields["_n_sigmas"] = MS_list.ensemble_fields["_n_chains"]
                if "chains_per_sigma" not in MS_list.ensemble_fields:
                    MS_list.ensemble_fields["chains_per_sigma"] = 1


        global_states = MS_list.H.states
        global_logll = MS_list.H.loglikelihood
        global_accept = MS_list.H.accept
        state_dims = global_states.shape

        shared_fields = MS_list.ensemble_fields
        unique_fields = MS_list.unique_fields
        RNG_state = MS_list.random_state

        # From this point on, for consistency, work with ONLY the MetroState objects
        logger.info(f"Sim info: {MS_list.ensemble_fields['_sim_info']}")
        logger.info(f"Ensemble fields: {MS_list.ensemble_fields}")
        for i, MS in enumerate(MS_list.unique_fields):
            logger.info(f"Metrostate #{i}:")
            logger.info(f"MCMC fields: {MS}")
    else:
        MS_list = None

    if serial_fallback:
        if rank == 0:
            RNG.bit_generator.state = RNG_state
            need_initial_state = (load_checkpoint is None)

            ending_iter = min(starting_iter + checkpoint_freq, num_iters)
            while ending_iter <= num_iters:
                logger.info(f"Simulating from {starting_iter} to {ending_iter}")
                global_states, global_logll, global_accept, all_swap_attempts, all_swap_accept = main_metro_loop_serial(
                    global_states, global_logll, global_accept,
                    starting_iter, ending_iter, shared_fields, unique_fields,
                    RNG, logger, need_initial_state=need_initial_state
                )
                if ending_iter == num_iters:
                    break

                MS_list.latest_iter = ending_iter
                MS_list.H.pack(global_states, global_logll, global_accept)
                MS_list.random_state = RNG.bit_generator.state
                logger.info(f"Saving checkpoint at k={ending_iter}")
                MS_list.checkpoint(os.path.join(MS_list.ensemble_fields["output_path"], export_path))

                need_initial_state = False
                starting_iter = ending_iter
                ending_iter = min(ending_iter + checkpoint_freq, num_iters)
            logger.info(f"Rank {rank} took {perf_counter() - clock0} s")
    else:
        state_dims = COMM.bcast(state_dims, root=0)  # (n_chains, n_params, n_iters)
        local_states = np.empty((state_dims[1], state_dims[2]), dtype=float)
        COMM.Scatter(global_states, local_states, root=0)
        local_logll = np.empty(state_dims[2], dtype=float)
        COMM.Scatter(global_logll, local_logll, root=0)
        local_accept = np.empty(state_dims[2], dtype=int)
        COMM.Scatter(global_accept, local_accept, root=0)

        unique_fields = COMM.scatter(unique_fields, root=0)

        RNG_state = COMM.bcast(RNG_state, root=0)
        RNG.bit_generator.state = RNG_state
        starting_iter = COMM.bcast(starting_iter, root=0)
        shared_fields = COMM.bcast(shared_fields, root=0)
        need_initial_state = (load_checkpoint is None)

        ending_iter = min(starting_iter + checkpoint_freq, num_iters)
        while ending_iter <= num_iters:
            logger.info(f"Simulating from {starting_iter} to {ending_iter}")
            local_states, local_logll, local_accept, local_swap_attempts, local_swap_accept = main_metro_loop(
                rank, local_states, local_logll, local_accept,
                starting_iter, ending_iter, shared_fields, unique_fields,
                RNG, logger, need_initial_state=need_initial_state
            )
            if ending_iter == num_iters:
                break

            all_swap_attempts = COMM.gather(local_swap_attempts, root=0)
            all_swap_accept = COMM.gather(local_swap_accept, root=0)
            COMM.Gather(local_states, global_states, root=0)
            COMM.Gather(local_logll, global_logll, root=0)
            COMM.Gather(local_accept, global_accept, root=0)

            if rank == 0:
                MS_list.H.swap_attempts += np.array(all_swap_attempts)
                MS_list.H.swap_accept += np.array(all_swap_accept)

                MS_list.latest_iter = ending_iter
                MS_list.H.pack(global_states, global_logll, global_accept)
                MS_list.random_state = RNG.bit_generator.state
                logger.info(f"Saving checkpoint at k={ending_iter}")
                MS_list.checkpoint(os.path.join(MS_list.ensemble_fields["output_path"], export_path))

            need_initial_state = False
            starting_iter = ending_iter
            ending_iter = min(ending_iter + checkpoint_freq, num_iters)


        all_swap_attempts = COMM.gather(local_swap_attempts, root=0)
        all_swap_accept = COMM.gather(local_swap_accept, root=0)
        COMM.Gather(local_states, global_states, root=0)
        COMM.Gather(local_logll, global_logll, root=0)
        COMM.Gather(local_accept, global_accept, root=0)
        logger.info(f"Rank {rank} took {perf_counter() - clock0} s")

    if rank == 0:
        MS_list.H.swap_attempts += np.array(all_swap_attempts)
        MS_list.H.swap_accept += np.array(all_swap_accept)
        MS_list.latest_iter = ending_iter
        MS_list.H.pack(global_states, global_logll, global_accept)
        MS_list.random_state = RNG.bit_generator.state
        swap_percent =  100*MS_list.H.swap_accept[:-MS_list.ensemble_fields['chains_per_sigma']] / MS_list.H.swap_attempts[:-MS_list.ensemble_fields['chains_per_sigma']]
        logger.info(f"Swap accept rate: {MS_list.H.swap_accept} accepted of {MS_list.H.swap_attempts} attempts ({swap_percent} %)")
        logger.info(f"Exporting to {MS_list.ensemble_fields['output_path']}")
        MS_list.checkpoint(os.path.join(MS_list.ensemble_fields["output_path"], export_path))

    # final_t = perf_counter() - clock0
    # logger.info(f"Metro took {final_t} s ({final_t / 3600} hr)")
    # logger.info(f"Avg: {final_t / MS_list.ensemble_fields['num_iters']} s per iter")
    # for i, MS in enumerate(MS_list.MS):
    #     logger.info(f"Metrostate #{i}:")
    #     logger.info(f"Acceptance rate: {np.sum(MS_list.H.accept[i]) / len(MS_list.H.accept[i].flatten())}")

    stop_logging(logger, handler, 0)
    # return MS_list
