# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:04:20 2022

@author: cfai2
"""
from sys import float_info
import pickle
import logging

import numpy as np

from mcmc_logging import start_logging, stop_logging

# Constants
eps0 = 8.854 * 1e-12 * 1e-9  # [C / V m] to {C / V nm}
q_C = 1.602e-19  # [C per carrier]
# Default settings for solve_ivp
DEFAULT_HMAX = 4

DEFAULT_TEMPER_FREQ = 10
MAX_PROPOSALS = 100
# Allow this proportion of simulation values to become negative due to convolution,
# else the simulation is failed.
NEGATIVE_FRAC_TOL = 0.2

class History:
    """Record of past states the walk has been to."""

    def __init__(self, n_chains, num_iters, names):
        self.states_are_one_array = True
        self.states = np.zeros((n_chains, len(names), num_iters), dtype=float)
        self.accept = np.zeros((n_chains, num_iters), dtype=bool)
        self.loglikelihood = np.zeros((n_chains, num_iters), dtype=float)
        return

    def update(self, names):
        """Compatibility - repackage self.states array into attributes per parameter"""
        for i, param in enumerate(names):
            setattr(self, f"mean_{param}", self.states[:, i])
        return

    def pack(self, names, num_iters):
        """Compatibility - turn individual attributes into self.states"""
        self.states = np.zeros((len(names), num_iters))
        for k, param in enumerate(names):
            self.states[k] = getattr(self, f"mean_{param}")

    def truncate(self, k):
        """Cut off any incomplete iterations should the walk be terminated early"""
        self.states = self.states[:, :, :k]
        self.accept = self.accept[:, :k]
        self.loglikelihood = self.loglikelihood[:, :k]
        return

    def extend(self, new_num_iters):
        """Enlarge an existing MC chain to length new_num_iters, if needed"""
        current_num_iters = len(self.accept[0])
        if new_num_iters < current_num_iters:  # No extension needed
            self.truncate(new_num_iters)
            return
        if new_num_iters == current_num_iters:
            return

        addtl_iters = new_num_iters - current_num_iters
        self.accept = np.concatenate((self.accept, np.zeros((self.accept.shape[0], addtl_iters))), axis=1)
        self.loglikelihood = np.concatenate(
            (self.loglikelihood, np.zeros((self.loglikelihood.shape[0], addtl_iters))), axis=1
        )

        self.states = np.concatenate(
            (self.states, np.zeros((self.states.shape[0], self.states.shape[1], addtl_iters))), axis=2
        )
        return


class EnsembleTemplate:
    """
    Base class for Ensembles
    """

    ensemble_fields: dict  # Monte Carlo settings and data shared across all chains
    unique_fields: list[dict]  # List of settings unique to each chain
    H: History  # List of visited states
    # Lists of functions that can be used to repeat the logLL calculations for each chain's latest state
    ll_funcs: list
    random_state: dict  # State of the random number generator
    latest_iter: int  # Latest iteration # reached by chains
    logger: logging.Logger  # A standard logging.logger instance
    handler: logging.FileHandler  # A standard FileHandler instance

    def __init__(self):
        return

    def checkpoint(self, fname):
        """Save the current ensemble as a pickle object."""
        self.H.update(self.ensemble_fields["names"])

        with open(fname, "wb+") as ofstream:
            # TODO: This might break checkpoints
            handler = self.handler  # FileHandlers aren't pickleable
            self.handler = None
            ll_funcs = self.ll_funcs  # Lambda functions aren't pickleable either
            self.ll_funcs = None
            pickle.dump(self, ofstream)
            self.handler = handler
            self.ll_funcs = ll_funcs

    def print_status(self):
        k = self.latest_iter
        self.logger.info(f"Current loglikelihoods : {self.H.loglikelihood[:, k]}")
        for m in range(len(self.unique_fields)):
            self.logger.info(f"Chain {m}:")
            self.logger.info(f"Current state: {self.H.states[m, :, k]}"
            )

class Ensemble(EnsembleTemplate):
    """
    Ensemble of MetroStates that form a parallel tempering network.

    """

    def __init__(self, param_info, sim_info, MCMC_fields, num_iters, logger_name, verbose=False):
        super().__init__()
        self.logger, self.handler = start_logging(
            log_dir=MCMC_fields["output_path"], name=logger_name, verbose=verbose)
        # Transfer shared fields from chains to ensemble
        self.ensemble_fields = {}
        # Essential fields with no defaults
        for field in ["output_path", "load_checkpoint", "init_cond_path",
                      "measurement_path", "checkpoint_dirname", "checkpoint_freq",
                      "solver", "model", "num_iters", "log_y", "likel2move_ratio"]:
            self.ensemble_fields[field] = MCMC_fields.pop(field)

        # Optional fields that can default to None
        for field in ["parallel_tempering", "rtol", "atol", "scale_factor",
                      "fittable_fluences", "fittable_absps", "irf_convolution",
                      "do_mu_constraint", "checkpoint_header"]:
            self.ensemble_fields[field] = MCMC_fields.pop(field, None)

        self.ensemble_fields["temper_freq"] = MCMC_fields.pop(
            "temper_freq", DEFAULT_TEMPER_FREQ
        )
        self.ensemble_fields["hard_bounds"] = MCMC_fields.pop("hard_bounds", 0)
        self.ensemble_fields["hmax"] = MCMC_fields.pop("hmax", DEFAULT_HMAX)
        self.ensemble_fields["force_min_y"] = MCMC_fields.pop("force_min_y", 0)
        
        self.ensemble_fields["prior_dist"] = param_info.pop("prior_dist")
        # Transfer shared fields that need to become arrays
        self.ensemble_fields["do_log"] = param_info.pop("do_log")
        self.ensemble_fields["do_log"] = np.array(
            [self.ensemble_fields["do_log"][param] for param in param_info["names"]],
            dtype=bool,
        )

        self.ensemble_fields["base_trial_move"] = param_info.pop("trial_move")
        self.ensemble_fields["base_trial_move"] = np.array(
            [
                self.ensemble_fields["base_trial_move"][param] if param_info["active"][param] else 0
                for param in param_info["names"]
            ],
            dtype=float,
        )

        self.ensemble_fields["active"] = param_info.pop("active")
        self.ensemble_fields["active"] = np.array(
            [self.ensemble_fields["active"][param] for param in param_info["names"]],
            dtype=bool,
        )

        self.ensemble_fields["units"] = param_info.pop("unit_conversions")
        self.ensemble_fields["units"] = np.array(
            [
                self.ensemble_fields["units"].get(param, 1)
                for param in param_info["names"]
            ],
            dtype=float,
        )

        self.ensemble_fields["_param_indexes"] = {
            name: param_info["names"].index(name) for name in param_info["names"]
        }

        if self.ensemble_fields["parallel_tempering"] is None:
            n_chains = 1
            temperatures = [1]
        else:
            n_chains = len(self.ensemble_fields["parallel_tempering"])
            temperatures = self.ensemble_fields["parallel_tempering"]

        self.ensemble_fields["names"] = param_info.pop("names")

        # Record initial state
        init_state = np.array(
                [
                    param_info["init_guess"][param]
                    for param in self.ensemble_fields["names"]
                ],
            dtype=float,
        )
        self.H = History(n_chains, num_iters, self.ensemble_fields["names"])
        self.H.states[:, :, 0] = init_state

        self.ll_funcs = [None for _ in range(n_chains)]
        self.unique_fields: list[dict] = []
        for i in range(n_chains):
            self.unique_fields.append(dict(MCMC_fields))
            self.unique_fields[-1]["_T"] = temperatures[i]
            self.unique_fields[-1]["current_sigma"] = {
                m: max(self.ensemble_fields["base_trial_move"])
                * self.ensemble_fields["likel2move_ratio"][m]
                for m in sim_info["meas_types"]
            }

        self.ensemble_fields["do_parallel_tempering"] = n_chains > 1

        self.ensemble_fields["_sim_info"] = sim_info
        self.random_state = np.random.get_state()
        self.latest_iter = 0

    def stop_logging(self, err_code):
        stop_logging(self.logger, self.handler, err_code)


class MetroState:
    """
    Fields specific to each chain
    This class is deprecated and available only for compatiblity with older pickle files
    """

    def __init__(self, MCMC_fields):
        print(
            "Warning - Metrostate class is deprecated and will have no effect or functionality."
        )
        return


class Parameters:
    """
    Collection of parameters defining where the metropolis walker is right
    now.
    This class is deprecated - and available only for compatibility
    with older pickle files.
    """

    def __init__(self, param_info):
        print(
            "Warning - Parameters class is deprecated and will have no effect or functionality."
        )
        return


class Covariance:
    """
    The covariance matrix used to select the next trial move.
    This class is deprecated - and available only for compatibility
    with older pickle files.

    """

    def __init__(self, param_info):
        print(
            "Warning - Covariance class is deprecated and will have no effect or functionality."
        )
        return


class Grid:
    """
    Collection of values describing the grid dimensions for the simulation.
    """

    thickness: float  # Material thickness (maximum space coordinate)
    nx: int  # Number of space steps
    dx: float  # Space step size
    xSteps: np.ndarray  # Space node locations
    time: float  # Maximum delay time
    nt: int  # Number of time steps
    tSteps: np.ndarray  # Delay time values
    hmax: float  # Maximum time step size used by solver
    start_time: float  # Starting delay time
    final_time: float  # Starting delay time
    min_y: float  # Minimum value the simulated signal is allowed to reach before the sim terminates.

    def __init__(self, thickness, nx, tSteps, hmax):
        self.thickness = thickness
        self.nx = nx
        self.dx = self.thickness / self.nx
        self.xSteps = np.linspace(self.dx / 2, self.thickness - self.dx / 2, self.nx)

        if tSteps[0] != 0:
            raise ValueError("Grid error - times must start at t=0")
        self.tSteps = tSteps
        self.start_time = 0
        self.nt = len(tSteps) - 1
        self.hmax = hmax
        self.final_time = self.tSteps[-1]
        self.min_y = float_info.min
        return


class Solution:
    """
    This class is deprecated - and available only for compatibility
    with older pickle files.

    """

    def __init__(self):
        print(
            "Warning - Covariance class is deprecated and will have no effect or functionality."
        )
        return


# def check_threshold(t, y, L, dx, min_y=0, mode="TRPL", ks=0, mu_n=0, mu_p=0, n0=0, p0=0):
#     """Event - terminate integration if PL(t) is below the starting PL0=PL(t=0) * thr"""
#     N = y[0:L]
#     P = y[L:2*(L)]

#     if mode == "TRPL":
#         y_test = calculate_PL(dx, N, P, ks, n0, p0)
#     elif mode == "TRTS":
#         y_test = calculate_TRTS(dx, N, P, mu_n, mu_p, n0, p0)
#     else:
#         raise ValueError("Unsupported threshold mode")

#     if y_test <= min_y:
#         return 0
#     else:
#         return 1
