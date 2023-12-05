# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:04:20 2022

@author: cfai2
"""
from sys import float_info
import os
import pickle
import warnings

import numpy as np
from numba import njit

warnings.simplefilter("always", DeprecationWarning)
# Constants
eps0 = 8.854 * 1e-12 * 1e-9  # [C / V m] to {C / V nm}
q_C = 1.602e-19  # [C per carrier]
DEFAULT_ANN_STEP = np.sqrt(0.5)
DEFAULT_TEMPER_FREQ = 10

class Ensemble():
    """ Ensemble of MetroStates controlled by a single process.
    
    """
    iniPar: np.ndarray      # Initial conditions for simulations
    times: list[np.ndarray] # Measurement delay times
    vals: list[np.ndarray]  # Measurement values
    uncs: list[np.ndarray]  # Measurement uncertainties
    IRF_tables: dict        # Instrument response functions

    def __init__(self, param_info, sim_info, MCMC_fields, num_iters):
        self.ensemble_fields = {}
        self.ensemble_fields["output_path"] = MCMC_fields.pop("output_path")
        self.ensemble_fields["checkpoint_dirname"] = MCMC_fields.pop("checkpoint_dirname")
        if "checkpoint_header" in MCMC_fields:
            self.ensemble_fields["checkpoint_header"] = MCMC_fields.pop("checkpoint_header")
        self.ensemble_fields["checkpoint_freq"] = MCMC_fields.pop("checkpoint_freq")
        self.ensemble_fields["parallel_tempering"] = MCMC_fields.pop("parallel_tempering", None)
        self.ensemble_fields["temper_freq"] = MCMC_fields.pop("temper_freq", DEFAULT_TEMPER_FREQ)

        if self.ensemble_fields["parallel_tempering"] is None:
            n_states = 1
            temperatures = [1]
        else:
            n_states = len(self.ensemble_fields["parallel_tempering"])
            temperatures = self.ensemble_fields["parallel_tempering"]
        
        self.MS : list[MetroState] = []
        for i in range(n_states):
            self.MS.append(MetroState(param_info, dict(MCMC_fields), num_iters))
            self.MS[-1].MCMC_fields["_beta"] = temperatures[i] ** -1
            if isinstance(MCMC_fields["likel2move_ratio"], dict):
                self.MS[-1].MCMC_fields["current_sigma"] = {m:max(param_info["trial_move"].values()) * MCMC_fields["likel2move_ratio"][m]
                                                            for m in sim_info["meas_types"]}
            else:
                self.MS[-1].MCMC_fields["current_sigma"] = {m:max(param_info["trial_move"].values()) * MCMC_fields["likel2move_ratio"]
                                                            for m in sim_info["meas_types"]}
            
        self.ensemble_fields["do_parallel_tempering"] = (n_states > 1)

        self.sim_info = sim_info
        self.random_state = np.random.get_state()
        self.latest_iter = 0


    def checkpoint(self, fname):
        """ Save the current ensemble as a pickle object. """
        for MS in self.MS:
            MS.H.update(MS.param_info)

        with open(fname, "wb+") as ofstream:
            pickle.dump(self, ofstream)
        return


class MetroState():
    """ Overall management of the metropolis random walker: its current state,
        the states it's been to, and the trial move function used to get the
        next state.
    """
    def __init__(self, param_info, MCMC_fields, num_iters):
        self.H = History(num_iters, param_info)

        self.param_info = param_info
        self.MCMC_fields = MCMC_fields

        self.init_state = np.array([self.param_info["init_guess"][param]
                                    for param in self.param_info["names"]], dtype=float)
        
        self.param_indexes = {name: self.param_info["names"].index(name) for name in self.param_info["names"]}

        return

    def print_status(self, k, new_state, logger):
        is_active = self.param_info['active']

        logger.info(f"Current loglikelihood : {self.H.loglikelihood[0, k]:.6e}")
        for i, param in enumerate(self.param_info['names']):
            if is_active.get(param, 0):
                logger.info(f"Next {param}: {new_state[i]:.6e} from mean {self.H.states[i, k]:.6e}")

        return


class Parameters():
    """
    Collection of parameters defining where the metropolis walker is right
    now.
    This class is deprecated - and available only for compatibility
    with older pickle files.
    """
    def __init__(self, param_info):
        print("Warning - Parameters class is deprecated and will have no effect or functionality.")
        return


class History():
    """ Record of past states the walk has been to. """

    def __init__(self, num_iters, param_info):
        # for param in param_info["names"]:
        #     setattr(self, f"mean_{param}", np.zeros((1, num_iters)))
        self.states_are_one_array = True
        self.states = np.zeros((len(param_info["names"]), num_iters))
        self.accept = np.zeros((1, num_iters))
        self.loglikelihood = np.zeros((1, num_iters))
        return

    def update(self, param_info):
        """Compatibility - repackage self.states array into attributes per parameter"""
        for i, param in enumerate(param_info['names']):
            setattr(self, f"mean_{param}", self.states[i])
        return
    
    def pack(self, param_info, num_iters):
        """Compatibility - turn individual attributes into self.states"""
        self.states = np.zeros((len(param_info["names"]), num_iters))
        for k, param in enumerate(param_info["names"]):
            self.states[k] = getattr(self, f"mean_{param}")

    def truncate(self, k):
        """ Cut off any incomplete iterations should the walk be terminated early"""
        # for param in param_info["names"]:
        #     val = getattr(self, f"mean_{param}")
        #     setattr(self, f"mean_{param}", val[:, :k])

        self.states = self.states[:, :k]
        self.accept = self.accept[:, :k]
        self.loglikelihood = self.loglikelihood[:, :k]
        return

    def extend(self, new_num_iters):
        """ Enlarge an existing MC chain to length new_num_iters, if needed """
        current_num_iters = len(self.accept[0])
        if new_num_iters < current_num_iters:  # No extension needed
            self.truncate(new_num_iters)
            return
        if new_num_iters == current_num_iters:
            return

        addtl_iters = new_num_iters - current_num_iters
        self.accept = np.concatenate(
            (self.accept, np.zeros((1, addtl_iters))), axis=1)
        self.loglikelihood = np.concatenate(
            (self.loglikelihood, np.zeros((1, addtl_iters))), axis=1)
        
        self.states = np.concatenate(
            (self.states, np.zeros((self.states.shape[0], addtl_iters))), axis=1)

        # for param in param_info["names"]:
        #     val = getattr(self, f"mean_{param}")
        #     setattr(self, f"mean_{param}", np.concatenate(
        #         (val, np.zeros((1, addtl_iters))), axis=1))
        return


class Covariance():
    """
    The covariance matrix used to select the next trial move.
    This class is deprecated - and available only for compatibility
    with older pickle files.
    
    """

    def __init__(self, param_info):
        print("Warning - Covariance class is deprecated and will have no effect or functionality.")
        return

class Grid():
    """
    Collection of values describing the grid dimensions for the simulation.

    min_y : float
        Minimum value the simulated signal is allowed to reach before the sim terminates.
    """
    def __init__(self):
        self.min_y = float_info.min
        return


class Solution():
    def __init__(self):
        return

    def calculate_PL(self, g, ks, n0, p0):
        """ Time-resolved photoluminescence """
        self.PL = calculate_PL(g.dx, self.N, self.P, ks, n0, p0)

    def calculate_TRTS(self, g, mu_n, mu_p, n0, p0):
        """ Transient terahertz decay """
        self.trts = calculate_TRTS(g.dx, self.N, self.P, mu_n, mu_p, n0, p0)

def calculate_PL(dx, N, P, ks, n0, p0):
    rr = calculate_RR(N, P, ks, n0, p0)
    if rr.ndim == 2:
        PL = integrate_2D(dx, rr)
    elif rr.ndim == 1:
        PL = integrate_1D(dx, rr)
    else:
        raise ValueError(f"Invalid number of dims (got {rr.ndim} dims) in Solution")
    return PL

def calculate_TRTS(dx, N, P, mu_n, mu_p, n0, p0):
    trts = calculate_photoc(N, P, mu_n, mu_p, n0, p0)
    if trts.ndim == 2:
        trts = integrate_2D(dx, trts)
    elif trts.ndim == 1:
        trts = integrate_1D(dx, trts)
    else:
        raise ValueError(f"Invalid number of dims (got {trts.ndim} dims) in Solution")
    return trts

@njit(cache=True)
def integrate_2D(dx, y):
    y_int = np.zeros(len(y))
    for i in range(len(y)):
        y_int[i] = integrate_1D(dx, y[i])
    return y_int

@njit(cache=True)
def integrate_1D(dx, y):
    y_int = y[0] * dx / 2
    for i in range(1, len(y)):
        y_int += dx * (y[i] + y[i-1]) / 2
    y_int += y[-1] * dx / 2
    return y_int

@njit(cache=True)
def calculate_RR(N, P, ks, n0, p0):
    return ks * (N * P - n0 * p0)

@njit(cache=True)
def calculate_photoc(N, P, mu_n, mu_p, n0, p0):
    return q_C * (mu_n * (N - n0) + mu_p * (P - p0))

def check_threshold(t, y, L, dx, min_y=0, mode="TRPL", ks=0, mu_n=0, mu_p=0, n0=0, p0=0):
    """Event - terminate integration if PL(t) is below the starting PL0=PL(t=0) * thr"""
    N = y[0:L]
    P = y[L:2*(L)]

    if mode == "TRPL":
        y_test = calculate_PL(dx, N, P, ks, n0, p0)
    elif mode == "TRTS":
        y_test = calculate_TRTS(dx, N, P, mu_n, mu_p, n0, p0)
    else:
        raise ValueError("Unsupported threshold mode")

    if y_test <= min_y:
        return 0
    else:
        return 1
