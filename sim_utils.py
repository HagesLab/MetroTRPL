# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:04:20 2022

@author: cfai2
"""
from sys import float_info
import os
import pickle
import numpy as np
from numba import njit
# Constants
eps0 = 8.854 * 1e-12 * 1e-9  # [C / V m] to {C / V nm}
q_C = 1.602e-19  # [C per carrier]
DEFAULT_ANN_STEP = np.sqrt(0.5)

class MetroState():
    """ Overall management of the metropolis random walker: its current state,
        the states it's been to, and the trial move function used to get the
        next state.
    """
    sim_info: dict
    iniPar: np.ndarray
    times: list[np.ndarray]
    vals: list[np.ndarray]
    uncs: list[np.ndarray]
    IRF_tables: dict
    def __init__(self, param_info, MCMC_fields, num_iters):
        self.p = Parameters(param_info)

        self.H = History(num_iters, param_info)

        self.prev_p = Parameters(param_info)

        self.means = Parameters(param_info)

        self.variances = Covariance(param_info)
        self.variances.apply_values(param_info["init_variance"])

        self.param_info = param_info
        self.MCMC_fields = MCMC_fields
        self.MCMC_fields["current_sigma"] = dict(self.MCMC_fields["annealing"][0])
        self.latest_iter = 0
        self.random_state = np.random.get_state()
        return

    def anneal(self, k, uncs=None, force=False, step=DEFAULT_ANN_STEP):
        """ "Adjust the model sigma according to an annealing schedule -
            sigma *= sqrt(0.5) every kth step
        """
        steprate = self.MCMC_fields["annealing"][1]
        min_sigma = self.MCMC_fields["annealing"][2]
        l2v = self.MCMC_fields["likel2variance_ratio"]
        if force or (k > 0 and k % steprate == 0):

            for m in self.MCMC_fields["current_sigma"]:
                self.MCMC_fields["current_sigma"][m] *= step

                self.MCMC_fields["current_sigma"][m] = max(self.MCMC_fields["current_sigma"][m],
                                                        min_sigma[m])

            # Doesn't matter which meas_type, because all current_sigma are proportional to init_variance
            random_m = next(iter(self.MCMC_fields["current_sigma"].keys()))
            new_variance = self.MCMC_fields["current_sigma"][random_m] / l2v[random_m]
            self.variances.apply_values(
                {param: new_variance for param in self.param_info["names"]})

            # Recalculate the previous state's likelihood, for consistency
            meas_types = self.sim_info["meas_types"]
            for i in range(len(self.prev_p.likelihood)):
                if uncs is not None:
                    exp_unc = 2 * uncs[i] ** 2
                else:
                    exp_unc = 0
                new_uncertainty = self.MCMC_fields["current_sigma"][meas_types[i]]**2 + exp_unc
                self.prev_p.likelihood[i] = - \
                    np.sum(self.prev_p.err_sq[i] / new_uncertainty)
        return

    def print_status(self, logger):
        is_active = self.param_info['active']

        if hasattr(self.prev_p, "likelihood"):
            logger.info("Current loglikelihood : {:.6e} ".format(
                np.sum(self.prev_p.likelihood)))
        for param in self.param_info['names']:
            if is_active.get(param, 0):
                trial = getattr(self.p, param)
                mean = getattr(self.means, param)
                logger.info("Next {}: {:.6e} from mean {:.6e}".format(param, trial, mean))

        return

    def checkpoint(self, fname):
        """ Save the current state as a pickle object. """
        with open(fname, "wb+") as ofstream:
            pickle.dump(self, ofstream)
        return


class Parameters():
    """ Collection of parameters defining where the metropolis walker is right
        now. For the OneLayer (single isolated absorber) carrier dynamics model,
        these parameters are:
    """
    Sf: float      # Front surface recombination velocity
    Sb: float      # Back surface recombination velocity
    mu_n: float    # Electron mobility
    mu_p: float    # Hole mobility
    n0: float      # Electron doping level
    p0: float      # Hole doping level
    B: float       # Radiative recombination rate
    Cn: float      # Auger coef for two-electron one-hole
    Cp: float      # Auger coef for two-hole one-electron
    tauN: float    # Electron bulk nonradiative decay lifetime
    tauP: float    # Hole bulk nonradiative decayl lifetime
    eps: float     # Relative dielectric cofficient
    Tm: float      # Temperature
    likelihood: list # Current likelihood of each simulation vs its respective measurement
    err_sq: list     # Current squared error of each simulation vs its respective measurement

    def __init__(self, param_info):
        self.param_names = param_info["names"]
        self.ucs = param_info.get("unit_conversions", dict[str, float]())
        self.actives = [(param, index) for index, param in enumerate(self.param_names)
                        if param_info["active"].get(param, False)]

        for param in self.param_names:
            if hasattr(self, param):
                raise KeyError(f"Param with name {param} already exists")
            setattr(self, param, param_info["init_guess"][param])


        return

    def apply_unit_conversions(self, reverse=False):
        """ Multiply the currently stored parameters according to a stored
            unit conversion dictionary.
        """
        for param in self.param_names:
            val = getattr(self, param)
            if reverse:
                setattr(self, param, val / self.ucs.get(param, 1))
            else:
                setattr(self, param, val * self.ucs.get(param, 1))
        return

    def make_log(self, param_info=None):
        """ Convert currently stored parameters to log space.
            This is nearly always recommended for TRPL, as TRPL decay can span
            many orders of magnitude.
        """
        for param in self.param_names:
            if param_info["do_log"].get(param, 0) and hasattr(self, param):
                val = getattr(self, param)
                setattr(self, param, np.log10(val))
        return

    def to_array(self, param_info=None):
        """ Compress the currently stored parameters into a 1D array. Some
            operations are easier with matrix operations while others are more
            intuitive when params are callable by name.
        """
        arr = np.array([getattr(self, param)
                       for param in self.param_names], dtype=float)
        return arr

    def transfer_from(self, sender, param_info):
        """ Update this Parameters() stored parameters with values from another
            Parameters(). """
        for param in param_info['names']:
            setattr(self, param, getattr(sender, param))
        return

    def suppress_scale_factor(self, scale_info, i):
        """ Force _s scale factors to 1 if they aren't needed, such as if self_normalize
            is used.
        """
        if scale_info is None:
            return
        setattr(self, f"_s{i}", 1)
        return


class Covariance():
    """ The covariance matrix used to select the next trial move. """

    def __init__(self, param_info):
        self.names = param_info["names"]
        self.actives = param_info['active']
        d = len(self.names)
        self.cov = np.zeros((d, d))
        return

    def set_variance(self, param, var):
        """ Update the variance of one parameter, telling the trial move
            function at most how far away the next state should wander
            from the current state.
        """
        # Type safety - names could be an ndarray which lacks the .index mtd
        i = list(self.names).index(param)

        if isinstance(var, (int, float)):
            self.cov[i, i] = var
        elif isinstance(var, dict):
            self.cov[i, i] = var[param]
        return

    def trace(self):
        return np.diag(self.cov)

    def apply_values(self, initial_variance):
        """ Initialize the covariance matrix for active paramters. Inactive
            parameters are assigned a variance of zero, preventing the walk from
            ever moving in their direction.

            The little-sigma big-sigma decomposition is needed for some
            adaptive covariance MC algorithms and also preserves the original
            cov after mask_covariance().
        """
        for param in self.names:
            if self.actives[param]:
                self.set_variance(param, initial_variance)

        iv_arr = 0
        if isinstance(initial_variance, dict):
            iv_arr = np.ones(len(self.cov))
            for i, param in enumerate(self.names):
                if self.actives[param]:
                    iv_arr[i] = initial_variance[param]

        elif isinstance(initial_variance, (float, int)):
            iv_arr = initial_variance

        self.little_sigma = np.ones(len(self.cov)) * iv_arr
        self.big_sigma = self.cov * iv_arr**-1
        return

    def mask_covariance(self, picked_param):
        """ Induce a univariate gaussian if doing one-param-at-a-time
            picked_param = tuple(param_name, its index)
        """
        if picked_param is None:
            self.cov = self.little_sigma * self.big_sigma
        else:
            i = picked_param[1]
            self.cov = np.zeros_like(self.cov)
            self.cov[i, i] = self.little_sigma[i] * self.big_sigma[i, i]
        return


class History():
    """ Record of past states the walk has been to. """

    def __init__(self, num_iters, param_info):
        for param in param_info["names"]:
            setattr(self, f"mean_{param}", np.zeros((1, num_iters)))

        self.accept = np.zeros((1, num_iters))
        self.loglikelihood = np.zeros((1, num_iters))
        return

    def record_best_logll(self, k, prev_p):
        # prev_p is essentially the latest accepted move
        self.loglikelihood[0, k] = np.sum(prev_p.likelihood)
        return

    def update(self, k, p, means, param_info):
        for param in param_info['names']:
            h_mean = getattr(self, f"mean_{param}")
            h_mean[0, k] = getattr(means, param)
        return

    def export(self, param_info, out_pathname):
        for param in param_info["names"]:
            np.save(os.path.join(out_pathname, f"mean_{param}"), getattr(
                self, f"mean_{param}"))

        np.save(os.path.join(out_pathname, "accept"), self.accept)
        np.save(os.path.join(out_pathname, "loglikelihood"), self.loglikelihood)
        return

    def truncate(self, k, param_info):
        """ Cut off any incomplete iterations should the walk be terminated early"""
        for param in param_info["names"]:
            val = getattr(self, f"mean_{param}")
            setattr(self, f"mean_{param}", val[:, :k])

        self.accept = self.accept[:, :k]
        self.loglikelihood = self.loglikelihood[:, :k]
        return

    def extend(self, new_num_iters, param_info):
        """ Enlarge an existing MC chain to length new_num_iters, if needed """
        current_num_iters = len(self.accept[0])
        if new_num_iters < current_num_iters:  # No extension needed
            self.truncate(new_num_iters, param_info)
            return
        if new_num_iters == current_num_iters:
            return

        addtl_iters = new_num_iters - current_num_iters
        self.accept = np.concatenate(
            (self.accept, np.zeros((1, addtl_iters))), axis=1)
        self.loglikelihood = np.concatenate(
            (self.loglikelihood, np.zeros((1, addtl_iters))), axis=1)

        for param in param_info["names"]:
            val = getattr(self, f"mean_{param}")
            setattr(self, f"mean_{param}", np.concatenate(
                (val, np.zeros((1, addtl_iters))), axis=1))
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

    def calculate_PL(self, g, p):
        """ Time-resolved photoluminescence """
        self.PL = calculate_PL(g.dx, self.N, self.P, p.ks, p.n0, p.p0)

    def calculate_TRTS(self, g, p):
        """ Transient terahertz decay """
        self.trts = calculate_TRTS(g.dx, self.N, self.P, p.mu_n, p.mu_p, p.n0, p.p0)

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

if __name__ == "__main__":
    S = Solution()
    g = Grid()
    g.dx = 20
    p = Grid()
    p.mu_n = 20 * 1e5
    p.mu_p = 20 * 1e5
    S.N = 1e15 * np.ones((1, 100)) * 1e-21
    S.P = 1e15 * np.ones((1, 100)) * 1e-21
    p.n0 = 1e8 * 1e-21
    p.p0 = 1e13 * 1e-21
    S.calculate_TRTS(g, p)
    print(S.trts)
