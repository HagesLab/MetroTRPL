# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:04:20 2022

@author: cfai2
"""
import numpy as np
from scipy.integrate import trapz
import os
import pickle
from numba import njit
# Constants
eps0 = 8.854 * 1e-12 * 1e-9  # [C / V m] to {C / V nm}
q_C = 1.602e-19  # [C per carrier]


class MetroState():
    """ Overall management of the metropolis random walker: its current state,
        the states it's been to, and the trial move function used to get the
        next state.
    """

    def __init__(self, param_info, MCMC_fields, num_iters):
        self.p = Parameters(param_info)
        self.p.apply_unit_conversions(param_info)

        self.H = History(num_iters, param_info)

        self.prev_p = Parameters(param_info)
        self.prev_p.apply_unit_conversions(param_info)

        self.means = Parameters(param_info)
        self.means.apply_unit_conversions(param_info)

        self.variances = Covariance(param_info)
        self.variances.apply_values(param_info["init_variance"])

        self.param_info = param_info
        self.MCMC_fields = MCMC_fields
        self.MCMC_fields["current_sigma"] = self.MCMC_fields["annealing"][0]
        return

    def anneal(self, k, uncs=None):
        steprate = self.MCMC_fields["annealing"][1]
        min_sigma = self.MCMC_fields["annealing"][2]
        l2v = self.MCMC_fields["likel2variance_ratio"]
        if k > 0 and l2v > 0 and k % steprate == 0:

            self.MCMC_fields["current_sigma"] *= 0.1

            self.MCMC_fields["current_sigma"] = max(self.MCMC_fields["current_sigma"],
                                                    min_sigma)

            new_variance = self.MCMC_fields["current_sigma"] / l2v
            self.variances.apply_values(
                {param: new_variance for param in self.param_info["names"]})

            # Ensure we aren't comparing states calculated with two different
            # likelihoods

            for i in range(len(self.prev_p.likelihood)):
                if uncs is not None:
                    exp_unc = 2 * uncs[i] ** 2
                else:
                    exp_unc = 0
                new_uncertainty = self.MCMC_fields["current_sigma"]**2 + exp_unc
                self.prev_p.likelihood[i] = - \
                    np.sum(self.prev_p.err_sq[i] / new_uncertainty)
        return

    def print_status(self, logger):
        is_active = self.param_info['active']
        ucs = self.param_info["unit_conversions"]

        if hasattr(self.prev_p, "likelihood"):
            logger.info("Current loglikelihood : {:.6e} ".format(
                np.sum(self.prev_p.likelihood)))
        for param in self.param_info['names']:
            if is_active.get(param, 0):
                logger.info("Next {}: {:.6e} from mean {:.6e}".format(param, getattr(
                    self.p, param) / ucs.get(param, 1), getattr(self.means, param) / ucs.get(param, 1)))

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

    def __init__(self, param_info):
        self.param_names = param_info["names"]
        self.actives = [(param, index) for index, param in enumerate(self.param_names)
                        if param_info["active"].get(param, False)]

        for param in self.param_names:
            if hasattr(self, param):
                raise KeyError(f"Param with name {param} already exists")
            setattr(self, param, param_info["init_guess"][param])

        # Global scale factor is an optional fitting param - if not defined,
        # default to x1
        if "m" not in self.param_names:
            setattr(self, "m", 1)
        return

    def apply_unit_conversions(self, param_info=None):
        """ Multiply the currently stored parameters according to a provided
            unit conversion dictionary.
        """
        for param in self.param_names:
            val = getattr(self, param)
            setattr(self, param, val *
                    param_info["unit_conversions"].get(param, 1))

    def make_log(self, param_info=None):
        """ Convert currently stored parameters to log space. 
            This is nearly always recommended for TRPL, as TRPL decay can span
            many orders of magnitude.
        """
        for param in self.param_names:
            if param_info["do_log"].get(param, 0) and hasattr(self, param):
                val = getattr(self, param)
                setattr(self, param, np.log10(val))

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


class History():
    """ Record of past states the walk has been to. """

    def __init__(self, num_iters, param_info):
        """ param referring to all proposed trial moves, including rejects,
            and mean_param referring to the state after each iteration.

            If a lot of moves get rejected, mean_param will record the same
            value for a while while param will record all the rejected moves.

            We need a better name for this.
        """
        for param in param_info["names"]:
            setattr(self, param, np.zeros(num_iters))
            setattr(self, f"mean_{param}", np.zeros(num_iters))

        self.accept = np.zeros(num_iters)
        self.loglikelihood = np.zeros(num_iters)

    def record_loglikelihood(self, k, p):
        self.loglikelihood[k] = np.sum(p.likelihood)
        return

    def update(self, k, p, means, param_info):
        self.loglikelihood[k] = np.sum(p.likelihood)

        for param in param_info['names']:
            # Proposed states
            h = getattr(self, param)
            h[k] = getattr(p, param)
            # Accepted states
            h_mean = getattr(self, f"mean_{param}")
            h_mean[k] = getattr(means, param)
        return

    def apply_unit_conversions(self, param_info):
        for param in param_info["names"]:
            val = getattr(self, param)
            setattr(self, param, val /
                    param_info["unit_conversions"].get(param, 1))

            val = getattr(self, f"mean_{param}")
            setattr(self, f"mean_{param}", val /
                    param_info["unit_conversions"].get(param, 1))

    def export(self, param_info, out_pathname):
        for param in param_info["names"]:
            np.save(os.path.join(out_pathname,
                    f"{param}"), getattr(self, param))
            np.save(os.path.join(out_pathname, f"mean_{param}"), getattr(
                self, f"mean_{param}"))

        np.save(os.path.join(out_pathname, "accept"), self.accept)
        np.save(os.path.join(out_pathname, "loglikelihood"), self.loglikelihood)
        #np.save(os.path.join(out_pathname, "final_cov"), self.final_cov)

    def truncate(self, k, param_info):
        """ Cut off any incomplete iterations should the walk be terminated early"""
        for param in param_info["names"]:
            val = getattr(self, param)
            setattr(self, param, val[:k])

            val = getattr(self, f"mean_{param}")
            setattr(self, f"mean_{param}", val[:k])

        self.accept = self.accept[:k]
        self.loglikelihood = self.loglikelihood[:k]
        return

    def extend(self, new_num_iters, param_info):
        """ Enlarge an existing MC chain to length new_num_iters, if needed """
        current_num_iters = len(self.accept)
        if new_num_iters <= current_num_iters:  # No extension needed
            return

        addtl_iters = new_num_iters - current_num_iters
        self.accept = np.concatenate(
            (self.accept, np.zeros(addtl_iters)), axis=0)
        self.loglikelihood = np.concatenate(
            (self.loglikelihood, np.zeros(addtl_iters)), axis=0)

        for param in param_info["names"]:
            val = getattr(self, param)
            setattr(self, param, np.concatenate(
                (val, np.zeros(addtl_iters)), axis=0))

            val = getattr(self, f"mean_{param}")
            setattr(self, f"mean_{param}", np.concatenate(
                (val, np.zeros(addtl_iters)), axis=0))
        return


class Grid():
    def __init__(self):
        return


class Solution():
    def __init__(self):
        return

    def calculate_PL(self, g, p):
        """ Time-resolved photoluminescence """
        rr = calculate_RR(self.N, self.P, p.ks, p.n0, p.p0)

        self.PL = trapz(rr, dx=g.dx, axis=1)
        self.PL += rr[:, 0] * g.dx / 2
        self.PL += rr[:, -1] * g.dx / 2

    def calculate_TRTS(self, g, p):
        """ Transient terahertz decay """
        trts = p.mu_n * (self.N - p.n0) + p.mu_p * (self.P - p.p0)
        self.trts = trapz(trts, dx=g.dx, axis=1)
        self.trts += trts[:, 0] * g.dx / 2
        self.trts += trts[:, -1] * g.dx / 2


@njit(cache=True)
def calculate_RR(N, P, ks, n0, p0):
    return ks * (N * P - n0 * p0)
