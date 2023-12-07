# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:04:20 2022

@author: cfai2
"""
from sys import float_info
import pickle
import logging

import numpy as np

from forward_solver import solve
from utils import search_c_grps, set_min_y, unpack_simpar, U
from laplace import do_irf_convolution, post_conv_trim
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


class MetroState:
    """Overall management of the metropolis random walker: its current state,
    the states it's been to, and the trial move function used to get the
    next state.
    """

    def __init__(self, param_info, init_state, MCMC_fields, num_iters):
        self.H = History(num_iters, param_info["names"])

        self.param_info = param_info
        self.MCMC_fields = MCMC_fields

        self.init_state = init_state
        return

    def print_status(self, k, is_active, new_state, logger):
        logger.info(f"Current loglikelihood : {self.H.loglikelihood[0, k]:.6e}")
        for i, param in enumerate(self.param_info["names"]):
            if is_active[i]:
                logger.info(
                    f"Next {param}: {new_state[i]:.6e} from mean {self.H.states[i, k]:.6e}"
                )

        return


class EnsembleTemplate:
    """
    Base class for Ensembles
    """

    iniPar: np.ndarray  # Initial conditions for simulations
    times: list[np.ndarray]  # Measurement delay times
    vals: list[np.ndarray]  # Measurement values
    uncs: list[np.ndarray]  # Measurement uncertainties
    IRF_tables: dict  # Instrument response functions
    sim_info: dict  # Simulation settings
    ensemble_fields: dict  # Monte Carlo settings shared across all chains
    MS: list[MetroState]  # List of Monte Carlo chains
    param_indexes: dict[
        str, int
    ]  # Map of material parameter names and the order they appear in state arrays
    RNG: np.random.Generator  # Random number generator
    random_state: dict  # State of the RNG
    latest_iter: int  # Latest iteration # reached by chains
    logger: logging.Logger  # A standard logging.logger instance
    handler: logging.FileHandler  # A standard FileHandler instance

    def __init__(self):
        return

    def checkpoint(self, fname):
        """Save the current ensemble as a pickle object."""
        for MS in self.MS:
            MS.H.update(MS.param_info["names"])

        with open(fname, "wb+") as ofstream:
            handler = self.handler  # FileHandlers aren't pickleable
            self.handler = None
            pickle.dump(self, ofstream)
            self.handler = handler
        return

    def eval_trial_move(self, state, MCMC_fields):
        """
        Calculates log likelihood of a new proposed state
        """

        logll = np.zeros(self.sim_info["num_meas"])

        for i in range(self.sim_info["num_meas"]):
            logll[i] = self.one_sim_likelihood(i, state, MCMC_fields)

        logll = np.sum(logll)

        return logll

    def one_sim_likelihood(self, meas_index, state, MCMC_fields):
        """
        Calculates log likelihood of one measurement within a proposed state
        """
        iniPar = self.iniPar[meas_index]
        meas_type = self.sim_info["meas_types"][meas_index]
        irf_convolution = MCMC_fields.get("irf_convolution", None)

        ff = MCMC_fields.get("fittable_fluences", None)
        if ff is not None and meas_index in ff[1]:
            if ff[2] is not None and len(ff[2]) > 0:
                name = f"_f{search_c_grps(ff[2], meas_index)}"
            else:
                name = f"_f{meas_index}"
            iniPar[0] *= state[self.param_indexes[name]]
        fa = MCMC_fields.get("fittable_absps", None)
        if fa is not None and meas_index in fa[1]:
            if fa[2] is not None and len(fa[2]) > 0:
                name = f"_a{search_c_grps(fa[2], meas_index)}"
            else:
                name = f"_a{meas_index}"
            iniPar[1] *= state[self.param_indexes[name]]
        fs = MCMC_fields.get("scale_factor", None)
        if fs is not None and meas_index in fs[1]:
            if fs[2] is not None and len(fs[2]) > 0:
                name = f"_s{search_c_grps(fs[2], meas_index)}"
            else:
                name = f"_s{meas_index}"
            scale_shift = np.log10(state[self.param_indexes[name]])
        else:
            scale_shift = 0

        if meas_type == "pa":
            tSteps = np.array([0])
            sol = np.array([U(state[0])])
            success = True
        else:
            tSteps, sol, success = self.converge_simulation(
                meas_index, state, iniPar
            )
        if not success:
            likelihood = -np.inf
            return likelihood

        try:
            if irf_convolution is not None and irf_convolution[meas_index] != 0:
                self.logger.debug(
                    f"Convolving with wavelength {irf_convolution[meas_index]}"
                )
                wave = int(irf_convolution[meas_index])
                tSteps, sol, success = do_irf_convolution(
                    tSteps, sol, self.IRF_tables[wave], time_max_shift=True
                )
                if not success:
                    raise ValueError(
                        "Conv failed. Check measurement data times for floating-point inaccuracies.\n"
                        "This may also happen if simulated signal decays extremely slowly."
                    )
                sol, times_c, vals_c, uncs_c = post_conv_trim(
                    tSteps,
                    sol,
                    self.times[meas_index],
                    self.vals[meas_index],
                    self.uncs[meas_index],
                )

            else:
                # Still need to trim, in case experimental data doesn't start at t=0
                times_c = self.times[meas_index]
                vals_c = self.vals[meas_index]
                uncs_c = self.uncs[meas_index]
                sol = sol[-len(times_c) :]

        except ValueError as e:
            self.logger.warning(e)
            likelihood = -np.inf
            return likelihood

        self.logger.debug(f"Comparing times {times_c[0]}-{times_c[-1]}")

        try:
            # TRPL must be positive!
            # Any simulation which results in depleted carrier is clearly incorrect
            # A few negative values may also be introduced during convolution -
            # so we want to tolerate these, while too many suggests that depletion
            # is happening instead

            where_failed = sol < 0
            n_fails = np.sum(where_failed)
            success = n_fails < NEGATIVE_FRAC_TOL * len(sol)
            if not success:
                raise ValueError(
                    f"{meas_index}: Simulation failed: too many negative vals"
                )

            if n_fails > 0:
                self.logger.warning(
                    f"{meas_index}: {n_fails} / {len(sol)} non-positive vals"
                )

            sol[where_failed] *= -1
        except ValueError as e:
            self.logger.warning(e)
            likelihood = -np.inf
            return likelihood

        if MCMC_fields.get("force_min_y", False):
            sol, min_y, n_set = set_min_y(sol, vals_c, scale_shift)
            self.logger.debug(f"min_y: {min_y}")
            if n_set > 0:
                self.logger.debug(f"{n_set} values raised to min_y")

        if meas_type == "pa":
            likelihood = -sol[0]
        else:
            try:
                err_sq = (np.log10(sol) + scale_shift - vals_c) ** 2

                # Compatibility with single sigma
                if isinstance(MCMC_fields["current_sigma"], dict):
                    likelihood = -np.sum(
                        err_sq
                        / (
                            MCMC_fields["current_sigma"][meas_type] ** 2
                            + 2 * uncs_c**2
                        )
                    )
                else:
                    likelihood = -np.sum(
                        err_sq / (MCMC_fields["current_sigma"] ** 2 + 2 * uncs_c**2)
                    )

                if np.isnan(likelihood):
                    raise ValueError(
                        f"{meas_index}: Simulation failed: invalid likelihood"
                    )
            except ValueError as e:
                self.logger.warning(e)
                likelihood = -np.inf
        return likelihood

    def converge_simulation(self, meas_index, state, init_conds):
        """
        Handle mishaps from do_simulation.

        Parameters
        ----------
        meas_index : int
            Index of ith simulation in a measurement set requiring n simulations.
        state : ndarray
            An array of parameters, ordered according to param_info["names"],
            corresponding to a state in the parameter space.
        init_conds : ndarray
            Array of initial conditions (e.g. an initial carrier profile) for simulation.

        Returns
        -------
        sol : ndarray
            Array of values (e.g. TRPL) from final simulation.
        tSteps : ndarray
            Array of times the final simulation was evaluated at.
        success : bool
            Whether the final simulation passed all convergence criteria.

        """
        success = True

        t_steps = np.array(self.times[meas_index])
        sol = np.zeros_like(t_steps)

        try:
            t_steps, sol = self.do_simulation(meas_index, state, init_conds)
        except ValueError as e:
            success = False
            self.logger.warning(f"{meas_index}: Simulation error occurred: {e}")
            return t_steps, sol, success

        self.logger.debug(
            f"{meas_index}: Simulation complete t {t_steps[0]}-{t_steps[-1]}"
        )

        return t_steps, sol, success

    def do_simulation(self, meas_index, state, init_conds):
        """Set up and run one simulation."""
        thickness, nx, meas_type = unpack_simpar(self.sim_info, meas_index)
        g = Grid(thickness, nx, self.times[meas_index], self.ensemble_fields["hmax"])

        sol = solve(
            init_conds,
            g,
            state,
            self.param_indexes,
            meas=meas_type,
            units=self.ensemble_fields["units"],
            solver=self.ensemble_fields["solver"],
            model=self.ensemble_fields["model"],
            RTOL=self.ensemble_fields["rtol"],
            ATOL=self.ensemble_fields["atol"],
        )
        return g.tSteps, sol

    def check_approved_param(self, new_state, param_info):
        """ Raise a warning for non-physical or unrealistic proposed trial moves,
            or proposed moves that exceed the prior distribution.
        """
        order = param_info['names']
        checks = {}
        prior_dist = self.ensemble_fields["prior_dist"]

        # Ensure proposal stays within bounds of prior distribution
        diff = np.where(self.ensemble_fields["do_log"], 10 ** new_state, new_state)
        for i, param in enumerate(order):
            if not self.ensemble_fields["active"][i]:
                continue

            lb = prior_dist[param][0]
            ub = prior_dist[param][1]
            checks[f"{param}_size"] = (lb < diff[i] < ub)

        # TRPL specific checks:
        # p0 > n0 by definition of a p-doped material
        if 'p0' in order and 'n0' in order:
            checks["p0_greater"] = (new_state[self.param_indexes["p0"]]
                                    > new_state[self.param_indexes["n0"]])
        else:
            checks["p0_greater"] = True

        # tau_n and tau_p must be *close* (within 2 OM) for a reasonable midgap SRH
        if 'tauN' in order and 'tauP' in order:
            # Compel logscale for this one - makes for easier check
            logtn = new_state[self.param_indexes['tauN']]
            if not self.ensemble_fields["do_log"][self.param_indexes["tauN"]]:
                logtn = np.log10(logtn)

            logtp = new_state[self.param_indexes['tauP']]
            if not self.ensemble_fields["do_log"][self.param_indexes["tauP"]]:
                logtp = np.log10(logtp)

            diff = np.abs(logtn - logtp)
            checks["tn_tp_close"] = (diff <= 2)

        else:
            checks["tn_tp_close"] = True

        failed_checks = [k for k in checks if not checks[k]]

        return failed_checks

    def select_next_params(self, current_state, param_info):
        """ 
        Trial move function: returns a new proposed state equal to the current_state plus a uniform random displacement
        """

        _current_state = np.array(current_state, dtype=float)

        mu_constraint = self.ensemble_fields.get("do_mu_constraint", None)

        _current_state = np.where(self.ensemble_fields["do_log"],
                                  np.log10(_current_state),
                                  _current_state)

        tries = 0

        # Try up to MAX_PROPOSALS times to come up with a proposal that stays within
        # the hard boundaries, if we ask
        if self.ensemble_fields.get("hard_bounds", 0):
            max_tries = MAX_PROPOSALS
        else:
            max_tries = 1

        new_state = np.array(_current_state)
        while tries < max_tries:
            tries += 1

            new_state = np.where(self.ensemble_fields["active"],
                                 self.RNG.uniform(_current_state-self.ensemble_fields["trial_move"],
                                                  _current_state+self.ensemble_fields["trial_move"]),
                                 _current_state)

            if mu_constraint is not None:
                ambi = mu_constraint[0]
                ambi_std = mu_constraint[1]
                self.logger.debug(f"mu constraint: ambi {ambi} +/- {ambi_std}")
                new_muambi = np.random.uniform(ambi - ambi_std, ambi + ambi_std)
                new_state[self.param_indexes["mu_p"]] = np.log10(
                    (2 / new_muambi - 1 / 10 ** new_state[self.param_indexes["mu_n"]])**-1)

            failed_checks = self.check_approved_param(new_state, param_info)
            success = len(failed_checks) == 0
            if success:
                self.logger.debug(f"Found params in {tries} tries")
                break

            if len(failed_checks) > 0:
                self.logger.warning(f"Failed checks: {failed_checks}")

        new_state = np.where(self.ensemble_fields["do_log"], 10 ** new_state, new_state)
        return new_state


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
        self.ensemble_fields["output_path"] = MCMC_fields.pop("output_path")
        self.ensemble_fields["checkpoint_dirname"] = MCMC_fields.pop(
            "checkpoint_dirname"
        )
        if "checkpoint_header" in MCMC_fields:
            self.ensemble_fields["checkpoint_header"] = MCMC_fields.pop(
                "checkpoint_header"
            )
        self.ensemble_fields["checkpoint_freq"] = MCMC_fields.pop("checkpoint_freq")
        self.ensemble_fields["parallel_tempering"] = MCMC_fields.pop(
            "parallel_tempering", None
        )
        self.ensemble_fields["temper_freq"] = MCMC_fields.pop(
            "temper_freq", DEFAULT_TEMPER_FREQ
        )
        self.ensemble_fields["solver"] = MCMC_fields.pop("solver")
        self.ensemble_fields["model"] = MCMC_fields.pop("model")
        self.ensemble_fields["hard_bounds"] = MCMC_fields.pop("hard_bounds", 0)
        self.ensemble_fields["rtol"] = MCMC_fields.pop("rtol", None)
        self.ensemble_fields["atol"] = MCMC_fields.pop("atol", None)
        self.ensemble_fields["hmax"] = MCMC_fields.pop("hmax", DEFAULT_HMAX)

        self.ensemble_fields["do_mu_constraint"] = param_info.pop("do_mu_constraint", None)
        self.ensemble_fields["prior_dist"] = param_info.pop("prior_dist")
        # Transfer shared fields, as arrays ordered by param_info["names"]
        self.ensemble_fields["do_log"] = param_info.pop("do_log")
        self.ensemble_fields["do_log"] = np.array(
            [self.ensemble_fields["do_log"][param] for param in param_info["names"]],
            dtype=bool,
        )

        self.ensemble_fields["trial_move"] = param_info.pop("trial_move")
        self.ensemble_fields["trial_move"] = np.array(
            [
                self.ensemble_fields["trial_move"][param]
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

        self.param_indexes = {
            name: param_info["names"].index(name) for name in param_info["names"]
        }

        if self.ensemble_fields["parallel_tempering"] is None:
            n_states = 1
            temperatures = [1]
        else:
            n_states = len(self.ensemble_fields["parallel_tempering"])
            temperatures = self.ensemble_fields["parallel_tempering"]

        self.MS: list[MetroState] = []
        for i in range(n_states):
            init_state = np.array(
                [
                    param_info["init_guess"][param]
                    for param in param_info["names"]
                ],
            dtype=float,
        )
            self.MS.append(MetroState(param_info, init_state, dict(MCMC_fields), num_iters))
            self.MS[-1].MCMC_fields["_beta"] = temperatures[i] ** -1
            if isinstance(MCMC_fields["likel2move_ratio"], dict):
                self.MS[-1].MCMC_fields["current_sigma"] = {
                    m: max(self.ensemble_fields["trial_move"])
                    * MCMC_fields["likel2move_ratio"][m]
                    for m in sim_info["meas_types"]
                }
            else:
                self.MS[-1].MCMC_fields["current_sigma"] = {
                    m: max(self.ensemble_fields["trial_move"])
                    * MCMC_fields["likel2move_ratio"]
                    for m in sim_info["meas_types"]
                }

        self.ensemble_fields["do_parallel_tempering"] = n_states > 1

        self.sim_info = sim_info
        self.RNG = np.random.default_rng(235817049752375780)
        self.random_state = np.random.get_state()
        self.latest_iter = 0

    def stop_logging(self, err_code):
        stop_logging(self.logger, self.handler, err_code)


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


class History:
    """Record of past states the walk has been to."""

    def __init__(self, num_iters, names):
        self.states_are_one_array = True
        self.states = np.zeros((len(names), num_iters))
        self.accept = np.zeros((1, num_iters))
        self.loglikelihood = np.zeros((1, num_iters))
        return

    def update(self, names):
        """Compatibility - repackage self.states array into attributes per parameter"""
        for i, param in enumerate(names):
            setattr(self, f"mean_{param}", self.states[i])
        return

    def pack(self, names, num_iters):
        """Compatibility - turn individual attributes into self.states"""
        self.states = np.zeros((len(names), num_iters))
        for k, param in enumerate(names):
            self.states[k] = getattr(self, f"mean_{param}")

    def truncate(self, k):
        """Cut off any incomplete iterations should the walk be terminated early"""
        self.states = self.states[:, :k]
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
        self.accept = np.concatenate((self.accept, np.zeros((1, addtl_iters))), axis=1)
        self.loglikelihood = np.concatenate(
            (self.loglikelihood, np.zeros((1, addtl_iters))), axis=1
        )

        self.states = np.concatenate(
            (self.states, np.zeros((self.states.shape[0], addtl_iters))), axis=1
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
