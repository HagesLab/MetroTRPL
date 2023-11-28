# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 22:13:26 2022

@author: cfai2
"""
import os
import sys
import signal
import pickle
import numpy as np
from scipy.integrate import solve_ivp, odeint

from forward_solver import MODELS
from sim_utils import MetroState, Grid, Solution
from mcmc_logging import start_logging, stop_logging
from bayes_io import make_dir
from laplace import make_I_tables, do_irf_convolution, post_conv_trim
try:
    from nn_features import NeuralNetwork
    HAS_NN_LIB = True
    nn = NeuralNetwork()
except ImportError:
    HAS_NN_LIB = False

# Constants
eps0 = 8.854 * 1e-12 * 1e-9  # [C / V m] to {C / V nm}
q = 1.0  # [e]
q_C = 1.602e-19  # [C]
kB = 8.61773e-5  # [eV / K]
DEFAULT_RTOL = 1e-7
DEFAULT_ATOL = 1e-10
DEFAULT_HMAX = 4
MAX_PROPOSALS = 100
MSG_FREQ = 100
MSG_COOLDOWN = 3 # Log first few states regardless of verbose

# Allow this proportion of simulation values to become negative due to convolution,
# else the simulation is failed.
NEGATIVE_FRAC_TOL = 0.2

def U(x):
    return 1000 * (x < -2) + 1 * (1 + np.sin(2*np.pi*x)) * np.logical_and(-2 <= x, x <= -1.25) \
                             + 2 * (1 + np.sin(2*np.pi*x)) * np.logical_and(-1.25 <= x, x <= -0.25) \
                             + 3 * (1 + np.sin(2*np.pi*x)) * np.logical_and(-0.25 <= x, x <= 0.75) \
                             + 4 * (1 + np.sin(2*np.pi*x)) * np.logical_and(0.75 <= x, x <= 1.75) \
                             + 5 * (1 + np.sin(2*np.pi*x)) * np.logical_and(1.75 <= x, x <= 2) \
                             + 1000 * (x > 2)


def E_field(N, P, PA, dx, corner_E=0):
    if N.ndim == 1:
        E = corner_E + q_C / (PA.eps * eps0) * dx * \
            np.cumsum(((P - PA.p0) - (N - PA.n0)))
        E = np.concatenate(([corner_E], E))
    elif N.ndim == 2:
        E = corner_E + q_C / (PA.eps * eps0) * dx * \
            np.cumsum(((P - PA.p0) - (N - PA.n0)), axis=1)
        num_tsteps = len(N)
        E = np.concatenate((np.ones(shape=(num_tsteps, 1))*corner_E, E), axis=1)
    return E


def solve(iniPar, g, p, meas="TRPL", solver=("solveivp",), model="std",
          RTOL=DEFAULT_RTOL, ATOL=DEFAULT_ATOL):
    """
    Calculate one simulation. Outputs in same units as measurement data,
    ([cm, V, s]) for PL.

    Parameters
    ----------
    iniPar : np.ndarray
        Initial conditions - either an array of one initial value per g.nx, or an array
        of parameters (e.g. [fluence, alpha, direction]) usable to generate the initial condition.
    g : Grid
        Object containing space and time grid information.
    p : Parameters
        Object corresponding to current state of MMC walk.
    meas : str, optional
        Type of measurement (e.g. TRPL, TRTS) being simulated. The default is "TRPL".
    solver : tuple(str), optional
        Solution method used to perform simulation and optional related args.
        The first element is the solver type.
        Choices include:
        solveivp - scipy.integrate.solve_ivp()
        odeint - scipy.integrate.odeint()
        NN - a tensorflow/keras model (WIP!)
        All subsequent elements are optional. For NN the second element is
        a path to the NN weight file, the third element is a path
        to the corresponding NN scale factor file.
        The default is ("solveivp",).
    model : str, optional
        Physics model to be solved by the solver, chosen from MODELS.
        The default is "std".
    RTOL, ATOL : float, optional
        Tolerance parameters for scipy solvers. See the solve_ivp() docs for details.

    Returns
    -------
    sol : np.ndarray
        Array of values (e.g. TRPL) from final simulation.
    next_init : np.ndarray
        Values (e.g. the electron profile) at the final time of the simulation.

    """
    if solver[0] == "solveivp" or solver[0] == "odeint" or solver[0] == "diagnostic":
        if len(iniPar) == g.nx:         # If list of initial values
            init_dN = iniPar * 1e-21    # [cm^-3] to [nm^-3]
        else:                           # List of parameters
            fluence = iniPar[0] * 1e-14 # [cm^-2] to [nm^-2]
            alpha = iniPar[1] * 1e-7    # [cm^-1] to [nm^-1]
            init_dN = fluence * alpha * np.exp(-alpha * g.xSteps)
            try:
                init_dN = init_dN[::np.sign(int(iniPar[2]))]
            except (IndexError, ValueError):
                pass

        p.apply_unit_conversions()
        N = init_dN + p.n0
        P = init_dN + p.p0
        E_f = E_field(N, P, p, g.dx)


        # Depends on how many dependent variables and parameters are in the selected model
        if model == "std":
            init_condition = np.concatenate([N, P, E_f], axis=None)
            args = (g.nx, g.dx, p.n0, p.p0, p.mu_n, p.mu_p, p.ks, p.Cn, p.Cp,
                p.Sf, p.Sb, p.tauN, p.tauP, ((q_C) / (p.eps * eps0)), p.Tm,)
        elif model == "traps":
            init_condition = np.concatenate([N, np.zeros_like(N), P, E_f], axis=None)
            args = (g.nx, g.dx, p.n0, p.p0, p.mu_n, p.mu_p, p.ks, p.Cn, p.Cp,
                p.Sf, p.Sb, p.tauN, p.tauP, ((q_C) / (p.eps * eps0)), p.Tm,
                p.kC, p.Nt, p.tauE)
        else:
            raise ValueError(f"Invalid model {model}")

        dy = lambda t, y: MODELS[model](t, y, *args)

        s = Solution()
        # Can't get solve_ivp's root finder to work reliably, so leaving out the early termination events for now.
        # Instead we will let solve_ivp run as-is, and then find where the early termination cutoff would have been
        # after the fact.
        # if meas == "TRPL":
        #     min_y = g.min_y * 1e-23 # To nm/ns units
        #     stop_integrate = lambda t, y: check_threshold(t, y, g.nx, g.dx, min_y=min_y, mode="TRPL",
        #                                                   ks=p.ks, n0=p.n0, p0=p.p0)
        #     stop_integrate.terminal = 1

        # elif meas == "TRTS":
        #     min_y = g.min_y * 1e-9
        #     stop_integrate = lambda t, y: check_threshold(t, y, g.nx, g.dx, min_y=min_y, mode="TRTS",
        #                                                   mu_n=p.mu_n, mu_p=p.mu_p, n0=p.n0, p0=p.p0)
        #     stop_integrate.terminal = 1
        # else:
        #     raise NotImplementedError("TRPL or TRTS only")

        i_final = len(g.tSteps)
        if solver[0] == "solveivp" or solver[0] == "diagnostic":
            sol = solve_ivp(dy, [g.start_time, g.time], init_condition,
                            method='LSODA', dense_output=True, # events=stop_integrate,
                            max_step=g.hmax, rtol=RTOL, atol=ATOL)

            data = sol.sol(g.tSteps).T
            data[g.tSteps > sol.t[-1]] = 0 # Disallow sol from extrapolating beyond time it solved up to
            # if len(sol.t_events[0]) > 0:
            #     t_final = sol.t_events[0][0]
            #     try:
            #         i_final = np.where(g.tSteps < t_final)[0][-1]
            #     except IndexError:
            #         pass

        else:
            data = odeint(MODELS[model], init_condition, g.tSteps, args=args,
                      hmax=g.hmax, rtol=RTOL, atol=ATOL, tfirst=True)

        # Also depends on how many dependent variables
        if model == "std":
            s.N, s.P, E_f = np.split(data, [g.nx, 2*g.nx], axis=1)
        elif model == "traps":
            s.N, s.N_trap, s.P, E_f = np.split(data, [g.nx, 2*g.nx, 3*g.nx], axis=1)

        if meas == "TRPL":
            s.calculate_PL(g, p)
            next_init = s.N[-1] - p.n0
            p.apply_unit_conversions(reverse=True)  # [nm, V, ns] to [cm, V, s]
            s.PL *= 1e23                            # [nm^-2 ns^-1] to [cm^-2 s^-1]
            i_final = np.argmax(s.PL < g.min_y)
            if s.PL[i_final] < g.min_y:
                s.PL[i_final:] = g.min_y
            return s.PL, next_init
        elif meas == "TRTS":
            s.calculate_TRTS(g, p)
            next_init = s.N[-1] - p.n0
            p.apply_unit_conversions(reverse=True)
            s.trts *= 1e9
            i_final = np.argmax(s.trts < g.min_y)
            if s.trts[i_final] < g.min_y:
                s.trts[i_final:] = g.min_y
            return s.trts, next_init
        else:
            raise NotImplementedError("TRTS or TRPL only")

    elif solver[0] == "NN":
        if not HAS_NN_LIB:
            raise ImportError("Failed to load neural network library")

        if meas != "TRPL":
            raise NotImplementedError("TRPL only")

        if not nn.has_model:
            nn.load_model(solver[1], solver[2])

        scaled_matPar = np.zeros((1, 14))
        scaled_matPar[0] = [p.p0, p.mu_n, p.mu_p, p.ks, p.Cn, p.Cp, p.Sf, p.Sb, p.tauN, p.tauP, p.eps**-1,
                            iniPar[0], iniPar[1], g.thickness]

        pl_from_NN = nn.predict(g.tSteps, scaled_matPar)
        return pl_from_NN, None

    else:
        raise NotImplementedError


def check_approved_param(new_p, param_info):
    """ Raise a warning for non-physical or unrealistic proposed trial moves,
        or proposed moves that exceed the prior distribution.
    """
    order = list(param_info['names'])
    do_log = param_info["do_log"]
    active = param_info["active"]
    checks = {}
    prior_dist = param_info["prior_dist"]

    # Ensure proposal stays within bounds of prior distribution
    for param in order:
        if param not in order:
            continue
        if not active[param]:
            continue

        lb = prior_dist[param][0]
        ub = prior_dist[param][1]
        if do_log[param]:
            diff = 10 ** new_p[order.index(param)]
        else:
            diff = new_p[order.index(param)]
        checks[f"{param}_size"] = (lb < diff < ub)

    # TRPL specific checks:
    # p0 > n0 by definition of a p-doped material
    if 'p0' in order and 'n0' in order:
        checks["p0_greater"] = (new_p[order.index('p0')]
                                > new_p[order.index('n0')])
    else:
        checks["p0_greater"] = True

    # tau_n and tau_p must be *close* (within 2 OM) for a reasonable midgap SRH
    if 'tauN' in order and 'tauP' in order:
        # Compel logscale for this one - makes for easier check
        logtn = new_p[order.index('tauN')]
        if not do_log["tauN"]:
            logtn = np.log10(logtn)

        logtp = new_p[order.index('tauP')]
        if not do_log["tauP"]:
            logtp = np.log10(logtp)

        diff = np.abs(logtn - logtp)
        checks["tn_tp_close"] = (diff <= 2)

    else:
        checks["tn_tp_close"] = True

    failed_checks = [k for k in checks if not checks[k]]

    return failed_checks


def select_next_params(p, means, variances, param_info, trial_function="box",
                       coerce_hard_bounds=False, logger=None, verbose=False):
    """ Trial move function:
        box: uniform rectangle centered about current state.
        gauss: gaussian centered about current state.
    """
    is_active = param_info["active"]
    do_log = param_info["do_log"]
    names = param_info["names"]

    mu_constraint = param_info.get("do_mu_constraint", None)

    mean = means.to_array(param_info)
    for i, param in enumerate(names):
        if do_log[param]:
            mean[i] = np.log10(mean[i])

    cov = variances.cov

    tries = 0

    # Try up to MAX_PROPOSALS times to come up with a proposal that stays within
    # the hard boundaries, if we ask
    if coerce_hard_bounds:
        max_tries = MAX_PROPOSALS
    else:
        max_tries = 1

    new_p = np.zeros_like(mean)
    while tries < max_tries:
        tries += 1

        if trial_function == "box":
            for i, param in enumerate(names):
                new_p[i] = np.random.uniform(
                    mean[i]-cov[i, i], mean[i]+cov[i, i])
                if mu_constraint is not None and param == "mu_p":
                    ambi = mu_constraint[0]
                    ambi_std = mu_constraint[1]
                    if verbose and logger is not None:
                        logger.debug(f"mu constraint: ambi {ambi} +/- {ambi_std}")
                    new_muambi = np.random.uniform(ambi - ambi_std, ambi + ambi_std)
                    new_p[i] = np.log10(
                        (2 / new_muambi - 1 / 10 ** new_p[i-1])**-1)

        elif trial_function == "gauss":
            try:
                if not np.all(cov >= 0):
                    raise RuntimeError
                new_p = np.random.multivariate_normal(mean, cov)
            except RuntimeError:
                if logger is not None:
                    logger.error(
                        f"multivar_norm failed: mean {mean}, cov {cov}")
                new_p = mean

        else:
            raise ValueError("Invalid trial function - must be \"box\" or \"gauss\"")

        failed_checks = check_approved_param(new_p, param_info)
        success = len(failed_checks) == 0
        if success:
            if verbose and logger is not None:
                logger.info(f"Found params in {tries} tries")
            break

        if logger is not None and len(failed_checks) > 0:
            logger.warning(f"Failed checks: {failed_checks}")

    for i, param in enumerate(names):
        if is_active[param]:
            if do_log[param]:
                setattr(p, param, 10 ** new_p[i])
            else:
                setattr(p, param, new_p[i])
    return


def do_simulation(p, thickness, nx, iniPar, times, hmax, meas="TRPL",
                  solver=("solveivp",), model="std", rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
    """ Set up one simulation. """
    g = Grid()
    g.thickness = thickness
    g.nx = nx
    g.dx = g.thickness / g.nx
    g.xSteps = np.linspace(g.dx / 2, g.thickness - g.dx/2, g.nx)

    g.time = times[-1]
    g.start_time = 0
    g.nt = len(times) - 1
    g.hmax = hmax
    g.tSteps = times

    if times[0] > 0:
        # Always start sims from t=0 even if experimental data doesn't, to verify initial conditions
        dt_estimate = times[1] - times[0]
        g.tSteps = np.concatenate((np.arange(0, times[0], dt_estimate), g.tSteps))

    sol, next_init_condition = solve(
        iniPar, g, p, meas=meas, solver=solver, model=model, RTOL=rtol, ATOL=atol)
    return g.tSteps, sol


def converge_simulation(i, p, sim_info, iniPar, times, vals,
                        hmax, MCMC_fields, logger=None, verbose=True):
    """
    Retest and repeat simulation until all stipulated convergence criteria
    are met.
    For some models, this can be as simple as running do_simulation()
    or equivalent once.

    Parameters
    ----------
    i : int
        Index of ith simulation in a measurement set requiring n simulations.
    p : Parameters
        Object corresponding to current state of MMC walk.
    sim_info : dict
        Dictionary compatible with unpack_simpar(),
        containing a thickness, nx, and measurement type info
        needed for simulation.
    iniPar : ndarray
        Array of initial conditions for simulation.
    times : ndarray
        Time points at which to evaluate the simulation.
    vals : ndarray
        Actual values to compare simulation output against.
    hmax : list
        List of length n, indexable by i, containing adaptive time steps
        to be used for each simulation.
    MCMC_fields : dict
        Dictionary of MMC control parameters.
    logger : logger
        Logger to write status messages to. The default is None.
    verbose : bool, optional
        Print more detailed status messages. The default is False.

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
    thickness, nx, meas_type = unpack_simpar(sim_info, i)

    rtol = MCMC_fields.get("rtol", DEFAULT_RTOL)
    atol = MCMC_fields.get("atol", DEFAULT_ATOL)

    t_steps = np.array(times)
    sol = np.zeros_like(t_steps)

    try:
        t_steps, sol = do_simulation(p, thickness, nx, iniPar, times, hmax[i],
                                    meas=meas_type,
                                    solver=MCMC_fields["solver"], model=MCMC_fields["model"],
                                    rtol=rtol, atol=atol)
    except ValueError as e:
        success = False
        if logger is not None:
            logger.warning(f"{i}: Simulation error occurred: {e}")
        return t_steps, sol, success
    
    # Other tests for validity may be inserted here

    if MCMC_fields["solver"][0] == "diagnostic":
        # Replace this with curve_fitting code as needed
        pass

    if verbose and logger is not None:
        logger.info(f"{i}: Simulation complete hmax={hmax}; t {t_steps[0]}-{t_steps[-1]}; x {thickness}")

    return t_steps, sol, success


def roll_acceptance(logratio):
    accepted = False
    if logratio >= 0: # Automatic accept
        accepted = True

    else:
        accept = np.random.random()
        if accept < 10 ** logratio:
            accepted = True
    return accepted


def unpack_simpar(sim_info, i):
    thickness = sim_info["lengths"][i]
    nx = sim_info["nx"][i]
    meas_type = sim_info["meas_types"][i]
    return thickness, nx, meas_type


def search_c_grps(c_grps : list[tuple], i : int) -> int:
    """
    Find the constraint group that contains i
    and return its first value
    """
    for c_grp in c_grps:
        for c in c_grp:
            if i == c:
                return c_grp[0]
    return i


def set_min_y(sol, vals, scale_shift):
    """
    Raise the values in (sol + scale_shift) to at least the minimum of vals.
    scale_shift and vals should be in log scale; sol in regular scale
    Returns:
    sol : np.ndarray
        New sol with raised values.
    min_y : float
        min_val sol was raised to. Regular scale.
    n_set : int
        Number of values in sol raised.
    """
    min_y = 10 ** min(vals - scale_shift)
    i_final = np.searchsorted(-sol, -min_y)
    sol[i_final:] = min_y
    return sol, min_y, len(sol[i_final:])


def almost_equal(x, x0, threshold=1e-10):
    if x.shape != x0.shape:
        return False

    return np.abs(np.nanmax((x - x0) / x0)) < threshold


def one_sim_likelihood(p, sim_info, IRF_tables, hmax, MCMC_fields, logger, verbose, args):
    i, iniPar, times, vals, uncs = args
    meas_type = sim_info["meas_types"][i]
    irf_convolution = MCMC_fields.get("irf_convolution", None)

    ff = MCMC_fields.get("fittable_fluences", None)
    if (ff is not None and i in ff[1]):
        if ff[2] is not None and len(ff[2]) > 0:
            iniPar[0] *= getattr(p, f"_f{search_c_grps(ff[2], i)}")
        else:
            iniPar[0] *= getattr(p, f"_f{i}")
    fa = MCMC_fields.get("fittable_absps", None)
    if (fa is not None and i in fa[1]):
        if fa[2] is not None and len(fa[2]) > 0:
            iniPar[1] *= getattr(p, f"_a{search_c_grps(fa[2], i)}")
        else:
            iniPar[1] *= getattr(p, f"_a{i}")
    fs = MCMC_fields.get("scale_factor", None)
    if (fs is not None and i in fs[1]):
        if fs[2] is not None and len(fs[2]) > 0:
            scale_shift = np.log10(getattr(p, f"_s{search_c_grps(fs[2], i)}"))
        else:
            scale_shift = np.log10(getattr(p, f"_s{i}"))
    else:
        scale_shift = 0

    if meas_type == "pa":
        tSteps = np.array([0])
        sol = np.array([U(p.x)])
        success = True
    else:
        tSteps, sol, success = converge_simulation(i, p, sim_info, iniPar, times, vals,
                                                   hmax, MCMC_fields, logger, verbose)
    if not success:
        likelihood = -np.inf
        err_sq = np.inf
        return likelihood, err_sq

    try:
        if irf_convolution is not None and irf_convolution[i] != 0:
            if verbose and logger is not None:
                logger.info(f"Convolving with wavelength {irf_convolution[i]}")
            wave = int(irf_convolution[i])
            tSteps, sol, success = do_irf_convolution(
                tSteps, sol, IRF_tables[wave], time_max_shift=True)
            if not success:
                raise ValueError("Conv failed. Check measurement data times for floating-point inaccuracies.\n"
                                 "This may also happen if simulated signal decays extremely slowly.")
            sol, times_c, vals_c, uncs_c = post_conv_trim(tSteps, sol, times, vals, uncs)

        else:
            # Still need to trim, in case experimental data doesn't start at t=0
            times_c = times
            vals_c = vals
            uncs_c = uncs
            sol = sol[-len(times_c):]

    except ValueError as e:
        logger.warning(e)
        likelihood = -np.inf
        err_sq = np.inf
        return likelihood, err_sq

    if verbose and logger is not None:
        logger.info(f"Comparing times {times_c[0]}-{times_c[-1]}")

    if (MCMC_fields["self_normalize"] is not None and
        sim_info["meas_types"][i] in MCMC_fields["self_normalize"]):
        if verbose and logger is not None:
            logger.info("Normalizing sim result...")
        sol /= np.nanmax(sol)

        # Suppress scale_factor for all measurements being normalized
        p.suppress_scale_factor(MCMC_fields.get("scale_factor", None), i)
        scale_shift = 0

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
            raise ValueError(f"{i}: Simulation failed: too many negative vals")

        if n_fails > 0:
            logger.warning(f"{i}: {n_fails} / {len(sol)} non-positive vals")

        sol[where_failed] *= -1
    except ValueError as e:
        logger.warning(e)
        likelihood = -np.inf
        err_sq = np.inf
        return likelihood, err_sq

    if MCMC_fields.get("force_min_y", False):
        sol, min_y, n_set = set_min_y(sol, vals_c, scale_shift)
        if verbose and logger is not None:
            logger.warning(f"min_y: {min_y}")
            if n_set > 0:
                logger.warning(f"{n_set} values raised to min_y")

    if meas_type == "pa":
        likelihood = -sol[0] / MCMC_fields["current_sigma"][meas_type]
        err_sq = -sol[0]
    else:
        try:
            err_sq = (np.log10(sol) + scale_shift - vals_c) ** 2

            # Compatibility with single sigma
            if isinstance(MCMC_fields["current_sigma"], dict):
                likelihood = - \
                    np.sum(err_sq / (MCMC_fields["current_sigma"][meas_type]**2 + 2*uncs_c**2))
            else:
                likelihood = - \
                    np.sum(err_sq / (MCMC_fields["current_sigma"]**2 + 2*uncs_c**2))

            if np.isnan(likelihood):
                raise ValueError(f"{i}: Simulation failed: invalid likelihood")
        except ValueError as e:
            logger.warning(e)
            likelihood = -np.inf
            err_sq = np.inf
    return likelihood, err_sq


def run_iteration(p, sim_info, iniPar, times, vals, uncs, IRF_tables, hmax,
                  MCMC_fields, verbose, logger, prev_p=None, t=0):
    # Calculates likelihood of a new proposed parameter set
    accepted = True
    logratio = 0  # acceptance ratio = 1
    p.likelihood = np.zeros(sim_info["num_meas"])

    # Can't use ndarray - err_sq for each sim can be different length
    p.err_sq = [[] * sim_info["num_meas"]]

    if MCMC_fields.get("use_multi_cpus", False):
        raise NotImplementedError("WIP - multi_cpus")
        # with Pool(MCMC_fields["num_cpus"]) as pool:
        #    likelihoods = pool.map(partial(one_sim_likelihood, p, sim_info, hmax, MCMC_fields, logger), zip(np.arange(len(iniPar)), iniPar, times, vals))
        #    p.likelihood = np.array(likelihoods)

    for i in range(sim_info["num_meas"]):
        p.likelihood[i], p.err_sq[i] = one_sim_likelihood(
            p, sim_info, IRF_tables, hmax, MCMC_fields, logger, verbose,
            (i, np.array(iniPar[i]), times[i], vals[i], uncs[i]))

    if prev_p is not None:
        if verbose and logger is not None:
            logger.info(f"Likelihood of proposed move: {np.sum(p.likelihood)}")
        logratio = (np.sum(p.likelihood) - np.sum(prev_p.likelihood))
        if np.isnan(logratio):
            logratio = -np.inf

        if verbose and logger is not None:
            logger.info(f"Partial Ratio: {10 ** logratio}")

        accepted = roll_acceptance(logratio)

    if prev_p is not None and accepted:
        prev_p.likelihood = np.array(p.likelihood)
        prev_p.err_sq = list(p.err_sq)
    return accepted


def main_metro_loop(MS, starting_iter, num_iters,
                    need_initial_state=True, logger=None, verbose=False):
    """
    Run the Metropolis loop for a specified number of iterations,
    storing all info in a MetroState() object and saving the MetroState()
    as occasional checkpoints.

    Parameters
    ----------
    MS : MetroState() object

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
    None. MetroState() object is updated throughout.

    """
    DA_mode = MS.MCMC_fields.get("delayed_acceptance", "off")
    checkpoint_freq = MS.MCMC_fields["checkpoint_freq"]

    if need_initial_state:
        logger.info("Simulating initial state:")
        # Calculate likelihood of initial guess
        STARTING_HMAX = MS.MCMC_fields.get("hmax", DEFAULT_HMAX)
        MS.running_hmax = [STARTING_HMAX] * len(MS.iniPar)
        run_iteration(MS.prev_p, MS.sim_info, MS.iniPar,
                      MS.times, MS.vals, MS.uncs, MS.IRF_tables,
                      MS.running_hmax, MS.MCMC_fields, verbose, logger)
        MS.H.update(0, MS.prev_p, MS.means, MS.param_info)
    for k in range(starting_iter, num_iters):
        try:
            logger.info("#####")
            logger.info(f"Iter {k}")
            logger.info("#####")

            # Check if anneal needed
            MS.anneal(k, MS.uncs)
            if (verbose or k % MSG_FREQ == 0 or k < starting_iter + MSG_COOLDOWN) and logger is not None:
                logger.debug(f"Current model sigma: {MS.MCMC_fields['current_sigma']}")
                logger.debug(f"Current variances: {MS.variances.trace()}")

            # Identify which parameter to move
            if MS.MCMC_fields.get("one_param_at_a_time", 0):
                picked_param = MS.means.actives[k % len(MS.means.actives)]
            else:
                picked_param = None

            # Select next sample from distribution

            if MS.MCMC_fields.get("adaptive_covariance", "None") == "None":
                MS.variances.mask_covariance(picked_param)

            select_next_params(MS.p, MS.means, MS.variances, MS.param_info,
                               MS.MCMC_fields["proposal_function"],
                               MS.MCMC_fields.get("hard_bounds", 0), logger)

            if (verbose or k % MSG_FREQ == 0 or k < starting_iter + MSG_COOLDOWN) and logger is not None:
                MS.print_status(logger)

            MS.H.record_best_logll(k, MS.prev_p)

            if DA_mode == "off":
                accepted = run_iteration(MS.p, MS.sim_info, MS.iniPar,
                                         MS.times, MS.vals, MS.uncs, MS.IRF_tables,
                                         MS.running_hmax, MS.MCMC_fields, verbose,
                                         logger, prev_p=MS.prev_p, t=k)

            elif DA_mode == 'DEBUG':
                accepted = False

            else:
                raise ValueError("Invalid DA mode - must be \"off\" or \"DEBUG\"")

            if verbose and not accepted and logger is not None:
                logger.info("Rejected!")

            if accepted:
                MS.means.transfer_from(MS.p, MS.param_info)
                MS.H.accept[0, k] = 1

            MS.H.update(k, MS.p, MS.means, MS.param_info)
            MS.latest_iter = k

        except KeyboardInterrupt:
            logger.warning(f"Terminating with k={k-1} iters completed:")
            MS.H.truncate(k, MS.param_info)
            break

        if checkpoint_freq is not None and k % checkpoint_freq == 0:
            chpt_header = MS.MCMC_fields["checkpoint_header"]
            chpt_fname = os.path.join(MS.MCMC_fields["checkpoint_dirname"],
                                    f"{chpt_header}.pik")
            logger.info(f"Saving checkpoint at k={k}; fname {chpt_fname}")
            MS.random_state = np.random.get_state()
            MS.checkpoint(chpt_fname)
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
        MS = MetroState(param_info, MCMC_fields, num_iters)
        MS.checkpoint(os.path.join(MS.MCMC_fields["output_path"], export_path))
        if "checkpoint_header" not in MS.MCMC_fields:
            MS.MCMC_fields["checkpoint_header"] = export_path[:export_path.find(".pik")]

        starting_iter = 1

        # Just so MS saves a record of these
        MS.sim_info = sim_info
        MS.iniPar = iniPar
        MS.times, MS.vals, MS.uncs = e_data

        if logger is not None:
            for i, unc in enumerate(MS.uncs):
                logger.info(f"{i} exp unc max: {np.amax(unc)} avg: {np.mean(unc)}")

        if MCMC_fields.get("irf_convolution", None) is not None:
            irfs = {}
            for i in MCMC_fields["irf_convolution"]:
                if i > 0 and i not in irfs:
                    irfs[int(i)] = np.loadtxt(os.path.join("IRFs", f"irf_{int(i)}nm.csv"),
                                              delimiter=",")

            MS.IRF_tables = make_I_tables(irfs)
        else:
            MS.IRF_tables = {}

    else:
        with open(os.path.join(MCMC_fields["checkpoint_dirname"],
                               load_checkpoint), 'rb') as ifstream:
            MS = pickle.load(ifstream)
            np.random.set_state(MS.random_state)

            if "starting_iter" in MCMC_fields and MCMC_fields["starting_iter"] < MS.latest_iter:
                starting_iter = MCMC_fields["starting_iter"]
                MS.H.extend(starting_iter, MS.param_info)
                for param in MS.param_info["names"]:
                    setattr(MS.means, param, getattr(MS.H, f"mean_{param}")[0, starting_iter - 1])
                MS.prev_p.likelihood = np.zeros_like(MS.prev_p.likelihood)
                MS.prev_p.likelihood = MS.H.loglikelihood[0, -1]
            else:
                starting_iter = MS.latest_iter + 1

            MS.H.extend(num_iters, param_info)
            MS.MCMC_fields["num_iters"] = MCMC_fields["num_iters"]
            # Induce annealing, which also corrects the prev_likelihood and adjust the step size
            # MS.anneal(-1, MS.uncs, force=True)

    # From this point on, for consistency, work with ONLY the MetroState object!
    logger.info(f"Sim info: {MS.sim_info}")
    logger.info(f"Param infos: {MS.param_info}")
    logger.info(f"MCMC fields: {MS.MCMC_fields}")

    need_initial_state = (load_checkpoint is None)
    main_metro_loop(MS, starting_iter, num_iters,
                    need_initial_state=need_initial_state,
                    logger=logger, verbose=verbose)

    MS.random_state = np.random.get_state()
    if export_path is not None:
        logger.info(f"Exporting to {MS.MCMC_fields['output_path']}")
        MS.checkpoint(os.path.join(MS.MCMC_fields["output_path"], export_path))

    if using_default_logger:
        stop_logging(logger, handler, 0)
    return MS
