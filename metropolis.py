# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 22:13:26 2022

@author: cfai2
"""
import numpy as np
from scipy.integrate import solve_ivp, odeint
import os
import sys
import signal
import pickle

from forward_solver import dydt_numba
from sim_utils import MetroState, Grid, Solution
from mcmc_logging import start_logging, stop_logging
from bayes_io import make_dir, clear_checkpoint_dir


# Constants
eps0 = 8.854 * 1e-12 * 1e-9  # [C / V m] to {C / V nm}
q = 1.0  # [e]
q_C = 1.602e-19  # [C]
kB = 8.61773e-5  # [eV / K]
MIN_HMAX = 1e-2  # [ns]
DEFAULT_RTOL = 1e-7
DEFAULT_ATOL = 1e-10
DEFAULT_HMAX = 4
MAX_PROPOSALS = 100


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


def model(init_dN, g, p, meas="TRPL", solver="solveivp",
          RTOL=DEFAULT_RTOL, ATOL=DEFAULT_ATOL):
    """ Calculate one simulation. """
    N = init_dN + p.n0
    P = init_dN + p.p0
    E_f = E_field(N, P, p, g.dx)

    init_condition = np.concatenate([N, P, E_f], axis=None)

    if solver == "solveivp":
        args = (g.nx, g.dx, p.n0, p.p0, p.mu_n, p.mu_p, p.ks, p.Cn, p.Cp,
                p.Sf, p.Sb, p.tauN, p.tauP, ((q_C) / (p.eps * eps0)), p.Tm)
        sol = solve_ivp(dydt_numba, [g.start_time, g.time], init_condition,
                        args=args, t_eval=g.tSteps, method='LSODA',
                        max_step=g.hmax, rtol=RTOL, atol=ATOL)
        data = sol.y.T
    elif solver == "odeint":
        args = (g.nx, g.dx, p.n0, p.p0, p.mu_n, p.mu_p, p.ks, p.Cn, p.Cp,
                p.Sf, p.Sb, p.tauN, p.tauP, ((q_C) / (p.eps * eps0)), p.Tm)
        data = odeint(dydt_numba, init_condition, g.tSteps, args=args,
                      hmax=g.hmax, rtol=RTOL, atol=ATOL, tfirst=True)
    else:
        raise NotImplementedError

    s = Solution()
    s.N, s.P, E_f = np.split(data, [g.nx, 2*g.nx], axis=1)
    if meas == "TRPL":
        s.calculate_PL(g, p)
        return s.PL, (s.N[-1]-p.n0)
    elif meas == "TRTS":
        s.calculate_TRTS(g, p)
        return s.trts, (s.N[-1] - p.n0)
    else:
        raise NotImplementedError("TRTS or TRPL only")


def check_approved_param(new_p, param_info):
    """ Raise a warning for non-physical or unrealistic proposed trial moves,
        or proposed moves that exceed the prior distribution.
    """
    order = list(param_info['names'])
    ucs = param_info.get('unit_conversions', {})
    do_log = param_info["do_log"]
    checks = {}
    prior_dist = param_info["prior_dist"]

    # Ensure proposal stays within bounds of prior distribution
    for param in order:
        if param not in order:
            continue

        lb = prior_dist[param][0]
        ub = prior_dist[param][1]
        if do_log[param]:
            diff = 10 ** new_p[order.index(param)]
        else:
            diff = new_p[order.index(param)]
        diff /= ucs.get(param, 1)
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
        logtn -= np.log10(ucs.get('tauN', 1))

        logtp = new_p[order.index('tauP')]
        if not do_log["tauP"]:
            logtp = np.log10(logtp)
        logtp -= np.log10(ucs.get('tauP', 1))

        diff = np.abs(logtn - logtp)
        checks["tn_tp_close"] = (diff <= 2)

    else:
        checks["tn_tp_close"] = True

    failed_checks = [k for k in checks if not checks[k]]

    return failed_checks


def select_next_params(p, means, variances, param_info, trial_function="box",
                       coerce_hard_bounds=False, logger=None):
    """ Trial move function:
        box: uniform rectangle centered about current state.
        gauss: gaussian centered about current state.
    """
    is_active = param_info["active"]
    do_log = param_info["do_log"]
    names = param_info["names"]

    secret_mu = param_info.get("secret_mu", False)

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

    while tries < max_tries:
        tries += 1

        if trial_function == "box":
            new_p = np.zeros_like(mean)
            for i, param in enumerate(names):
                new_p[i] = np.random.uniform(
                    mean[i]-cov[i, i], mean[i]+cov[i, i])
                if secret_mu and param == "mu_p":
                    new_muambi = np.random.normal(20, 0.3) * \
                        param_info["unit_conversions"]["mu_n"]
                    new_p[i] = np.log10(
                        (2 / new_muambi - 1 / 10 ** new_p[i-1])**-1)

        elif trial_function == "gauss":
            try:
                assert np.all(cov >= 0)
                new_p = np.random.multivariate_normal(mean, cov)
            except Exception:
                if logger is not None:
                    logger.error(
                        f"multivar_norm failed: mean {mean}, cov {cov}")
                new_p = mean

        failed_checks = check_approved_param(new_p, param_info)
        success = len(failed_checks) == 0
        if success:
            logger.info("Found params in {} tries".format(tries))
            break

        if logger is not None and len(failed_checks) > 0:
            logger.warning("Failed checks: {}".format(failed_checks))

    for i, param in enumerate(names):
        if is_active[param]:
            if do_log[param]:
                setattr(p, param, 10 ** new_p[i])
            else:
                setattr(p, param, new_p[i])
    return


def do_simulation(p, thickness, nx, iniPar, times, hmax, meas="TRPL",
                  solver="solveivp", rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
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

    sol, next_init_condition = model(
        iniPar, g, p, meas=meas, solver=solver, RTOL=rtol, ATOL=atol)
    return sol


def roll_acceptance(logratio):
    accepted = False
    if logratio >= 0:
        # Continue
        accepted = True

    else:
        accept = np.random.random()
        if accept < 10 ** logratio:
            # Continue
            accepted = True
    return accepted


def unpack_simpar(sim_info, i):
    thickness = sim_info["lengths"][i]
    nx = sim_info["nx"]
    meas_type = sim_info["meas_types"][i]
    return thickness, nx, meas_type


def detect_sim_fail(sol, ref_vals):
    fail = len(sol) < len(ref_vals)
    if fail:
        sol2 = np.ones_like(ref_vals) * sys.float_info.min
        sol2[:len(sol)] = sol
        sol = np.array(sol2)

    return sol, fail


def detect_sim_depleted(sol):
    fail = np.any(sol < 0)
    if fail:
        sol = np.abs(sol) + sys.float_info.min
    return sol, fail


def almost_equal(x, x0, threshold=1e-10):
    if x.shape != x0.shape:
        return False

    return np.abs(np.nanmax((x - x0) / x0)) < threshold


def one_sim_likelihood(p, sim_info, hmax, MCMC_fields, logger, args):
    i, iniPar, times, vals, uncs = args
    STARTING_HMAX = MCMC_fields.get("hmax", DEFAULT_HMAX)
    RTOL = MCMC_fields.get("rtol", DEFAULT_RTOL)
    ATOL = MCMC_fields.get("atol", DEFAULT_ATOL)
    thickness, nx, meas_type = unpack_simpar(sim_info, i)
    # Always attempt a slightly larger hmax than what worked at previous proposal
    hmax[i] = min(STARTING_HMAX, hmax[i] * 2)

    while hmax[i] > MIN_HMAX:
        sol = do_simulation(p, thickness, nx, iniPar, times, hmax[i], meas=meas_type,
                            solver=MCMC_fields["solver"], rtol=RTOL, atol=ATOL)

        # if verbose:
        logger.info("{}: Simulation complete hmax={}; t {}-{}".format(i,
                    hmax, times[0], times[len(sol)-1]))

        sol, fail = detect_sim_fail(sol, vals)
        if fail:
            logger.warning(f"{i}: Simulation terminated early!")

        sol, fail = detect_sim_depleted(sol)
        if fail:
            logger.warning(f"{i}: Carriers depleted!")
            hmax[i] = max(MIN_HMAX, hmax[i] / 2)
            logger.warning(f"{i}: Retrying hmax={hmax}")

        elif MCMC_fields.get("verify_hmax", False):
            hmax[i] = max(MIN_HMAX, hmax[i] / 2)
            logger.info(f"{i}: Verifying convergence with hmax={hmax}...")
            sol2 = do_simulation(p, thickness, nx, iniPar, times, hmax[i],
                                 meas=meas_type, solver=MCMC_fields["solver"],
                                 rtol=RTOL, atol=ATOL)
            if almost_equal(sol, sol2, threshold=RTOL):
                logger.info("Success!")
                break
            else:
                logger.info(f"{i}: Fail - not converged")
                if hmax[i] <= MIN_HMAX:
                    logger.warning(f"{i}: MIN_HMAX reached")

        else:
            break
    try:
        if MCMC_fields.get("self_normalize", False):
            sol /= np.nanmax(sol)
        # TODO: accomodate multiple experiments, just like bayes

        err_sq = (np.log10(sol) + np.log10(p.m) - vals) ** 2
        likelihood = - \
            np.sum(err_sq / (MCMC_fields["current_sigma"]**2 + uncs**2))

        # TRPL must be positive!
        # Any simulation which results in depleted carrier is clearly incorrect
        if fail or np.isnan(likelihood):
            raise ValueError(f"{i}: Simulation failed!")
    except ValueError as e:
        logger.warning(e)
        likelihood = -np.inf
        err_sq = np.inf
    return likelihood, err_sq


def run_iteration(p, sim_info, iniPar, times, vals, uncs, hmax,
                  MCMC_fields, verbose, logger, prev_p=None, t=0):
    # Calculates likelihood of a new proposed parameter set
    accepted = True
    logratio = 0  # acceptance ratio = 1
    p.likelihood = np.zeros(len(iniPar))

    # Can't use ndarray - err_sq for each sim can be different length
    p.err_sq = [[] for i in iniPar]

    if MCMC_fields.get("use_multi_cpus", False):
        raise NotImplementedError
        # with Pool(MCMC_fields["num_cpus"]) as pool:
        #    likelihoods = pool.map(partial(one_sim_likelihood, p, sim_info, hmax, MCMC_fields, logger), zip(np.arange(len(iniPar)), iniPar, times, vals))
        #    p.likelihood = np.array(likelihoods)

    else:
        for i in range(len(iniPar)):
            p.likelihood[i], p.err_sq[i] = one_sim_likelihood(
                p, sim_info, hmax, MCMC_fields, logger,
                (i, iniPar[i], times[i], vals[i], uncs[i]))

    if prev_p is not None:
        logratio = (np.sum(p.likelihood) - np.sum(prev_p.likelihood))
        if np.isnan(logratio):
            logratio = -np.inf

        if verbose and logger is not None:
            logger.info("Partial Ratio: {}".format(10 ** logratio))

        accepted = roll_acceptance(logratio)

    if prev_p is not None and accepted:
        prev_p.likelihood = np.array(p.likelihood)
        prev_p.err_sq = list(p.err_sq)
    return accepted


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
          verbose=False, export_path=None, logger=None):

    if logger is None:  # Require a logger
        logger, handler = start_logging(log_dir=MCMC_fields["output_path"],
                                        name="Log-")
        using_default_logger = True
    else:
        using_default_logger = False

    # Setup
    logger.info("PID: {}".format(os.getpid()))
    all_signal_handler(kill_from_cl)

    if verbose:
        logger.info("Sim info: {}".format(sim_info))
        logger.info("Param infos: {}".format(param_info))
        logger.info("MCMC fields: {}".format(MCMC_fields))

    num_iters = MCMC_fields["num_iters"]
    DA_mode = MCMC_fields.get("delayed_acceptance", "off")
    checkpoint_freq = MCMC_fields["checkpoint_freq"]
    load_checkpoint = MCMC_fields["load_checkpoint"]
    STARTING_HMAX = MCMC_fields.get("hmax", DEFAULT_HMAX)

    times, vals, uncs = e_data

    if verbose and logger is not None:
        for i in range(len(uncs)):
            logger.debug("{} exp unc max: {} avg: {}".format(i, np.amax(uncs[i]), np.mean(uncs[i])))

    make_dir(MCMC_fields["checkpoint_dirname"])
    clear_checkpoint_dir(MCMC_fields)

    make_dir(MCMC_fields["output_path"])
    if load_checkpoint is not None:
        with open(os.path.join(MCMC_fields["checkpoint_dirname"],
                               load_checkpoint), 'rb') as ifstream:
            MS = pickle.load(ifstream)
            np.random.set_state(MS.random_state)
            first_under = load_checkpoint.find("_")
            tail = load_checkpoint.rfind(".pik")

            starting_iter = int(load_checkpoint[first_under+1:tail])+1
            MS.H.extend(num_iters, param_info)
            MS.MCMC_fields["num_iters"] = MCMC_fields["num_iters"]

    else:
        MS = MetroState(param_info, MCMC_fields, num_iters)

        starting_iter = 1

        # Calculate likelihood of initial guess
        MS.running_hmax = [STARTING_HMAX] * len(iniPar)
        run_iteration(MS.prev_p, sim_info, iniPar, times, vals, uncs,
                      MS.running_hmax, MS.MCMC_fields, verbose, logger)
        MS.H.update(0, MS.prev_p, MS.means, MS.param_info)

    for k in range(starting_iter, num_iters):
        try:
            logger.info("#####")
            logger.info("Iter {}".format(k))
            logger.info("#####")

            # Check if anneal needed
            MS.anneal(k, uncs)
            logger.debug("Current sigma: {}".format(
                MS.MCMC_fields["current_sigma"]))
            logger.debug("Current variances: {}".format(MS.variances.trace()))

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

            if verbose:
                MS.print_status(logger)

            if DA_mode == "off":
                accepted = run_iteration(MS.p, sim_info, iniPar, times, vals, uncs,
                                         MS.running_hmax, MS.MCMC_fields, verbose,
                                         logger, prev_p=MS.prev_p, t=k)

            elif DA_mode == 'DEBUG':
                accepted = False

            if verbose and not accepted:
                logger.info("Rejected!")

            if accepted:
                MS.means.transfer_from(MS.p, MS.param_info)
                MS.H.accept[k] = 1

            MS.H.update(k, MS.p, MS.means, MS.param_info)
        except KeyboardInterrupt:
            logger.info("Terminating with k={} iters completed:".format(k-1))
            MS.H.truncate(k, MS.param_info)
            break

        if checkpoint_freq is not None and k % checkpoint_freq == 0:
            chpt_header = MS.MCMC_fields["checkpoint_header"]
            chpt_fname = os.path.join(MS.MCMC_fields["checkpoint_dirname"],
                                      f"checkpoint{chpt_header}_{k}.pik")
            logger.info(f"Saving checkpoint at k={k}; fname {chpt_fname}")
            MS.random_state = np.random.get_state()
            MS.checkpoint(chpt_fname)

    MS.H.apply_unit_conversions(MS.param_info)
    MS.H.final_cov = MS.variances.cov

    if export_path is not None:
        logger.info("Exporting to {}".format(MS.MCMC_fields["output_path"]))
        MS.checkpoint(os.path.join(MS.MCMC_fields["output_path"], export_path))

    if using_default_logger:
        stop_logging(logger, handler, 0)
    return MS
