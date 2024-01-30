"""Routines for calculating log likelihoods of trial moves"""
import numpy as np

from forward_solver import solve
from utils import search_c_grps, set_min_y, unpack_simpar, U
from laplace import do_irf_convolution, post_conv_trim
from sim_utils import Grid, NEGATIVE_FRAC_TOL

def eval_trial_move(state, unique_fields, shared_fields, logger):
    """
    Calculates log likelihood of a new proposed state
    Returns:
    logll : float
        Log likelihood value
    ll_funcs : list
        List of lambda funcs that can be used to recalculate logll
        for each measurement with different beta temperatures
    """

    logll = np.zeros(shared_fields["_sim_info"]["num_meas"])
    ll_funcs = [None for _ in range(shared_fields["_sim_info"]["num_meas"])]

    for i in range(shared_fields["_sim_info"]["num_meas"]):
        logll[i], ll_funcs[i] = one_sim_likelihood(i, state, unique_fields, shared_fields, logger)

    logll = np.sum(logll)

    return logll, ll_funcs

def one_sim_likelihood(meas_index, state, unique_fields, shared_fields, logger):
    """
    Calculates log likelihood of one measurement within a proposed state
    """
    iniPar = shared_fields["_init_params"][meas_index]
    meas_type = shared_fields["_sim_info"]["meas_types"][meas_index]
    irf_convolution = shared_fields.get("irf_convolution", None)
    ll_func = lambda b: -np.inf
    ff = shared_fields.get("fittable_fluences", None)
    if ff is not None and meas_index in ff[1]:
        if ff[2] is not None and len(ff[2]) > 0:
            name = f"_f{search_c_grps(ff[2], meas_index)}"
        else:
            name = f"_f{meas_index}"
        iniPar[0] *= state[shared_fields["_param_indexes"][name]]
    fa = shared_fields.get("fittable_absps", None)
    if fa is not None and meas_index in fa[1]:
        if fa[2] is not None and len(fa[2]) > 0:
            name = f"_a{search_c_grps(fa[2], meas_index)}"
        else:
            name = f"_a{meas_index}"
        iniPar[1] *= state[shared_fields["_param_indexes"][name]]
    fs = shared_fields.get("scale_factor", None)
    if fs is not None and meas_index in fs[1]:
        if fs[2] is not None and len(fs[2]) > 0:
            name = f"_s{search_c_grps(fs[2], meas_index)}"
        else:
            name = f"_s{meas_index}"
        scale_shift = np.log10(state[shared_fields["_param_indexes"][name]])
    else:
        scale_shift = 0

    if meas_type == "pa":
        tSteps = np.array([0])
        sol = np.array([U(state[0])])
        success = True
    else:
        tSteps, sol, success = converge_simulation(
            meas_index, state, iniPar, shared_fields, logger
        )
    if not success:
        likelihood = -np.inf
        return likelihood, ll_func

    try:
        if irf_convolution is not None and irf_convolution[meas_index] != 0:
            logger.debug(
                f"Convolving with wavelength {irf_convolution[meas_index]}"
            )
            wave = int(irf_convolution[meas_index])
            tSteps, sol, success = do_irf_convolution(
                tSteps, sol, shared_fields["_IRF_tables"][wave], time_max_shift=True
            )
            if not success:
                raise ValueError(
                    "Conv failed. Check measurement data times for floating-point inaccuracies.\n"
                    "This may also happen if simulated signal decays extremely slowly."
                )
            sol, times_c, vals_c, uncs_c = post_conv_trim(
                tSteps,
                sol,
                shared_fields["_times"][meas_index],
                shared_fields["_vals"][meas_index],
                shared_fields["_uncs"][meas_index],
            )

        else:
            # Still need to trim, in case experimental data doesn't start at t=0
            times_c = shared_fields["_times"][meas_index]
            vals_c = shared_fields["_vals"][meas_index]
            uncs_c = shared_fields["_uncs"][meas_index]
            sol = sol[-len(times_c) :]

    except ValueError as e:
        logger.warning(e)
        likelihood = -np.inf
        return likelihood, ll_func

    logger.debug(f"Comparing times {times_c[0]}-{times_c[-1]}")

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
            logger.warning(
                f"{meas_index}: {n_fails} / {len(sol)} non-positive vals"
            )

        sol[where_failed] *= -1
    except ValueError as e:
        logger.warning(e)
        likelihood = -np.inf
        return likelihood, ll_func

    if shared_fields.get("force_min_y", False):
        sol, min_y, n_set = set_min_y(sol, vals_c, scale_shift)
        logger.debug(f"min_y: {min_y}")
        if n_set > 0:
            logger.debug(f"{n_set} values raised to min_y")

    if meas_type == "pa":
        ll_func = lambda T: -sol[0] * T ** -1
        likelihood = ll_func(unique_fields.get('_T', 1))
    else:
        try:
            err_sq = (np.log10(sol) + scale_shift - vals_c) ** 2

            # Compatibility with single sigma
            ll_func = lambda T: -np.sum(
                err_sq
                / (
                    unique_fields["current_sigma"][meas_type] ** 2 * T
                    + 2 * uncs_c**2
                )
            )

            likelihood = ll_func(unique_fields.get('_T', 1))
            if np.isnan(likelihood):
                raise ValueError(
                    f"{meas_index}: Simulation failed: invalid likelihood"
                )
        except ValueError as e:
            logger.warning(e)
            likelihood = -np.inf
    return likelihood, ll_func

def converge_simulation(meas_index, state, init_conds, shared_fields, logger):
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
    shared_fields : dict
        Monte Carlo settings and data shared by all chains in ensemble

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

    t_steps = np.array(shared_fields["_times"][meas_index])
    sol = np.zeros_like(t_steps)

    try:
        t_steps, sol = do_simulation(meas_index, state, init_conds, shared_fields)
    except ValueError as e:
        success = False
        logger.warning(f"{meas_index}: Simulation error occurred: {e}")
        return t_steps, sol, success

    logger.debug(
        f"{meas_index}: Simulation complete t {t_steps[0]}-{t_steps[-1]}"
    )

    return t_steps, sol, success

def do_simulation(meas_index, state, init_conds, shared_fields):
    """Set up and run one simulation."""
    thickness, nx, meas_type = unpack_simpar(shared_fields["_sim_info"], meas_index)
    g = Grid(thickness, nx, shared_fields["_times"][meas_index], shared_fields["hmax"])

    sol = solve(
        init_conds,
        g,
        state,
        shared_fields["_param_indexes"],
        meas=meas_type,
        units=shared_fields["units"],
        solver=shared_fields["solver"],
        model=shared_fields["model"],
        ini_mode=shared_fields["ini_mode"],
        RTOL=shared_fields["rtol"],
        ATOL=shared_fields["atol"],
    )
    return g.tSteps, sol
