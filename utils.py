"""Standalone utility functions"""
import numpy as np

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


def unpack_simpar(sim_info, i):
    thickness = sim_info["lengths"][i]
    nx = sim_info["nx"][i]
    meas_type = sim_info["meas_types"][i]
    return thickness, nx, meas_type


def U(x):
    """Potential for single particle toy problem"""
    return 1000 * (x < -2) + 1 * (1 + np.sin(2*np.pi*x)) * np.logical_and(-2 <= x, x <= -1.25) \
                             + 2 * (1 + np.sin(2*np.pi*x)) * np.logical_and(-1.25 <= x, x <= -0.25) \
                             + 3 * (1 + np.sin(2*np.pi*x)) * np.logical_and(-0.25 <= x, x <= 0.75) \
                             + 4 * (1 + np.sin(2*np.pi*x)) * np.logical_and(0.75 <= x, x <= 1.75) \
                             + 5 * (1 + np.sin(2*np.pi*x)) * np.logical_and(1.75 <= x, x <= 2) \
                             + 1000 * (x > 2)