"""Standalone utility functions"""
from numba import njit
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


@njit(cache=True)
def U(x):
    if   x < -2:
        u = np.inf
    elif x < -1.25:
        u =  1 * (1 + np.sin(2*np.pi*x))
    elif x < -0.25:
        u = 2 * (1 + np.sin(2*np.pi*x))
    elif x <  0.75:
        u = 3 * (1 + np.sin(2*np.pi*x))
    elif x <  1.75:
        u = 4 * (1 + np.sin(2*np.pi*x))
    elif x <=  2:
        u = 5 * (1 + np.sin(2*np.pi*x))
    else:
        u = np.inf
    return u