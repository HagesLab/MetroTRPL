# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 12:12:59 2023

@author: cfai2
"""
import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import griddata
from numba import njit


def make_I_tables(irfs):
    """
    Generate the moment integrals for a list of wavelengths requested by Monte Carlo.

    Parameters
    ----------
    irfs : dict {float: 2D array}
        List of raw (t, IRF(t)) values measured for each wavelength
    Returns
    -------
    I_tables : dict {float: tuple(1D array, 1D array)}
        Dictionary of moment integral tables and IRF time points
        for each wavelength.

    """
    I_tables = {}

    for w in irfs:
        w = int(w)
        irf = irfs[w]
        t_irf = irf[:, 0]
        f_t_irf = irf[:, 1]
        nk_irf = len(f_t_irf)
        I_tables[w] = (np.zeros((nk_irf, 3)), t_irf)
        for m in range(nk_irf - 1):
            for n in range(3):
                I_tables[w][0][m, n] = I_moment(t_irf, f_t_irf, m, n, u_spacing=1000)

    return I_tables


def do_irf_convolution(t, y, IRF_table, time_max_shift=False):
    """
    Apply the IRF convolution calculated in IRF_table to a measurement with points (t, y(t))
    A wrapper for convolve(), for y(t) that do not have half time step compared to IRF

    Parameters
    -------
    time_max_shift: bool
        Whether to shift the convolved_t such that max(convolved_y) is at time zero.
        Needed to mimic the max-shifting of some TRPL setups.

    Returns
    -------
    convolved_t : 1D array
        New time points for the convolved data, taken at the time step of the IRF.
    convolved_y : 1D array
        Convolved measurement y(t).
    success : bool
        Whether the convolution was successful
    """
    success = True
    t_irf = IRF_table[1]
    dt_irf = np.mean(np.diff(t_irf))

    resampled_t = np.arange(0, t[-1] + dt_irf / 4, dt_irf / 2)

    # If any resampled t larger than t (e.g. due to floating point err), griddata() will throw nans
    if resampled_t[-1] > t[-1]:
        resampled_t[-1] = t[-1]

    resampled_y = griddata(t, y, resampled_t)

    if any(np.isnan(resampled_y)):
        success = False

    convolved_y = convolve(resampled_y, IRF_table[0])
    convolved_t = resampled_t[::2]
    if time_max_shift:
        convolved_t -= convolved_t[np.argmax(convolved_y)]
        if convolved_t[-1] == 0: # May happen if y decays extremely slowly
            success = False
    return convolved_t, convolved_y, success


def post_conv_trim(conv_t, conv_y, exp_t, exp_y, exp_u):
    """
    Truncate experimental y and unc to time span of convolved y.
    Then, interpolate y back into experimental t.
    Convolution with IRFs that don't start at time zero introduce a time lag to y, which
    causes convolved y to have a shorter time span than the original y.

    Parameters
    ----------
    conv_t : 1D array
        Time points for convolved y, from do_irf_convolution().
    conv_y : 1D array
        Convolved solution from do_irf_convolution().
    exp_t : 1D array
        Time points of exp_y.
    exp_y : 1D array
        Measurement data conv_y will be compared against in likelihood calc.
    exp_u : 1D array
        Uncertainty values of exp_y.

    Returns
    -------
    conv_y : 1D array
        Reinterpolated conv_y.
    times_c : 1D array
        Truncated exp_t
    vals_c : 1D array
        Truncated exp_y.
    uncs_c : 1D array
        Truncated exp_u.

    """
    conv_cutoff = np.where(exp_t < np.nanmax(conv_t))[0][-1]
    conv_y = griddata(conv_t, conv_y, exp_t[:conv_cutoff+1])
    vals_c = exp_y[:conv_cutoff+1]
    uncs_c = exp_u[:conv_cutoff+1]
    times_c = exp_t[:conv_cutoff+1]

    return conv_y, times_c, vals_c, uncs_c


def I_moment(t, y, m, n, u_lower=0, u_upper=1, u_spacing=100):
    """
    Moment integral I_m^n for an instrument response function y(t).

    Parameters
    ----------
    t : 1D array
        Regularly spaced time (x) domain spanned by the IRF.
    y : 1D array
        IRF (y) values at times t.
    m : int
        Lag index. For a measurement spanning k time intervals,
        I_moments should be calculated for lags m=0 to m=k.
        In practice, the IRF is shorter than the measurement
        (nonzero only within intervals nk_irf < k), so lags m > nk_irf can
        be omitted.
    n : int
        Order of moment. Simpson rule used by convolve requires n=[0,1,2].
    u_lower : int, optional
        Lower bound of integral. 0 if nondimensionalized. The default is 0.
    u_upper : int, optional
        Upper bound of integral. 1 if nondimensionalized. The default is 1.
    u_spacing : int, optional
        Number of interpolants taken between successive values of y.
        The default is 100.

    Returns
    -------
    integral : float
        Result of I_m^n.

    """
    dt = t[1] - t[0]
    u = np.linspace(u_lower, u_upper, u_spacing)
    du = u[1] - u[0]
    y_intp = np.linspace(y[m+1-u_lower], y[m+1-u_upper], u_spacing)

    integral = dt * simpson((u-0.5) ** n * y_intp, dx=du)
    return integral


def convolve(resampled_y, I_table):
    """
    Convolve a (simulated) y = f(t) with the IRF moments stored in I_table.

    Parameters
    ----------
    resampled_y : 1D array
        f(t) values to convolve. For an IRF with time step dt, f(t) must be
        defined with time step dt / 2.
    I_table : 2D array
        An array of precalculated moment integrals from I_moment(). For an IRF
        g(t) with length (number of time intervals) nk_irf, I_table must be of
        size (nk_irf, 3), in which each entry I_table[m, n] is the moment of
        order n at time lag index m.

    Returns
    -------
    h_k : 1D array
        Convolved (f o g)(t).

    """
    nk_irf = len(I_table)
    nk = (len(resampled_y) - 1) // 2

    h_k = np.zeros(nk+1)

    # Vectorized version - cache some array accesses

    I2 = 2 * (resampled_y[2::2] - 2 * resampled_y[1::2] + resampled_y[:-1:2])
    I1 = (resampled_y[2::2] - resampled_y[:-1:2])
    I0 = resampled_y[1::2]

    h_k[0] = 0

    for k in range(1, nk+1):  # Outer loop to k of current point
        # We need only the neighboring points within the window of the IRF
        irf_lowest = max(0, k-nk_irf)
        h_kp = (I0[irf_lowest:k] * I_table[:k, 0][::-1] +
                I1[irf_lowest:k] * I_table[:k, 1][::-1] +
                I2[irf_lowest:k] * I_table[:k, 2][::-1])
        h_k[k] = np.sum(h_kp)

    return h_k


@njit(cache=True)
def n_convolve(resampled_y, I_table):
    """
    Numba-compatible equivalent of convolve().
    As of 1/31/23 no speed improvement.
    """
    nk_irf = len(I_table)
    nk = (len(resampled_y) - 1) // 2

    h_k = np.zeros(nk+1)
    h_k[0] = 0

    for k in range(1, nk+1):   # Outer loop to k of current point
        h_kp = np.zeros(k)
        for kp in range(k):  # Loop k_prime from 1 to kk
            # We need only the neighboring points within the window of the IRF
            if (k - 1 - kp) < nk_irf:
                I2 = 2 * (resampled_y[2*kp+2] -
                          2*resampled_y[2*kp+1] +
                          resampled_y[2*kp])
                I2 *= I_table[k - 1 - kp, 2]

                I1 = (resampled_y[2*kp+2] - resampled_y[2*kp])
                I1 *= I_table[k - 1 - kp, 1]

                I0 = resampled_y[2*kp+1]
                I0 *= I_table[k - 1 - kp, 0]

                h_kp[kp] = I0 + I1 + I2
        h_k[k] = np.sum(h_kp)

    return h_k


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     t = np.linspace(0, 10, 100)
#     max_t = t[-1]
#     dt = t[1] - t[0]
#     double_t = np.arange(0, max_t + dt / 4, dt / 2)

#     # Test 1: f(t) = exp(-t), g(t) = sin(t)
#     # Expected: (f o g)(t) = 0.5 * (exp(-t) + sin(t) - cos(t))

#     f_t = np.exp(-double_t)
#     g_t = np.sin(t)

#     nk_irf = len(g_t)
#     I_table = np.zeros((nk_irf, 3))
#     for i in range(nk_irf - 1):
#         for n in range(3):
#             I_table[i, n] = I_moment(t, g_t, i, n, u_spacing=1000)

#     h_k = convolve(f_t, I_table)
#     expected_h_k = 0.5 * (np.exp(-t) + np.sin(t) - np.cos(t))

#     fig, ax = plt.subplots(1, 1, dpi=240)
#     ax.plot(t, h_k, label="Actual")
#     ax.plot(t, expected_h_k, linestyle='dashed', label="Expected")
#     ax.legend()
#     ax.set_title("Test 1: f(t) = e^-t, g(t) = sin(t)")
#     ax.set_ylabel("(f o g)(t)")
#     ax.set_xlabel("t")

#     # Test 2: f(t) = sin(t), g(t) = exp(-t)
#     # Convolution is commutative, so this should have same result as Test 1
#     f_t = np.sin(double_t)
#     g_t = np.exp(-t)

#     nk_irf = len(g_t)
#     I_table = np.zeros((nk_irf, 3))
#     for i in range(nk_irf - 1):
#         for n in range(3):
#             I_table[i, n] = I_moment(t, g_t, i, n, u_spacing=1000)

#     h_k = convolve(f_t, I_table)
#     expected_h_k = 0.5 * (np.exp(-t) + np.sin(t) - np.cos(t))

#     fig, ax = plt.subplots(1, 1, dpi=240)
#     ax.plot(t, h_k, label="Actual")
#     ax.plot(t, expected_h_k, linestyle='dashed', label="Expected")
#     ax.legend()
#     ax.set_title("Test 2: f(t) = e^-t, g(t) = sin(t)")
#     ax.set_ylabel("(g o f)(t)")
#     ax.set_xlabel("t")

#     # Test 3: f(t) = g(t) = 1 {0 < t < 1}
#     # Window function
#     # This should produce a triangular pulse of length 2 and amplitude 1
#     t = np.linspace(0, 10, 1000)
#     max_t = t[-1]
#     dt = t[1] - t[0]
#     double_t = np.arange(0, max_t + dt / 4, dt / 2)

#     f_t = np.where(double_t < 1, 1, 0)
#     g_t = np.where(t < 1, 1, 0)

#     nk_irf = len(g_t)
#     I_table = np.zeros((nk_irf, 3))
#     for i in range(nk_irf - 1):
#         for n in range(3):
#             I_table[i, n] = I_moment(t, g_t, i, n, u_spacing=1000)

#     h_k = convolve(f_t, I_table)
#     expected_h_k = np.where(t < 1, t, 2 - t)
#     expected_h_k = np.where(t <= 2, expected_h_k, 0)

#     fig, ax = plt.subplots(1, 1, dpi=240)
#     ax.plot(t, h_k, label="Actual")
#     ax.plot(t, expected_h_k, linestyle='dashed', label="Expected")
#     ax.legend()
#     ax.set_title("Test 3: f(t) = g(t) = 1 {0 < t < 1}")
#     ax.set_ylabel("(f o g)(t)")
#     ax.set_xlabel("t")
