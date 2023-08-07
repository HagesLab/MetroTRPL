"""
Specific curve fitting functions.

@author: frooia
"""
from numpy.polynomial import Polynomial
import numpy as np

# Fit functions of different orders
def poly(xin, *args):
    """
    Arbitrary-order polynomial of form
    f(x) = (A + B*x + ... + Y*x**(n-1) + Z*x**n),
    
    in which args is a list of coefs [A, B, ..., Y, Z]
    
    Uses Horner's method for faster order of operations

    Parameters
    ----------
    xin : 1D ndarray
        x values, e.g. log(delay time).
    *args : list-like
        Sequence of polynomial coefs.

    Returns
    -------
    fit_y : 1D ndarray
        f(x) values.

    """
    # Baseline, naive, 59 s
    # fit_y = 0
    # for i in range(len(args)):
    #     fit_y += args[i] * xin ** i
        
    # Good, Horner's method, 4.7 s
    fit_y = args[len(args)-1]
    for i in range(len(args)-2, -1,-1):
        fit_y = fit_y * xin + args[i]
    
    return fit_y

def multiexp(x, *args):
    """
    Arbitrary-order multiexponential of form
    f(x) = a_0 * exp(k_0 * x) + a_1 * exp(k_1 * x) + ... + a_z * exp(k_z * x)
    
    in which args is a list of rates followed by coefs [k_0, k_1, ..., k_z, a_0, a_1, ..., a_z]

    Parameters
    ----------
    xin : 1D ndarray
        x values, e.g. delay time.
    *args : list-like
        Sequence of rates and coefs.

    Returns
    -------
    fit_y : 1D ndarray
        f(x) values.

    """
    fit_y = np.zeros_like(x, dtype=float)
    n = len(args) // 2
    for i in range(n):
        fit_y += args[i+n] * np.exp(args[i] * x)
        
    return fit_y


def preproc(xin, yin, loglog=True, interpolate=True):
    """
    Preprocessing for TRPL data prior to curve fitting.    

    Parameters
    ----------
    xin : 1D ndarray
        List of time points used by the solver.
    yin : 1D ndarray
        Simulated PL curve.
    loglog : bool, optional
        Whether to take log(time) and log(PL). Doing this avoids biasing curve fit
        toward high values or long times. The default is True.
    interpolate : bool, optional
        Whether to re-interpolate data to uniform x grid. Data which are uniform
        in normal space are not in log space. Also helps avoid biasing curve fit.
        The default is True.

    Returns
    -------
    xdata : 1D ndarray
        Processed time points.
    ydata : 1D ndarray
        Processed PL curve.

    """
    xdata, ydata = xin, yin

    # Filter ydata values that are less than 1
    ydata[ydata <= 1] = 1

    if loglog:
        xdata = np.log10(xdata[1:])
        ydata = np.log10(ydata[1:])
    if interpolate:
        x_interp = np.linspace(min(xdata), max(xdata), num=len(xdata))

        j = np.searchsorted(xdata, x_interp) - 1
        d = (x_interp - xdata[j]) / (xdata[j + 1] - xdata[j])
        ydata = (1 - d) * ydata[j] + ydata[j + 1] * d
        
        xdata = x_interp

    return xdata, ydata

# Returns curve of best fit and prints parameters
def polyfit(xin, yin, order=16, truncate_y=None):
    """
    Variable order polynomial fit (A + B*x + ... + Y*x**(n-1) + Z*x**n) of simulated data
    
    Parameters
    ----------
    xin : 1D array
        x values, e.g. log(delay time).
    yin : 1D array
        y values, e.g. log(PL).
    order : int, optional
        Polynomial fit order. The default is 16.
    truncate_y : float, optional
        If not None, ignores PL values below this value for purposes of fitting

    Returns
    -------
    params : 1D array
        List of fitted coefficients A, B, C...Z .
    poly : function
        Function that can be used to regenerate the fitted polynomial from coefs.
    success : bool
        Whether the fitting was successful

    """
    success = True
    if truncate_y is not None:
        cutoff = np.argmax(yin < truncate_y) # First occurance
        if cutoff == 0: cutoff = None
        xin = xin[:cutoff]
        yin = yin[:cutoff]

    # Best: 4 in 4.7 s, 20 in 28.8 s
    #params, covars = curve_fit(poly_fit, xin, yin, p0=np.zeros(order+1))
    
    # Best: 4 in 0.15 s, 20 in 0.84 s 
    p = Polynomial.fit(xin, yin, order)
    params = p.convert().coef

    return params, poly, success

def cumtrapz(x, y):
    """
    Cumulative trapezoidal integral

    Parameters
    ----------
    x : 1D array
        x values, used to calculate spacing
    y : 1D array
        f(x) values

    Returns
    -------
    Y : 1D array
        Cumulative integral; each Y[i] is integral of f(x) from x[0] to x[i]

    """
    Y = np.zeros_like(x, dtype=float)
    Y[1:] = np.cumsum(np.diff(x)/2 * (y[:-1] + np.roll(y, -1)[:-1]))
    return Y

def expfit(x, y, order):
    """
    No-iteration least-squares fit to sum of exponentials, with fit_order f;
    y ~ a_0 * exp(k_0 * x) + a_1 * exp(k_1 * x) + ... + a_f * exp(k_f * x) 
    
    Normalizes both x and y before fitting
    
    https://math.stackexchange.com/questions/1428566/fit-sum-of-exponentials/3808325#3808325

    Parameters
    ----------
    x : 1D array
        x values.
    y : 1D array
        f(x) values.
    order : int
        Fitting order f.

    Returns
    -------
    params : 1D array
        Array of exponentials and coefs [k_0, k_1, ..., k_f, a_0, a_1, ..., a_f]
    multiexp : function
        Function that can be used to regenerate the fitted polynomial from coefs.
    success : bool
        Whether the fitting was successful
    
    """
    success = True

    # calculate integrals and powers
    scale_x = np.amax(x)
    x_ = np.array(x / scale_x)
    scale_y = np.amax(y)
    y_ = np.array(y / scale_y)
    Y = np.zeros((2 * order, len(x_)))
    Y[0] = cumtrapz(x_, y_)
    for i in range(1, order):
        Y[i] = cumtrapz(x_, Y[i-1])
        
    j = order - 1
    for i in range(order, len(Y)):
        Y[i] = x_ ** j
        j -= 1
    
    # get exponentials lambdas
    A = np.dot(np.linalg.pinv(Y).T, y_)
    
    E = np.zeros((order, order))
    E[0] = A[:order]
    E[1:] = np.eye(order)[:-1]
    
    lambdas = np.linalg.eig(E)[0]
    if any(np.iscomplex(lambdas)):
        success = False
    # get exponentials multipliers
    X = np.zeros((order, len(x)))
    for i in range(order):
        X[i] = np.exp(lambdas[i] * x_)
    P = np.dot(np.linalg.pinv(X).T, y_)
    
    P *= scale_y
    lambdas /= scale_x
    
    # sort by descending exponentials / ascending lifetimes
    s = np.argsort(lambdas)
    lambdas = lambdas[s]
    P = P[s]
    
    # P[lambdas >= 0] = 0
    # P[P < 0] = 0
    params = np.concatenate((lambdas, P))
    return params, multiexp, success

if __name__ == "__main__":
    dx = 0.02
    x = np.arange(dx, 1000+dx/2, dx)
    y = 5*np.exp(-1e-2*x) + 4*np.exp(-5e-3*x) + 2*np.exp(-1e-3*x)
    y *= 1e20
    fit_order = 3
    params = expfit(x, y, 3)
    print(multiexp(x,*params))