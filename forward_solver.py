# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:04:09 2022

@author: cfai2
"""
import numpy as np
from scipy.integrate import solve_ivp, odeint
from numba import njit

try:
    from nn_features import NeuralNetwork
    nn = NeuralNetwork()
except (ImportError, AttributeError) as e:
    raise ImportError(f"Failed to load neural network library (Reason): {e}") from e

DEFAULT_RTOL = 1e-7
DEFAULT_ATOL = 1e-10
# Define constants
eps0 = 8.854 * 1e-12 * 1e-9  # [C / V m] to {C / V nm}
q = 1.0  # [e]
q_C = 1.602e-19  # [C]
kB = 8.61773e-5  # [eV / K]

def E_field(N, P, n0, p0, eps, dx, corner_E=0):
    if N.ndim == 1:
        E = corner_E + q_C / (eps * eps0) * dx * \
            np.cumsum(((P - p0) - (N - n0)))
        E = np.concatenate(([corner_E], E))
    elif N.ndim == 2:
        E = corner_E + q_C / (eps * eps0) * dx * \
            np.cumsum(((P - p0) - (N - n0)), axis=1)
        num_tsteps = len(N)
        E = np.concatenate((np.ones(shape=(num_tsteps, 1))*corner_E, E), axis=1)
    else:
        raise NotImplementedError(f"Unsupported number of dimensions: {N.ndim}")
    return E


def solve(iniPar, g, state, indexes, meas="TRPL", units=None, solver=("solveivp",), model="std", ini_mode="density",
          RTOL=None, ATOL=None):
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
    state : ndarray
        An array of parameters, ordered according to param_info["names"],
        corresponding to a state in the parameter space.
    indexes : dict[str, int]
        A map of parameter names and their corresponding indices in the state array.
    meas : str, optional
        Type of measurement (e.g. TRPL, TRTS) being simulated. The default is "TRPL".
    units : dict[str], optional
        Unit conversions to be applied to each parameter. The default is None.
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
    ini_mode : str, optional
        One of the following, depending on what the iniPar are:
        "density" - iniPar are carrier density arrays with lengths equal to g.nx
        "fluence" - iniPar are arrays of length 3 - [fluence, alpha, direction]
        The default is "density".
    RTOL, ATOL : float, optional
        Tolerance parameters for scipy solvers. See the solve_ivp() docs for details.

    Returns
    -------
    sol : np.ndarray
        Array of values (e.g. TRPL) from final simulation.
    next_init : np.ndarray
        Values (e.g. the electron profile) at the final time of the simulation.

    """
    if units is None:
        units = np.ones_like(state)
    if RTOL is None:
        RTOL = DEFAULT_RTOL
    if ATOL is None:
        ATOL = DEFAULT_ATOL
    if solver[0] == "solveivp" or solver[0] == "odeint" or solver[0] == "diagnostic":
        if ini_mode == "density":
            if len(iniPar) == g.nx:         # If list of initial values
                init_dN = iniPar * 1e-21    # [cm^-3] to [nm^-3]
            else:
                raise ValueError(f"Expected {g.nx} initial densities but initial condition file has {len(iniPar)}")
        elif ini_mode == "fluence":         # List of parameters
            if len(iniPar) <= 3:
                fluence = iniPar[0] * 1e-14 # [cm^-2] to [nm^-2]
                alpha = iniPar[1] * 1e-7    # [cm^-1] to [nm^-1]
                init_dN = fluence * alpha * np.exp(-alpha * g.xSteps)
                try:
                    init_dN = init_dN[::np.sign(int(iniPar[2]))]
                except (IndexError, ValueError):
                    pass
            else:
                raise ValueError(f"Expected only fluence, absorption coef, and direction but initial condition file has {len(iniPar)} values")
        else:
            raise ValueError("Invalid ini_mode - must be 'density' or 'fluence'")

        state *= units
        N = init_dN + state[indexes["n0"]]
        P = init_dN + state[indexes["p0"]]
        E_f = E_field(N, P, state[indexes["n0"]], state[indexes["p0"]], state[indexes["eps"]], g.dx)


        # Depends on how many dependent variables and parameters are in the selected model
        if model == "std":
            init_condition = np.concatenate([N, P, E_f], axis=None)
            args = (g.nx, g.dx, state[indexes["n0"]], state[indexes["p0"]], state[indexes["mu_n"]], state[indexes["mu_p"]],
                    state[indexes["ks"]], state[indexes["Cn"]], state[indexes["Cp"]],
                    state[indexes["Sf"]], state[indexes["Sb"]], state[indexes["tauN"]], state[indexes["tauP"]],
                    ((q_C) / (state[indexes["eps"]] * eps0)), state[indexes["Tm"]],)
        elif model == "traps":
            init_condition = np.concatenate([N, np.zeros_like(N), P, E_f], axis=None)
            args = (g.nx, g.dx, state[indexes["n0"]], state[indexes["p0"]], state[indexes["mu_n"]], state[indexes["mu_p"]],
                    state[indexes["ks"]], state[indexes["Cn"]], state[indexes["Cp"]],
                    state[indexes["Sf"]], state[indexes["Sb"]], state[indexes["tauN"]], state[indexes["tauP"]],
                    ((q_C) / (state[indexes["eps"]] * eps0)), state[indexes["Tm"]],
                    state[indexes["kC"]], state[indexes["Nt"]], state[indexes["tauE"]],)
        else:
            raise ValueError(f"Invalid model {model}")

        dy = lambda t, y: MODELS[model](t, y, *args)

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
            sol = solve_ivp(dy, [g.start_time, g.final_time], init_condition,
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
            N, P, E_f = np.split(data, [g.nx, 2*g.nx], axis=1)
        elif model == "traps":
            N, N_trap, P, E_f = np.split(data, [g.nx, 2*g.nx, 3*g.nx], axis=1)

        if meas == "TRPL":
            PL = calculate_PL(g.dx, N, P, state[indexes["ks"]], state[indexes["n0"]], state[indexes["p0"]])
            state /= units  # [nm, V, ns] to [cm, V, s]
            PL *= 1e23                            # [nm^-2 ns^-1] to [cm^-2 s^-1]
            i_final = np.argmax(PL < g.min_y)
            if PL[i_final] < g.min_y:
                PL[i_final:] = g.min_y
            return PL
        elif meas == "TRTS":
            trts = calculate_TRTS(g.dx, N, P, state[indexes["mu_n"]], state[indexes["mu_p"]], state[indexes["n0"]], state[indexes["p0"]])
            state /= units
            trts *= 1e9
            i_final = np.argmax(trts < g.min_y)
            if trts[i_final] < g.min_y:
                trts[i_final:] = g.min_y
            return trts
        else:
            raise NotImplementedError("TRTS or TRPL only")

    elif solver[0] == "NN":
        if meas != "TRPL":
            raise NotImplementedError("TRPL only")

        if not nn.has_model:
            nn.load_model(solver[1], solver[2])

        scaled_matPar = np.zeros((1, 14))
        scaled_matPar[0] = [state[indexes["p0"]], state[indexes["mu_n"]], state[indexes["mu_p"]],
                            state[indexes["ks"]], state[indexes["Cn"]], state[indexes["Cp"]],
                            state[indexes["Sf"]], state[indexes["Sb"]], state[indexes["tauN"]], state[indexes["tauP"]],
                            state[indexes["eps"]]**-1,
                            iniPar[0], iniPar[1], g.thickness]

        pl_from_NN = nn.predict(g.tSteps, scaled_matPar)
        return pl_from_NN

    else:
        raise NotImplementedError


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


def dydt(t, y, L, dx, N0, P0, mu_n, mu_p, r_rad, CN, CP, sr0, srL,
               tauN, tauP, Lambda, Tm):
    """Derivative function for drift-diffusion-decay carrier model."""

    Jn = np.zeros(L+1)
    Jp = np.zeros(L+1)
    dy = np.zeros(3*L+1)

    N = y[0:L]
    P = y[L:2*L]
    Efield = y[2*L:]
    NP = (N * P - N0 * P0)

    N_edges = (N[:-1] + np.roll(N, -1)[:-1]) / 2
    P_edges = (P[:-1] + np.roll(P, -1)[:-1]) / 2

    # Do boundary conditions of Jn, Jp
    Sft = sr0 * NP[0] / (N[0] + P[0])
    Sbt = srL * NP[-1] / (N[-1] + P[-1])

    Jn[0] = Sft
    Jn[L] = -Sbt
    Jp[0] = -Sft
    Jp[L] = Sbt

    # Calculate Jn, Jp [nm^-2 ns^-1] over the space dimension,
    # np.roll(y,m) shifts the values of array y by m places,
    # allowing for quick approximation of dy/dx ~ (y[m+1] - y[m-1] / 2*dx)
    # over entire array y
    Jn[1:-1] = (mu_n * N_edges * Efield[1:-1] +
                (mu_n*kB*Tm) * (np.roll(N, -1)[:-1] - N[:-1]) / dx)

    Jp[1:-1] = (mu_p * P_edges * Efield[1:-1] -
                (mu_p*kB*Tm) * (np.roll(P, -1)[:-1] - P[:-1]) / dx)

    # dEdt [V nm^-1 ns^-1]
    # Lambda = q / (eps * eps0)
    dy[2*L:] = -(Jn + Jp) * Lambda

    # Auger + Radiative + Bulk SRH
    recomb = ((CN * N + CP * P) + r_rad + 1 / ((tauN * P) + (tauP * N))) * NP

    # Calculate dndt, dpdt
    dJz = (np.roll(Jn, -1)[:-1] - Jn[:-1]) / dx

    dy[:L] = dJz - recomb

    # Calculate dJp/dx
    dJz = (np.roll(Jp, -1)[:-1] - Jp[:-1]) / dx

    dy[L:2*L] = -dJz - recomb

    return dy


@njit(cache=True)
def dydt_numba(t, y, L, dx, N0, P0, mu_n, mu_p, r_rad, CN, CP, sr0, srL,
               tauN, tauP, Lambda, Tm):
    """ Numba translation of dydt() """
    Jn = np.zeros(L+1)
    Jp = np.zeros(L+1)
    dy = np.zeros(3*L+1)

    N = y[0:L]
    P = y[L:2*L]
    Efield = y[2*L:]

    NP = (N * P - N0 * P0)

    Sft = sr0 * NP[0] / (N[0] + P[0])
    Sbt = srL * NP[-1] / (N[-1] + P[-1])

    Jn[0] = Sft
    Jn[L] = -Sbt
    Jp[0] = -Sft
    Jp[L] = Sbt

    # DN = mu_n*kB*T/q
    for i in range(1, L):
        Jn[i] = mu_n*((N[i-1] + N[i]) / 2 * Efield[i]) + \
            mu_n*kB*Tm*((N[i] - N[i-1]) / dx)
        Jp[i] = mu_p*((P[i-1] + P[i]) / 2 * Efield[i]) - \
            mu_p*kB*Tm*((P[i] - P[i-1]) / dx)

    # Lambda = q / (eps * eps0)
    for i in range(L+1):
        dy[2*L+i] = -(Jn[i] + Jp[i]) * Lambda

    # Auger + Radiative + Bulk SRH
    recomb = ((CN * N + CP * P) + r_rad + 1 / ((tauN * P) + (tauP * N))) * NP

    for i in range(L):
        dy[i] = ((Jn[i+1] - Jn[i]) / dx - recomb[i])
        dy[L+i] = (-(Jp[i+1] - Jp[i]) / dx - recomb[i])

    return dy

@njit(cache=True)
def dydt_numba_traps(t, y, L, dx, N0, P0, mu_n, mu_p, r_rad, CN, CP, sr0, srL,
                     tauN, tauP, Lambda, Tm, kC, Nt, tauE):
    """ Numba translation of dydt() """
    Jn = np.zeros((L+1))
    Jp = np.zeros((L+1))
    dy = np.zeros(4*L+1)

    N = y[0:L]
    N_trap = y[L:2*L]
    P = y[2*L:3*(L)]
    E_field = y[3*(L):]

    NP = (N * P - N0 * P0)

    Sft = sr0 * NP[0] / (N[0] + P[0])
    Sbt = srL * NP[-1] / (N[-1] + P[-1])

    Jn[0] = Sft
    Jn[L] = -Sbt
    Jp[0] = -Sft
    Jp[L] = Sbt

    # DN = mu_n*kB*T/q
    for i in range(1, len(Jn) - 1):
        Jn[i] = mu_n*((N[i-1] + N[i]) / 2 * E_field[i]) + \
            mu_n*kB*Tm*((N[i] - N[i-1]) / dx)
        Jp[i] = mu_p*((P[i-1] + P[i]) / 2 * E_field[i]) - \
            mu_p*kB*Tm*((P[i] - P[i-1]) / dx)

    # Lambda = q / (eps * eps0)
    for i in range(len(Jn)):
        dy[3*L+i] = -(Jn[i] + Jp[i]) * Lambda

    # Auger + Radiative + Bulk SRH
    recomb = ((CN * N + CP * P) + r_rad + 1 / ((tauN * P) + (tauP * N))) * NP
    trap = kC * N * (Nt - N_trap)
    detrap = N_trap / tauE

    for i in range(len(Jn) - 1):
        dy[i] = ((Jn[i+1] - Jn[i]) / dx - recomb[i] + detrap[i] - trap[i])
        dy[L+i] = trap[i] - detrap[i]
        dy[2*L+i] = (-(Jp[i+1] - Jp[i]) / dx - recomb[i])

    return dy

MODELS = {
    "std": dydt_numba,
    "traps": dydt_numba_traps
}
