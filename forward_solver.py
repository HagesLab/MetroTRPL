# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:04:09 2022

@author: cfai2
"""
import numpy as np
from numba import njit

# Define constants
eps0 = 8.854 * 1e-12 * 1e-9  # [C / V m] to {C / V nm}
q = 1.0  # [e]
q_C = 1.602e-19  # [C]
kB = 8.61773e-5  # [eV / K]


def dydt(t, y, g, p):
    """Derivative function for drift-diffusion-decay carrier model."""
    # Initialize arrays to store intermediate quantities
    # that do not need to be iteratively solved
    # These are calculated at node edges, of which there are m + 1
    # dn/dx and dp/dx are also node edge values
    Jn = np.zeros((g.nx+1))
    Jp = np.zeros((g.nx+1))

    # These are calculated at node centers, of which there are m
    # dE/dt, dn/dt, and dp/dt are also node center values
    dJz = np.zeros((g.nx))
    rad_rec = np.zeros((g.nx))
    non_rad_rec = np.zeros((g.nx))

    N = y[0:g.nx]
    P = y[g.nx:2*(g.nx)]
    E_field = y[2*(g.nx):]
    N_edges = (N[:-1] + np.roll(N, -1)[:-1]) / 2
    P_edges = (P[:-1] + np.roll(P, -1)[:-1]) / 2

    # Do boundary conditions of Jn, Jp
    Sft = p.Sf * (N[0] * P[0] - p.n0 * p.p0) / (N[0] + P[0])
    Sbt = p.Sb * (N[g.nx-1] * P[g.nx-1] - p.n0 * p.p0) / (N[g.nx-1] + P[g.nx-1])
    Jn[0] = Sft
    Jn[g.nx] = -Sbt
    Jp[0] = -Sft
    Jp[g.nx] = Sbt

    # Calculate Jn, Jp [nm^-2 ns^-1] over the space dimension,
    # np.roll(y,m) shifts the values of array y by m places,
    # allowing for quick approximation of dy/dx ~ (y[m+1] - y[m-1] / 2*dx)
    # over entire array y
    Jn[1:-1] = (p.mu_n * N_edges * q * E_field[1:-1] +
                (p.mu_n*kB*p.Tm) * (np.roll(N, -1)[:-1] - N[:-1]) / g.dx)

    Jp[1:-1] = (p.mu_p * (P_edges) * q * E_field[1:-1] -
                (p.mu_p*kB*p.Tm) * (np.roll(P, -1)[:-1] - P[:-1]) / g.dx)

    # [V nm^-1 ns^-1]
    dEdt = -(Jn + Jp) * ((q_C) / (p.eps * eps0))

    # Calculate recombination (consumption) terms
    rad_rec = p.ks * (N * P - p.n0 * p.p0)
    non_rad_rec = (N * P - p.n0 * p.p0) / ((p.tauN * P) + (p.tauP * N))
    auger_rec = (p.Cn * N + p.Cp * P) * (N * P - p.n0 * p.p0)

    # Calculate dJn/dx
    dJz = (np.roll(Jn, -1)[:-1] - Jn[:-1]) / (g.dx)

    dNdt = ((1/q) * dJz - rad_rec - non_rad_rec - auger_rec)

    # Calculate dJp/dx
    dJz = (np.roll(Jp, -1)[:-1] - Jp[:-1]) / (g.dx)

    dPdt = ((1/q) * -dJz - rad_rec - non_rad_rec - auger_rec)

    # Package results
    dy = np.concatenate([dNdt, dPdt, dEdt], axis=None)
    return dy


@njit(cache=True)
def dydt_numba(t, y, L, dx, N0, P0, mu_n, mu_p, r_rad, CN, CP, sr0, srL,
               tauN, tauP, Lambda, Tm):
    """ Numba translation of dydt() """
    Jn = np.zeros((L+1))
    Jp = np.zeros((L+1))
    dy = np.zeros(3*L+1)

    N = y[0:L]
    P = y[L:2*(L)]
    E_field = y[2*(L):]

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
        dy[2*L+i] = -(Jn[i] + Jp[i]) * Lambda

    # Auger + Radiative + Bulk SRH
    recomb = ((CN * N + CP * P) + r_rad + 1 / ((tauN * P) + (tauP * N))) * NP

    for i in range(len(Jn) - 1):
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
