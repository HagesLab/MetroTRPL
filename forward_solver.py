# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:04:09 2022

@author: cfai2
"""
import numpy as np
from numba import njit

## Define constants
eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
q = 1.0 # [e]
q_C = 1.602e-19 # [C]
kB = 8.61773e-5  # [eV / K]

def dydt(t, y, g, p):
    """Derivative function for drift-diffusion-decay carrier model."""
    ## Initialize arrays to store intermediate quantities that do not need to be iteratively solved
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
    N_edges = (N[:-1] + np.roll(N, -1)[:-1]) / 2 # Excluding the boundaries; see the following FIXME
    P_edges = (P[:-1] + np.roll(P, -1)[:-1]) / 2
    
    ## Do boundary conditions of Jn, Jp
    Sft = p.Sf * (N[0] * P[0] - p.n0 * p.p0) / (N[0] + P[0])
    Sbt = p.Sb * (N[g.nx-1] * P[g.nx-1] - p.n0 * p.p0) / (N[g.nx-1] + P[g.nx-1])
    Jn[0] = Sft
    Jn[g.nx] = -Sbt
    Jp[0] = -Sft
    Jp[g.nx] = Sbt

    ## Calculate Jn, Jp [nm^-2 ns^-1] over the space dimension, 
    # Jn(t) ~ N(t) * E_field(t) + (dN/dt)
    # np.roll(y,m) shifts the values of array y by m places, allowing for quick approximation of dy/dx ~ (y[m+1] - y[m-1] / 2*dx) over entire array y
    Jn[1:-1] = ( p.mu_n * N_edges * q * E_field[1:-1] +
                (p.mu_n*kB*p.Tm) * (np.roll(N,-1)[:-1] - N[:-1]) / g.dx  )

    ## Changed sign
    Jp[1:-1] = ( p.mu_p * (P_edges) * q * E_field[1:-1] 
                -(p.mu_p*kB*p.Tm) * (np.roll(P, -1)[:-1] - P[:-1]) / g.dx  )


    # [V nm^-1 ns^-1]
    dEdt = -(Jn + Jp) * ((q_C) / (p.eps * eps0))
    
    ## Calculate recombination (consumption) terms
    rad_rec = p.ks * (N * P - p.n0 * p.p0)
    non_rad_rec = (N * P - p.n0 * p.p0) / ((p.tauN * P) + (p.tauP * N))
    auger_rec = (p.Cn * N + p.Cp * P) * (N * P - p.n0 * p.p0)

    ## Calculate dJn/dx
    dJz = (np.roll(Jn, -1)[:-1] - Jn[:-1]) / (g.dx)

    ## N(t) = N(t-1) + dt * (dN/dt)
    #N_new = np.maximum(N_previous + dt * ((1/q) * dJz - rad_rec - non_rad_rec + G_array), 0)
    dNdt = ((1/q) * dJz - rad_rec - non_rad_rec - auger_rec)

    ## Calculate dJp/dx
    dJz = (np.roll(Jp, -1)[:-1] - Jp[:-1]) / (g.dx)

    ## P(t) = P(t-1) + dt * (dP/dt)
    #P_new = np.maximum(P_previous + dt * ((1/q) * dJz - rad_rec - non_rad_rec + G_array), 0)
    dPdt = ((1/q) * -dJz - rad_rec - non_rad_rec - auger_rec)


    ## Package results
    dydt = np.concatenate([dNdt, dPdt, dEdt], axis=None)
    return dydt

@njit(cache=True)
def dydt_numba(t, y, L, dx, N0, P0, mu_n, mu_p, r_rad, CN, CP, sr0, srL, tauN, tauP, Lambda, Tm, kB=8.61773e-5):
    Jn = np.zeros((L+1))
    Jp = np.zeros((L+1))
    dydt = np.zeros(3*L+1)

    N = y[0:L]
    P = y[L:2*(L)]
    E_field = y[2*(L):]
    
    NP = (N * P - N0 * P0)
    
    Sft = sr0 * NP[0] / (N[0] + P[0])
    Sbt = srL * NP[-1] / (N[-1] + P[-1])
    
    ## To invert the E field: switch these ##
    Jn[0] = Sft
    Jn[L] = -Sbt
    Jp[0] = -Sft
    Jp[L] = Sbt

    # DN = mu_n*kB*T/q
    for i in range(1, len(Jn) - 1):
        Jn[i] = mu_n*((N[i-1] + N[i]) / 2 * E_field[i]) + mu_n*kB*Tm*((N[i] - N[i-1]) / dx)
        Jp[i] = mu_p*((P[i-1] + P[i]) / 2 * E_field[i]) - mu_p*kB*Tm*((P[i] - P[i-1]) / dx)
    
    for i in range(len(Jn)):
        dydt[2*L+i] = -(Jn[i] + Jp[i]) * Lambda
    
    # Auger + Radiative + Bulk SRH
    recomb = ((CN * N + CP * P) + r_rad + 1 / ((tauN * P) + (tauP * N))) * NP

    for i in range(len(Jn) - 1):
        dydt[i] =   ( (Jn[i+1] - Jn[i]) / dx - recomb[i])
        dydt[L+i] = (-(Jp[i+1] - Jp[i]) / dx - recomb[i])
    #print(t)
        
    return dydt

if __name__ == "__main__":
    class Tuple():
        def __init__(self):
            return
    L = 128
    y = np.ones(3*L+1)
    g = Tuple()
    g.nx = L
    g.dx = 23.4375
    p = Tuple()
    p.n0 = 1e-13
    p.p0 = 1e-8
    p.mu_n = 32000000
    p.mu_p = 8000000
    p.Tm = 300
    p.ks = 200
    p.Cn = 1e-66
    p.Cp = 1e-66
    p.Sf = 100
    p.Sb = 10
    p.tauN = 10
    p.tauP = 10
    p.eps = 9.4
    import time
    t0 = time.perf_counter()
    for i in range(100000):
        f1 = dydt(0, y, g, p)
   
    print(time.perf_counter() - t0)
    
    t0 = time.perf_counter()
    for i in range(100000):
        f2 = dydt_numba(0, y, L, g.dx, p.n0, p.p0, p.mu_n, p.mu_p, p.ks, p.Cn, p.Cp, p.Sf, p.Sb, p.tauN, p.tauP, 
                        q_C / (p.eps * eps0), p.Tm)
    
    print(time.perf_counter() - t0)