import unittest
import numpy as np
from scipy.integrate import solve_ivp

from forward_solver import dydt
from sim_utils import Grid, Parameters, Solution
from metropolis import E_field

class TestUtils(unittest.TestCase):
    
    def test_solver(self):
        s = Solution()
        g = Grid()
        g.thickness = 1000
        g.nx = 100
        g.dx = g.thickness / g.nx
        g.xSteps = np.linspace(g.dx / 2, g.thickness - g.dx/2, g.nx)
        
        g.time = 10
        g.start_time = 0
        g.nt = 100
        g.dt = g.time / g.nt
        g.hmax = 4
        g.tSteps = np.linspace(g.start_time, g.time, g.nt+1)
        
        param_names = ["n0", "p0", "mu_n", "mu_p", "B", 
                       "Sf", "Sb", "tauN", "tauP", "eps", "m"]
        unit_conversions = {"n0":((1e-7) ** 3), "p0":((1e-7) ** 3), 
                            "mu_n":((1e7) ** 2) / (1e9), "mu_p":((1e7) ** 2) / (1e9), 
                            "B":((1e7) ** 3) / (1e9), "Sf":1e-2, "Sb":1e-2}
        do_log = {"n0":1, "p0":1,"mu_n":0,"mu_p":0,"B":1,
                  "Sf":1,"Sb":1,"tauN":0,"tauP":0,"eps":1,
                  "m":0}
        param_info = {"names":param_names,
                      "unit_conversions":unit_conversions,
                      "do_log":do_log,}
        
        ## Nothing ##
        initial_guess ={"n0":0, 
                        "p0":0, 
                        "mu_n":0, 
                        "mu_p":0, 
                        "B":0, 
                        "Sf":0, 
                        "Sb":0, 
                        "tauN":1e99, 
                        "tauP":1e99, 
                        "eps":10, 
                        "m":0}
        
        p = Parameters(param_info, initial_guess)
        p.apply_unit_conversions(param_info)
        
        init_dN = np.ones(g.nx) * 1e10 * unit_conversions["n0"]
        N = init_dN + p.n0
        P = init_dN + p.p0
        E_f = E_field(N, P, p, g.dx)
        
        init_condition = np.concatenate([N, P, E_f], axis=None)
        args = (g,p)
        sol = solve_ivp(dydt, [g.start_time,g.time], init_condition, args=args, t_eval=g.tSteps, method='BDF', max_step=g.hmax)
        data = sol.y.T
        
        s.N, s.P, E_f = np.split(data, [g.nx, 2*g.nx], axis=1)
        np.testing.assert_almost_equal(s.N, np.ones_like(s.N) * init_dN)
        np.testing.assert_almost_equal(s.P, np.ones_like(s.P) * init_dN)
        np.testing.assert_almost_equal(E_f, np.zeros_like(E_f))
        ##############
        
        ## Diff only ##
        initial_guess ={"n0":0, 
                        "p0":0, 
                        "mu_n":100, 
                        "mu_p":100, 
                        "B":0, 
                        "Sf":0, 
                        "Sb":0, 
                        "tauN":1e99, 
                        "tauP":1e99, 
                        "eps":10, 
                        "m":0}
        
        p = Parameters(param_info, initial_guess)
        p.apply_unit_conversions(param_info)
        
        init_dN = np.logspace(14, 8, g.nx) * unit_conversions["n0"]
        N = init_dN + p.n0
        P = init_dN + p.p0
        E_f = E_field(N, P, p, g.dx)
        total_N = np.sum(N)
        
        init_condition = np.concatenate([N, P, E_f], axis=None)
        args = (g,p)
        sol = solve_ivp(dydt, [g.start_time,g.time], init_condition, args=args, t_eval=g.tSteps, method='BDF', max_step=g.hmax)
        data = sol.y.T
        
        s.N, s.P, E_f = np.split(data, [g.nx, 2*g.nx], axis=1)
        np.testing.assert_almost_equal(s.N, np.ones_like(s.N) * total_N / g.nx)
        np.testing.assert_almost_equal(s.P, np.ones_like(s.P) * total_N / g.nx)
        np.testing.assert_almost_equal(E_f, np.zeros_like(E_f))
        ################
        
        ## SRH only ##
        initial_guess ={"n0":0, 
                        "p0":0, 
                        "mu_n":0, 
                        "mu_p":0, 
                        "B":0, 
                        "Sf":0, 
                        "Sb":0, 
                        "tauN":1, 
                        "tauP":1e99, 
                        "eps":10, 
                        "m":0}
        
        p = Parameters(param_info, initial_guess)
        p.apply_unit_conversions(param_info)
        
        init_dN = 1e10 * np.ones(g.nx) * unit_conversions["n0"]
        N = init_dN + p.n0
        P = init_dN + p.p0
        E_f = E_field(N, P, p, g.dx)
        total_N = np.sum(N)
        
        init_condition = np.concatenate([N, P, E_f], axis=None)
        args = (g,p)
        sol = solve_ivp(dydt, [g.start_time,g.time], init_condition, args=args, t_eval=g.tSteps, method='BDF', max_step=g.hmax)
        data = sol.y.T
        
        s.N, s.P, E_f = np.split(data, [g.nx, 2*g.nx], axis=1)
        np.testing.assert_almost_equal(s.N[:,0], init_dN[0] * np.exp(-g.tSteps / 1))
        np.testing.assert_almost_equal(s.P[:,0], init_dN[0] * np.exp(-g.tSteps / 1))
        np.testing.assert_almost_equal(E_f, np.zeros_like(E_f))
        ################