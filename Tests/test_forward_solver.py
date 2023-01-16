import unittest
import numpy as np
from scipy.integrate import solve_ivp

from forward_solver import dydt, dydt_numba
from sim_utils import Grid, Parameters, Solution
from metropolis import E_field

## Define constants
eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
q = 1.0 # [e]
q_C = 1.602e-19 # [C]
kB = 8.61773e-5  # [eV / K]

class TestUtils(unittest.TestCase):
    
    def setUp(self):
        self.s = Solution()
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
        self.g = g
        param_names = ["n0", "p0", "mu_n", "mu_p", "ks", "Cn", "Cp",
                       "Sf", "Sb", "tauN", "tauP", "eps", "Tm", "m"]
        self.unit_conversions = {"n0":((1e-7) ** 3), "p0":((1e-7) ** 3), 
                                 "mu_n":((1e7) ** 2) / (1e9), "mu_p":((1e7) ** 2) / (1e9), 
                                 "ks":((1e7) ** 3) / (1e9), 
                                 "Cn":((1e7) ** 6) / (1e9), "Cp":((1e7) ** 6) / (1e9),
                                 "Sf":1e-2, "Sb":1e-2}
        do_log = {"n0":1, "p0":1,"mu_n":0,"mu_p":0,"ks":1,"Cn":1, "Cp":1,
                  "Sf":1,"Sb":1,"tauN":0,"tauP":0,"eps":1,
                  "Tm":0, "m":0}
        active = {"n0":1, "p0":1,"mu_n":1,"mu_p":1,"ks":1,"Cn":1, "Cp":1,
                  "Sf":1,"Sb":1,"tauN":1,"tauP":1,"eps":1,
                  "Tm":0, "m":0}
        self.param_info = {"names":param_names,
                           "unit_conversions":self.unit_conversions,
                           "do_log":do_log,
                           "active":active}
        pass
    
    def test_solver_nothing(self):
        """ The carrier dynamics model is a sum of diffusion, electronic drift,
            and several recombination processes. Here we test each process
            individually.
        """
        
        ## Nothing ##
        initial_guess ={"n0":0, 
                        "p0":0, 
                        "mu_n":0, 
                        "mu_p":0, 
                        "ks":0, 
                        "Sf":0, 
                        "Sb":0,
                        "Cn":0,
                        "Cp":0,
                        "tauN":1e99, 
                        "tauP":1e99, 
                        "eps":10, 
                        "Tm":300,
                        "m":0}
        
        self.param_info["init_guess"] = initial_guess
        p = Parameters(self.param_info)

        p.apply_unit_conversions(self.param_info)
        
        init_dN = np.ones(self.g.nx) * 1e10 * self.unit_conversions["n0"]
        N = init_dN + p.n0
        P = init_dN + p.p0
        E_f = E_field(N, P, p, self.g.dx)
        
        init_condition = np.concatenate([N, P, E_f], axis=None)
        args = (self.g,p)
        sol = solve_ivp(dydt, [self.g.start_time,self.g.time], init_condition, 
                        args=args, t_eval=self.g.tSteps, method='LSODA', 
                        max_step=self.g.hmax)
        data = sol.y.T
        
        s = self.s
        s.N, s.P, E_f = np.split(data, [self.g.nx, 2*self.g.nx], axis=1)
        np.testing.assert_almost_equal(s.N, np.ones_like(s.N) * init_dN)
        np.testing.assert_almost_equal(s.P, np.ones_like(s.P) * init_dN)
        np.testing.assert_almost_equal(E_f, np.zeros_like(E_f))
        
        # The numba version should give same output as the non-numba version
        sol2 = solve_ivp(dydt_numba, [self.g.start_time, self.g.time], init_condition,
                         args=(self.g.nx, self.g.dx, p.n0, p.p0, p.mu_n, p.mu_p, p.ks,
                               p.Cn, p.Cp, p.Sf, p.Sb, p.tauN, p.tauP, (q_C / (p.eps * eps0)),
                               p.Tm),
                         t_eval=self.g.tSteps, method='LSODA', max_step=self.g.hmax)
        data2 = sol2.y.T
        np.testing.assert_equal(data, data2)
        ##############
        
    def test_solver_diffusion(self):
        ## Diff only ##
        initial_guess ={"n0":0, 
                        "p0":0, 
                        "mu_n":100, 
                        "mu_p":100, 
                        "ks":0, 
                        "Sf":0, 
                        "Sb":0, 
                        "Cn":0,
                        "Cp":0,
                        "tauN":1e99, 
                        "tauP":1e99, 
                        "eps":10, 
                        "Tm":300,
                        "m":0}
        
        self.param_info["init_guess"] = initial_guess
        p = Parameters(self.param_info)
        p.apply_unit_conversions(self.param_info)
        
        init_dN = np.logspace(14, 8, self.g.nx) * self.unit_conversions["n0"]
        N = init_dN + p.n0
        P = init_dN + p.p0
        E_f = E_field(N, P, p, self.g.dx)
        total_N = np.sum(N)
        
        init_condition = np.concatenate([N, P, E_f], axis=None)
        args = (self.g,p)
        sol = solve_ivp(dydt, [self.g.start_time,self.g.time], init_condition, 
                        args=args, t_eval=self.g.tSteps, method='LSODA', 
                        max_step=self.g.hmax)
        data = sol.y.T
        
        s = self.s
        s.N, s.P, E_f = np.split(data, [self.g.nx, 2*self.g.nx], axis=1)
        np.testing.assert_almost_equal(s.N, np.ones_like(s.N) * total_N / self.g.nx)
        np.testing.assert_almost_equal(s.P, np.ones_like(s.P) * total_N / self.g.nx)
        np.testing.assert_almost_equal(E_f, np.zeros_like(E_f))
        
        sol2 = solve_ivp(dydt_numba, [self.g.start_time, self.g.time], init_condition,
                         args=(self.g.nx, self.g.dx, p.n0, p.p0, p.mu_n, p.mu_p, p.ks,
                               p.Cn, p.Cp, p.Sf, p.Sb, p.tauN, p.tauP, (q_C / (p.eps * eps0)),
                               p.Tm),
                         t_eval=self.g.tSteps, method='LSODA', max_step=self.g.hmax)
        data2 = sol2.y.T
        
        # Diffusion is more prone to uncertainty than uniform recombination
        np.testing.assert_almost_equal(data, data2)
        ################
        
    def test_solver_LI_SRH(self):
        ## SRH only - effective lifetime = tau_n##
        initial_guess ={"n0":0, 
                        "p0":0, 
                        "mu_n":0, 
                        "mu_p":0, 
                        "ks":0, 
                        "Sf":0, 
                        "Sb":0, 
                        "Cn":0,
                        "Cp":0,
                        "tauN":1, 
                        "tauP":1e99, 
                        "eps":10, 
                        "Tm":300,
                        "m":0}
        
        self.param_info["init_guess"] = initial_guess
        p = Parameters(self.param_info)
        p.apply_unit_conversions(self.param_info)
        
        init_dN = 1e10 * np.ones(self.g.nx) * self.unit_conversions["n0"]
        N = init_dN + p.n0
        P = init_dN + p.p0
        E_f = E_field(N, P, p, self.g.dx)

        init_condition = np.concatenate([N, P, E_f], axis=None)
        args = (self.g,p)
        sol = solve_ivp(dydt, [self.g.start_time,self.g.time], init_condition, 
                        args=args, t_eval=self.g.tSteps, method='LSODA', 
                        max_step=self.g.hmax)
        data = sol.y.T
        
        s = self.s
        s.N, s.P, E_f = np.split(data, [self.g.nx, 2*self.g.nx], axis=1)
        tau = initial_guess["tauN"]
        np.testing.assert_almost_equal(s.N[:,0], init_dN[0] * np.exp(-self.g.tSteps / tau))
        np.testing.assert_almost_equal(s.P[:,0], init_dN[0] * np.exp(-self.g.tSteps / tau))
        np.testing.assert_almost_equal(E_f, np.zeros_like(E_f))
        sol2 = solve_ivp(dydt_numba, [self.g.start_time, self.g.time], init_condition,
                         args=(self.g.nx, self.g.dx, p.n0, p.p0, p.mu_n, p.mu_p, p.ks,
                               p.Cn, p.Cp, p.Sf, p.Sb, p.tauN, p.tauP, (q_C / (p.eps * eps0)),
                               p.Tm),
                         t_eval=self.g.tSteps, method='LSODA', max_step=self.g.hmax)
        data2 = sol2.y.T
        np.testing.assert_equal(data, data2)
        ################
        
    def test_solver_HI_srh(self):
        ## SRH only, high inj - effective lifetime = tau_n + tau_p ##
        initial_guess ={"n0":0, 
                        "p0":0, 
                        "mu_n":0, 
                        "mu_p":0, 
                        "ks":0, 
                        "Sf":0, 
                        "Sb":0, 
                        "Cn":0,
                        "Cp":0,
                        "tauN":1, 
                        "tauP":1, 
                        "eps":10, 
                        "Tm":300,
                        "m":0}
        
        self.param_info["init_guess"] = initial_guess
        p = Parameters(self.param_info)
        p.apply_unit_conversions(self.param_info)
        
        init_dN = 1e10 * np.ones(self.g.nx) * self.unit_conversions["n0"]
        N = init_dN + p.n0
        P = init_dN + p.p0
        E_f = E_field(N, P, p, self.g.dx)

        init_condition = np.concatenate([N, P, E_f], axis=None)
        args = (self.g,p)
        sol = solve_ivp(dydt, [self.g.start_time,self.g.time], init_condition, 
                        args=args, t_eval=self.g.tSteps, method='LSODA', 
                        max_step=self.g.hmax)
        data = sol.y.T
        
        s = self.s
        s.N, s.P, E_f = np.split(data, [self.g.nx, 2*self.g.nx], axis=1)
        tau = initial_guess["tauN"] + initial_guess["tauP"]
        np.testing.assert_almost_equal(s.N[:,0], init_dN[0] * np.exp(-self.g.tSteps / tau))
        np.testing.assert_almost_equal(s.P[:,0], init_dN[0] * np.exp(-self.g.tSteps / tau))
        np.testing.assert_almost_equal(E_f, np.zeros_like(E_f))
        ################
        
    def test_solver_LI_rad(self):
        ## SRH only, low inj - effective lifetime = (kp0)^-1 ##
        initial_guess ={"n0":0, 
                        "p0":1, 
                        "mu_n":0, 
                        "mu_p":0, 
                        "ks":1e-11, 
                        "Sf":0, 
                        "Sb":0, 
                        "Cn":0,
                        "Cp":0,
                        "tauN":1, 
                        "tauP":1, 
                        "eps":10, 
                        "Tm":300,
                        "m":0}
        
        self.param_info["init_guess"] = initial_guess
        p = Parameters(self.param_info)
        p.apply_unit_conversions(self.param_info)
        
        init_dN = 1e10 * np.ones(self.g.nx) * self.unit_conversions["n0"]
        N = init_dN + p.n0
        P = init_dN + p.p0
        E_f = E_field(N, P, p, self.g.dx)

        init_condition = np.concatenate([N, P, E_f], axis=None)
        args = (self.g,p)
        sol = solve_ivp(dydt, [self.g.start_time,self.g.time], init_condition, 
                        args=args, t_eval=self.g.tSteps, method='LSODA', 
                        max_step=self.g.hmax)
        data = sol.y.T
        
        s = self.s
        s.N, s.P, E_f = np.split(data, [self.g.nx, 2*self.g.nx], axis=1)
        tau = (initial_guess["p0"] * self.unit_conversions["p0"] *
               initial_guess["ks"] * self.unit_conversions["ks"]) ** -1
        np.testing.assert_almost_equal(s.N[:,0], init_dN[0] * np.exp(-self.g.tSteps / tau))
        np.testing.assert_almost_equal(s.P[:,0], init_dN[0] * np.exp(-self.g.tSteps / tau))
        np.testing.assert_almost_equal(E_f, np.zeros_like(E_f))
        ################
        
    def test_solver_LI_auger(self):
        ## SRH only, low inj - effective lifetime = (Cp*p0**2)^-1 ##
        initial_guess ={"n0":0, 
                        "p0":10, 
                        "mu_n":0, 
                        "mu_p":0, 
                        "ks":0, 
                        "Sf":0, 
                        "Sb":0, 
                        "Cn":0,
                        "Cp":1e-29,
                        "tauN":1, 
                        "tauP":1, 
                        "eps":10, 
                        "Tm":300,
                        "m":0}
        
        self.param_info["init_guess"] = initial_guess
        p = Parameters(self.param_info)
        p.apply_unit_conversions(self.param_info)
        
        init_dN = 1e10 * np.ones(self.g.nx) * self.unit_conversions["n0"]
        N = init_dN + p.n0
        P = init_dN + p.p0
        E_f = E_field(N, P, p, self.g.dx)

        init_condition = np.concatenate([N, P, E_f], axis=None)
        args = (self.g,p)
        sol = solve_ivp(dydt, [self.g.start_time,self.g.time], init_condition, 
                        args=args, t_eval=self.g.tSteps, method='LSODA', 
                        max_step=self.g.hmax)
        data = sol.y.T
        
        s = self.s
        s.N, s.P, E_f = np.split(data, [self.g.nx, 2*self.g.nx], axis=1)
        tau = (initial_guess["p0"] ** 2 * self.unit_conversions["p0"] ** 2 *
               initial_guess["Cp"] * self.unit_conversions["Cp"]) ** -1
        np.testing.assert_almost_equal(s.N[:,0], init_dN[0] * np.exp(-self.g.tSteps / tau))
        np.testing.assert_almost_equal(s.P[:,0], init_dN[0] * np.exp(-self.g.tSteps / tau))
        np.testing.assert_almost_equal(E_f, np.zeros_like(E_f))
        ################