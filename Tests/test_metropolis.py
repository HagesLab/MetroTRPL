import unittest
import numpy as np

from metropolis import E_field, model, select_next_params, update_means, update_history
from metropolis import do_simulation, roll_acceptance, unpack_simpar, convert_DA_times
from sim_utils import Parameters, Grid, History
from scipy.integrate import trapz
eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
q = 1.0 # [e]
q_C = 1.602e-19 # [C]
kB = 8.61773e-5  # [eV / K]

class TestUtils(unittest.TestCase):
    
    def test_E_field(self):
        # Test 1D
        param_info = {"names":["n0", "p0", "eps"]}
        vals = {'n0':0,
                'p0':0,
                'eps':1}
        pa = Parameters(param_info, vals)
        nx = 10
        dx = 1
        # Test N=P
        N = np.zeros(nx)
        P = np.zeros(nx)
        E = E_field(N, P, pa, dx)
        
        np.testing.assert_equal(E, np.zeros_like(E))
        
        # Test N>P
        P = np.ones(nx) * 2
        N = np.ones(nx)
        E = E_field(N, P, pa, dx)
        
        np.testing.assert_equal(E[1:], np.ones_like(E[1:]) * q_C/eps0 * np.cumsum(N))
        
        # Test N<P
        P = np.ones(nx) * -1
        E = E_field(N, P, pa, dx)
        
        np.testing.assert_equal(E[1:], -2 * np.ones_like(E[1:]) * q_C/eps0 * np.cumsum(N))
        
        # Test corner_E != 0
        corner_E = 24
        E = E_field(N, P, pa, dx, corner_E=corner_E)
        
        np.testing.assert_equal(E[1:], -2 * np.ones_like(E[1:]) * q_C/eps0 * np.cumsum(N) + corner_E)
        
        # Test 2D
        N = np.ones((nx, nx+1))
        P = np.ones((nx, nx+1)) * -1
        E = E_field(N, P, pa, dx, corner_E=corner_E)
        np.testing.assert_equal(E[:,1:], -2 * np.ones_like(E[:,1:]) * q_C/eps0 * np.cumsum(N, axis=1) + corner_E)
        
        # Test n0, p0
        vals = {'n0':1,
                'p0':1,
                'eps':1}
        pa = Parameters(param_info, vals)
        N = np.ones((nx, nx+1))
        P = np.ones((nx, nx+1))
        E = E_field(N, P, pa, dx, corner_E=corner_E)
        np.testing.assert_equal(E[1:], np.zeros_like(E[1:]) + corner_E)
        
        return
    
    def test_model(self):
        # A high-inj, rad-only sample problem
        g = Grid()
        g.nx = 100
        g.dx = 10
        g.start_time = 0
        g.time = 100
        g.nt = 1000
        g.hmax = 4
        g.tSteps = np.linspace(g.start_time, g.time, g.nt+1)
        
        unit_conversions = {"n0":((1e-7) ** 3), "p0":((1e-7) ** 3), 
                            "mu_n":((1e7) ** 2) / (1e9), "mu_p":((1e7) ** 2) / (1e9), 
                            "B":((1e7) ** 3) / (1e9), "Sf":1e-2, "Sb":1e-2}
        
        param_info = {"names":["n0", "p0", "mu_n", "mu_p", "B", "tauN", "tauP",
                               "Sf", "Sb", "eps"],
                      "unit_conversions":unit_conversions}
        vals = {'n0':0,
                'p0':0,
                'mu_n':0,
                'mu_p':0,
                'B':1e-11,
                'tauN':1e99,
                'tauP':1e99,
                'Sf':0,
                'Sb':0,
                'eps':1}
        pa = Parameters(param_info, vals)
        pa.apply_unit_conversions(param_info)
        init_dN = 1e20 * np.ones(g.nx) * 1e-21 # [cm^-3] to [nm^-3]
        
        test_PL, out_dN = model(init_dN, g, pa)
        rr = pa.B * (out_dN * out_dN - pa.n0 * pa.p0)
        self.assertAlmostEqual(test_PL[-1], trapz(rr, dx=g.dx) + rr[0]*g.dx/2 + rr[-1]*g.dx/2)
        return

    def test_select_next_params(self):
        # This function assigns a set of randomly generated values
        np.random.seed(1)
        param_names = ["a", "b", "c", "d"]

        do_log = {"a":0, "b":1,"c":0,"d":0}
        
        initial_guesses = {"a":0, 
                            "b":100, 
                            "c":0,
                            "d":10,}
        
        active_params = {"a":0, 
                         "b":1, 
                         "c":1, 
                         "d":1,}
        
        param_info = {"active":active_params,
                      "do_log":do_log,
                      "names":param_names,}
        
        
        pa = Parameters(param_info, initial_guesses)
        means = Parameters(param_info, initial_guesses)
        variances = Parameters(param_info, initial_guesses)
        variances.a = 10
        variances.b = 0.1
        variances.c = 1
        variances.d = 1
        select_next_params(pa, means, variances, param_info)
        
        self.assertEqual(pa.a, initial_guesses['a']) #Inactive and shouldn't change
        self.assertAlmostEqual(pa.b, 145.3565265)
        self.assertAlmostEqual(pa.c, 0.865407629)
        self.assertAlmostEqual(pa.d, 7.698461303)
        return
    
    def test_update_means(self):
        param_names = ["a", "b", "c", "d"]
        
        initial_guesses = {"a":0, 
                            "b":100, 
                            "c":0,
                            "d":10,}
        
        example_means = {"a":1, 
                        "b":2, 
                        "c":3,
                        "d":4,}

        param_info = {"names":param_names,}
        
        
        pa = Parameters(param_info, initial_guesses)
        means = Parameters(param_info, example_means)
        
        update_means(pa, means, param_info)
        
        for param in param_names:
            self.assertEqual(getattr(means, param), getattr(pa, param))
            
        return
    
    # Not testing print_status
    
    def test_update_history(self):
        num_iters = 10
        param_names = ["a", "b", "c", "d"]
        param_info = {"names":param_names,}
        history = History(num_iters, param_info)
        
        initial_guesses = {"a":0, 
                            "b":100, 
                            "c":0,
                            "d":10,}
        
        example_means = {"a":1, 
                        "b":2, 
                        "c":3,
                        "d":4,}
        pa = Parameters(param_info, initial_guesses)
        means = Parameters(param_info, example_means)
        
        update_history(history, 1, pa, means, param_info)
        update_history(history, 6, pa, means, param_info)
        
        for param in param_names:
            self.assertEqual(getattr(history, param)[1], initial_guesses[param])
            self.assertEqual(getattr(history, f"mean_{param}")[1], example_means[param])
            self.assertEqual(sum(getattr(history, param)), 2*initial_guesses[param])
            self.assertEqual(sum(getattr(history, f"mean_{param}")), 2*example_means[param])
        return
    
    def test_do_simulation(self):
        # Just verify this realistic simulation converges
        unit_conversions = {"n0":((1e-7) ** 3), "p0":((1e-7) ** 3), 
                            "mu_n":((1e7) ** 2) / (1e9), "mu_p":((1e7) ** 2) / (1e9), 
                            "B":((1e7) ** 3) / (1e9), "Sf":1e-2, "Sb":1e-2}
        
        param_info = {"names":["n0", "p0", "mu_n", "mu_p", "B", "tauN", "tauP",
                               "Sf", "Sb", "eps"],
                      "unit_conversions":unit_conversions}
        vals = {'n0':1e8,
                'p0':3e15,
                'mu_n':20,
                'mu_p':100,
                'B':1e-11,
                'tauN':120,
                'tauP':200,
                'Sf':5,
                'Sb':20,
                'eps':1}
        pa = Parameters(param_info, vals)
        pa.apply_unit_conversions(param_info)
        
        thickness = 1000
        nx = 100
        times = np.linspace(0, 100, 1000)
        
        iniPar = np.logspace(19, 14, nx) * 1e-21
        do_simulation(pa, thickness, nx, iniPar, times)
        return
    
    def test_roll_acceptance(self):
        np.random.seed(1)
        
        logratio = 1
        accept = np.zeros(100, dtype=bool)
        for i in range(len(accept)):
            accept[i] = roll_acceptance(logratio)

        self.assertTrue(all(accept))   

        logratio = -1
        accept = np.zeros(10000, dtype=bool)
        for i in range(len(accept)):
            accept[i] = roll_acceptance(logratio)
            
        self.assertEqual(accept.sum(), 1003)
        return

    def test_unpack_simpar(self):
        #Length = [311,2000,311,2000, 311, 2000]
        Length  = 2000                            # Length (nm)
        L   = 2 ** 7                                # Spatial points
        plT = 1                                  # Set PL interval (dt)
        pT  = (0,1,3,10,30,100)                   # Set plot intervals (%)
        tol = 7                                   # Convergence tolerance
        MAX = 10000                                  # Max iterations
        
        simPar = [Length, -1, L, -1, plT, pT, tol, MAX]
        
        thickness, nx = unpack_simpar(simPar, 99)
        self.assertEqual(Length, thickness)
        self.assertEqual(L, nx)
        
        Length = [311,2000,311,2000, 311, 2000]
        simPar = [Length, -1, L, -1, plT, pT, tol, MAX]
        thickness, nx = unpack_simpar(simPar, 2)
        self.assertEqual(Length[2], thickness)
        self.assertEqual(L, nx)
        return
    
    def test_convert_DA_times(self):
        total_len = 80000
        DA_time_subs = 10
        # DA_time_subs as int should just pass unchanged
        self.assertEqual(DA_time_subs, convert_DA_times(DA_time_subs, total_len))
        
        DA_time_subs = [10, 50]
        
        # DA_time_subs as list should take percentages
        np.testing.assert_equal([8000, 40000], convert_DA_times(DA_time_subs, total_len))
        return
