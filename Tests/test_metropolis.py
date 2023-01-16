import unittest
import numpy as np
import logging
import sys

from metropolis import all_signal_handler
from metropolis import E_field, model, select_next_params
from metropolis import do_simulation, roll_acceptance, unpack_simpar
from metropolis import detect_sim_fail, detect_sim_depleted, almost_equal
from metropolis import check_approved_param, anneal
from metropolis import run_iteration, one_sim_likelihood
from sim_utils import Parameters, Grid, Covariance
from scipy.integrate import trapz
eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
q = 1.0 # [e]
q_C = 1.602e-19 # [C]
kB = 8.61773e-5  # [eV / K]

MIN_HMAX = 0.01

class TestUtils(unittest.TestCase):
    
    def setUp(self):
        self.logger = logging.getLogger()
        pass

    def test_all_signal(self):
        f = lambda x: x
        all_signal_handler(f)
        
    def test_E_field(self):
        # Test 1D
        vals = {'n0':0,
                'p0':0,
                'eps':1}
        param_info = {"names":["n0", "p0", "eps"],
                      "active":{"n0":1, "p0":1, "eps":1},
                      "init_guess":vals}
        
        pa = Parameters(param_info)
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
        param_info["init_guess"] = vals
        pa = Parameters(param_info)
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
                            "ks":((1e7) ** 3) / (1e9), "Sf":1e-2, "Sb":1e-2}
        
        param_info = {"names":["n0", "p0", "mu_n", "mu_p", "ks", "tauN", "tauP",
                               "Cn", "Cp","Sf", "Sb", "eps", "Tm"],
                      "active":{"n0":0, "p0":1, 
                                "mu_n":0, "mu_p":0, 
                                "Cn":0, "Cp":0,
                                "ks":1, "Sf":1, "Sb":1,
                                "tauN":1,"tauP":1, "eps":0,
                                "Tm":0},
                      "unit_conversions":unit_conversions}
        vals = {'n0':0,
                'p0':0,
                'mu_n':0,
                'mu_p':0,
                "ks":1e-11,
                "Cn":0, "Cp":0,
                'tauN':1e99,
                'tauP':1e99,
                'Sf':0,
                'Sb':0,
                "Tm":300,
                'eps':1}
        
        param_info["init_guess"] = vals
        pa = Parameters(param_info)
        pa.apply_unit_conversions(param_info)
        init_dN = 1e20 * np.ones(g.nx) * 1e-21 # [cm^-3] to [nm^-3]
        
        # with solveivp
        test_PL, out_dN = model(init_dN, g, pa, meas="TRPL", solver="solveivp",
                                RTOL=1e-10,ATOL=1e-14)
        rr = pa.ks * (out_dN * out_dN - pa.n0 * pa.p0)
        self.assertAlmostEqual(test_PL[-1], trapz(rr, dx=g.dx) + rr[0]*g.dx/2 + rr[-1]*g.dx/2)
        
        # with odeint
        test_PL, out_DN = model(init_dN, g, pa, meas="TRPL", solver="odeint",
                                RTOL=1e-10,ATOL=1e-14)
        rr = pa.ks * (out_dN * out_dN - pa.n0 * pa.p0)
        self.assertAlmostEqual(test_PL[-1], trapz(rr, dx=g.dx) + rr[0]*g.dx/2 + rr[-1]*g.dx/2, places=6)
        
        # try a trts
        
        vals = {'n0':0,
                'p0':0,
                'mu_n':10,
                'mu_p':10,
                "ks":1e-11,
                "Cn":0, "Cp":0,
                'tauN':1e99,
                'tauP':1e99,
                'Sf':0,
                'Sb':0,
                "Tm":300,
                'eps':1}
        param_info["init_guess"] = vals
        pa = Parameters(param_info)
        pa.apply_unit_conversions(param_info)
        
        test_TRTS, out_dN = model(init_dN, g, pa, meas="TRTS", solver="solveivp")
        trts = pa.mu_n * out_dN + pa.mu_p * out_dN
        self.assertAlmostEqual(test_TRTS[-1], trapz(trts, dx=g.dx) + trts[0]*g.dx/2 + trts[-1]*g.dx/2)
        
        # try an undefined measurement
        with self.assertRaises(NotImplementedError):
            model(init_dN, g, pa, meas="something else")
            
        # try an undefined solver
        with self.assertRaises(NotImplementedError):
            model(init_dN, g, pa, meas="TRPL", solver="somethign else")
        
        return

    
    def test_approve_param(self):
        info = {'names':['tauP', 'tauN', 'somethingelse'],
                 'unit_conversions':{'tauP':1, 'tauN':1, 'somethingelse':1},
                 'do_log':{'tauP':1, 'tauN':1, 'somethingelse':1}}
        # taun, taup must be within 2 OM
        # Accepts new_p as log10
        # [n0, p0, mu_n, mu_p, ks, sf, sb, taun, taup, eps, m]
        new_p = np.log10([511, 511e2, 1])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)
        new_p = np.log10([511, 511e2+1,  1])
        self.assertTrue("tn_tp_close" in check_approved_param(new_p, info))
        
        # tn, tp size limit
        new_p = np.log10([0.11, 0.11, 1])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)
        new_p = np.log10([0.1, 0.11, 1])
        self.assertTrue("tp_size" in check_approved_param(new_p, info))
        new_p = np.log10([0.11, 0.1, 1])
        self.assertTrue("tn_size" in check_approved_param(new_p, info))
        
        # These should still work if p is not logscaled
        info = {'names':['tauP', 'tauN', 'somethingelse'],
                 'unit_conversions':{'tauP':1, 'tauN':1, 'somethingelse':1},
                 'do_log':{'tauP':0, 'tauN':0, 'somethingelse':1}}
        new_p = np.array([511, 511e2, 1])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)
        new_p = np.array([511, 511e2+1,  1])
        self.assertTrue("tn_tp_close" in check_approved_param(new_p, info))
        
        new_p = np.array([0.11, 0.11, 1])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)
        new_p = np.array([0.1, 0.11, 1])
        self.assertTrue("tp_size" in check_approved_param(new_p, info))
        new_p = np.array([0.11, 0.1, 1])
        self.assertTrue("tn_size" in check_approved_param(new_p, info))
        
        # These should also still work if new_p's unit system is different
        info = {'names':['tauP', 'tauN', 'somethingelse'],
                 'unit_conversions':{'tauP':0.1, 'tauN':0.01, 'somethingelse':0.1},
                 'do_log':{'tauP':0, 'tauN':0, 'somethingelse':1}}
        new_p = np.array([511 * 0.1, 511e2 * 0.01, 1 * 0.1])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)
        new_p = np.array([511 * 0.1, (511e2+1) * 0.01,  1 * 0.1])
        self.assertTrue("tn_tp_close" in check_approved_param(new_p, info))
        
        new_p = np.array([0.11 * 0.1, 0.11 * 0.01, 1 * 0.1])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)
        new_p = np.array([0.09 * 0.1, 0.11 * 0.01, 1 * 0.1])
        self.assertTrue("tp_size" in check_approved_param(new_p, info))
        new_p = np.array([0.11 * 0.1, 0.09 * 0.01, 1 * 0.1])
        self.assertTrue("tn_size" in check_approved_param(new_p, info))
        
        # Check mu_n, mu_p, Sf, and Sb size limits
        info = {"names":["mu_n", "mu_p", "Sf", "Sb"],
                'do_log':{"mu_n":1, "mu_p":1, "Sf":1, "Sb":1}}
        new_p = np.log10([1e6-1, 1e6-1, 1e7-1, 1e7-1])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)
        
        new_p = np.log10([1e6, 1e6-1, 1e7-1, 1e7-1])
        self.assertTrue("mu_n_size" in check_approved_param(new_p, info))
        new_p = np.log10([1e6-1, 1e6, 1e7-1, 1e7-1])
        self.assertTrue("mu_p_size" in check_approved_param(new_p, info))
        new_p = np.log10([1e6-1, 1e6-1, 1e7, 1e7-1])
        self.assertTrue("sf_size" in check_approved_param(new_p, info))
        new_p = np.log10([1e6-1, 1e6-1, 1e7-1, 1e7])
        self.assertTrue("sb_size" in check_approved_param(new_p, info))
        
        # These should still work if p is not logscaled
        info = {"names":["mu_n", "mu_p", "Sf", "Sb"],
                'do_log':{"mu_n":0, "mu_p":0, "Sf":0, "Sb":0}}
        new_p = np.array([1e6-1, 1e6-1, 1e7-1, 1e7-1])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)
        
        new_p = np.array([1e6, 1e6-1, 1e7-1, 1e7-1])
        self.assertTrue("mu_n_size" in check_approved_param(new_p, info))
        new_p = np.array([1e6-1, 1e6, 1e7-1, 1e7-1])
        self.assertTrue("mu_p_size" in check_approved_param(new_p, info))
        new_p = np.array([1e6-1, 1e6-1, 1e7, 1e7-1])
        self.assertTrue("sf_size" in check_approved_param(new_p, info))
        new_p = np.array([1e6-1, 1e6-1, 1e7-1, 1e7])
        self.assertTrue("sb_size" in check_approved_param(new_p, info))
        
        # Check ks, Cn, Cp size limits
        info = {"names":["ks", "Cn", "Cp"],
                "do_log":{"ks":1, "Cn":1, "Cp":1}}
        new_p = np.log10([1e-7*0.9, 1e-21*0.9, 1e-21*0.9])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)
        
        new_p = np.log10([1e-7, 1e-21*0.9, 1e-21*0.9])
        self.assertTrue("ks_size" in check_approved_param(new_p, info))
        new_p = np.log10([1e-7*0.9, 1e-21, 1e-21*0.9])
        self.assertTrue("cn_size" in check_approved_param(new_p, info))
        new_p = np.log10([1e-7*0.9, 1e-21*0.9, 1e-21])
        self.assertTrue("cp_size" in check_approved_param(new_p, info))
        
        # Should work without log
        info = {"names":["ks", "Cn", "Cp"],
                "do_log":{"ks":0, "Cn":0, "Cp":0}}
        new_p = np.array([1e-7*0.9, 1e-21*0.9, 1e-21*0.9])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)
        
        new_p = np.array([1e-7, 1e-21*0.9, 1e-21*0.9])
        self.assertTrue("ks_size" in check_approved_param(new_p, info))
        new_p = np.array([1e-7*0.9, 1e-21, 1e-21*0.9])
        self.assertTrue("cn_size" in check_approved_param(new_p, info))
        new_p = np.array([1e-7*0.9, 1e-21*0.9, 1e-21])
        self.assertTrue("cp_size" in check_approved_param(new_p, info))
        
        # Check p0, which has a size limit and must also be larger than n0
        info = {"names":["n0", "p0"],
                "do_log":{"n0":1, "p0":1}}
        new_p = np.log10([1e19 * 0.8, 1e19 * 0.9])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)
        
        new_p = np.log10([1e19 * 0.8, 1e19])
        self.assertTrue("p0_size" in check_approved_param(new_p, info))
        new_p = np.log10([1e19, 1e19 * 0.9])
        self.assertTrue("p0_greater" in check_approved_param(new_p, info))
        
        # Should work without log
        info = {"names":["n0", "p0"],
                "do_log":{"n0":0, "p0":0}}
        new_p = np.array([1e19 * 0.8, 1e19 * 0.9])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)
        
        new_p = np.array([1e19 * 0.8, 1e19]) # p0 too large
        self.assertTrue("p0_size" in check_approved_param(new_p, info))
        new_p = np.array([1e19, 1e19 * 0.9]) # p0 smaller than n0
        self.assertTrue("p0_greater" in check_approved_param(new_p, info))
        
        info_without_taus = {'names':['tauQ', 'somethingelse'],
                             "do_log":{'tauQ':1, 'somethingelse':1}}
        # No failures if criteria do not cover params
        new_p = np.log10([1,1e10])
        self.assertTrue(len(check_approved_param(new_p, info_without_taus)) == 0)
        
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
                      "names":param_names,
                      "init_guess":initial_guesses}
        
        
        pa = Parameters(param_info)
        means = Parameters(param_info)
        variances = Covariance(param_info)
        variances.set_variance('a', 10)
        variances.set_variance('b', 0.1)
        variances.set_variance('c', 0)
        variances.set_variance('d', 1)
        
        # Try Gaussian selection
        select_next_params(pa, means, variances, param_info, trial_function="gauss", logger=self.logger)
        
        self.assertEqual(pa.a, initial_guesses['a']) #Inactive and shouldn't change
        self.assertAlmostEqual(pa.b, 68.07339753)
        self.assertEqual(pa.c, initial_guesses['c']) # Shouldn't change because variance is zero
        self.assertAlmostEqual(pa.d, 9.38824359)
        
        # Try invalid covaraince: pa should fall back to pre-existing values
        variances.set_variance('d', -1)

        with self.assertLogs() as captured:
            select_next_params(pa, means, variances, param_info, trial_function="gauss", logger=self.logger)
                
        self.assertEqual(len(captured.records), 1) # One error about the multivariate norm failing
        
        self.assertEqual(pa.a, initial_guesses['a'])
        self.assertAlmostEqual(pa.b, initial_guesses['b'])
        self.assertEqual(pa.c, initial_guesses['c'])
        self.assertAlmostEqual(pa.d, initial_guesses['d'])
        
        
        # Try box selection
        select_next_params(pa, means, variances, param_info, trial_function="box", logger=self.logger)

        self.assertEqual(pa.a, initial_guesses['a']) #Inactive and shouldn't change
        self.assertEqual(pa.c, initial_guesses['c'])
        num_tests = 100
        for t in range(num_tests):
            select_next_params(pa, means, variances, param_info, trial_function="box", logger=self.logger)
            self.assertTrue(np.abs(np.log10(pa.b) - np.log10(initial_guesses['b'])) <= 0.1, 
                            msg="Uniform step #{} failed: {} from mean {} and width 0.1".format(t, pa.b, initial_guesses['b']))
            self.assertTrue(np.abs(pa.d-initial_guesses['d']) <= 1,
                            msg="Uniform step #{} failed: {} from mean {} and width 1".format(t, pa.d, initial_guesses['d']))
        
        return
    
    def test_do_simulation(self):
        # Just verify this realistic simulation converges
        unit_conversions = {"n0":((1e-7) ** 3), "p0":((1e-7) ** 3), 
                            "mu_n":((1e7) ** 2) / (1e9), "mu_p":((1e7) ** 2) / (1e9), 
                            "ks":((1e7) ** 3) / (1e9), "Sf":1e-2, "Sb":1e-2}
        
        param_info = {"names":["n0", "p0", "mu_n", "mu_p", "ks", "Cn", "Cp", "tauN", "tauP",
                               "Sf", "Sb", "eps", "Tm"],
                      "active":{"n0":0, "p0":1, 
                                "mu_n":0, "mu_p":0, 
                                "ks":1, "Sf":1, "Sb":1,
                                "Cn":0, "Cp":0, 
                                "tauN":1,"tauP":1, "eps":0, "Tm":0},
                      "unit_conversions":unit_conversions}
        vals = {'n0':1e8,
                'p0':3e15,
                'mu_n':20,
                'mu_p':100,
                "ks":1e-11,
                "Cn":0, "Cp":0,
                'tauN':120,
                'tauP':200,
                'Sf':5,
                'Sb':20,
                'eps':1,
                "Tm":300}
        
        param_info["init_guess"] = vals
        pa = Parameters(param_info)
        pa.apply_unit_conversions(param_info)
        
        thickness = 1000
        nx = 100
        times = np.linspace(0, 100, 1000)
        
        iniPar = np.logspace(19, 14, nx) * 1e-21
        do_simulation(pa, thickness, nx, iniPar, times, hmax=4)
        return
    
    def test_sim_fail(self):
        # Suppose our sim aborted halfway through...
        reference = np.ones(100)
        sim_output = np.zeros(50)
        sim_output, fail = detect_sim_fail(sim_output, reference)
        np.testing.assert_equal(sim_output, [0]*50 + [sys.float_info.min]*50)
        
    def test_sim_depleted(self):
        sim_output = np.ones(10) * -1
        sim_output, fail = detect_sim_depleted(sim_output)
        np.testing.assert_equal(sim_output, [1 + sys.float_info.min] * 10)
        
    def test_almost_equal(self):
        threshold = 1e-7
        
        # One element just too large
        a = np.array([1.0,1.0])
        b = np.array([1.0,1.0+threshold])
        self.assertFalse(almost_equal(b, a, threshold=threshold))
        
        # All elements just close enough
        b = np.array([1.0, 1.0+0.999*threshold])
        self.assertTrue(almost_equal(b, a, threshold=threshold))
        
        wrong_shape = np.array([1.0,1.0,1.0])
        self.assertFalse(almost_equal(a, wrong_shape))
    
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
        Length  = [2000]                            # Length (nm)
        L   = 2 ** 7                                # Spatial points
        meas_type = ["TRPL"]                          # measurement type

        simPar = {"lengths":Length, 
                  "nx":L, 
                  "meas_types":meas_type,
                  "num_meas":1}
        
        thickness, nx, mtype = unpack_simpar(simPar, 0)
        self.assertEqual(Length[0], thickness)
        self.assertEqual(L, nx)
        self.assertEqual(meas_type[0], mtype)
        
        
        Length = np.array([311,2000,311,2000, 311, 2000])
        meas_type = ["TRPL", "TRTS", "TRPL", "TRPL", "TRTS", "TRPL"]
        simPar = {"lengths":Length, 
                  "nx":L, 
                  "meas_types":meas_type,
                  "num_meas":6}
        
        thickness, nx, mtype = unpack_simpar(simPar, 2)
        self.assertEqual(Length[2], thickness)
        self.assertEqual(L, nx)
        self.assertEqual(meas_type[2], mtype)
        return
        
    def test_anneal(self):
        anneal_mode = None # T = T_2
        anneal_params = [0, 0, 1] #T_2
        t = 9999
        self.assertTrue(anneal(t, anneal_mode, anneal_params), anneal_params[0])
        
        anneal_mode = "exp" # T = T_0 * exp(-t/T_1) + T_2
        anneal_params = [10,1, 0]
        t = 1
        self.assertTrue(anneal(t, anneal_mode, anneal_params), anneal_params / np.exp(1))
        
        anneal_mode = "log" # T = (T_0 ln(2)) / (ln(2 + (t / T_1))) + T_2
        t = 23523
        anneal_params = [10, t / (np.exp(1) - 2), 0]
        self.assertTrue(anneal(t, anneal_mode, anneal_params), anneal_params[0] * np.log(2))
        
        anneal_mode = "not a mode"
        with self.assertRaises(ValueError):
            anneal(t, anneal_mode, anneal_params)
        
    def test_run_iter(self):
        # Will basically need to set up a full simulation for this
        np.random.seed(42)
        Length  = [2000, 2000]                            # Length (nm)
        L   = 2 ** 7                                # Spatial point
        mtype = ["TRPL", "TRPL"]
        simPar = {"lengths":Length, 
                  "nx":L, 
                  "meas_types":mtype,
                  "num_meas":2}
        
        iniPar = [1e15 * np.ones(L) * 1e-21, 1e16 * np.ones(L) * 1e-21]

        param_names = ["n0", "p0", "mu_n", "mu_p", "ks", "Cn", "Cp", "Tm",
                       "Sf", "Sb", "tauN", "tauP", "eps", "m"]
        unit_conversions = {"n0":((1e-7) ** 3), "p0":((1e-7) ** 3), 
                            "mu_n":((1e7) ** 2) / (1e9), "mu_p":((1e7) ** 2) / (1e9), 
                            "ks":((1e7) ** 3) / (1e9), "Sf":1e-2, "Sb":1e-2}
        
        # Iterations should proceed independent of which params are actively iterated,
        # as all params are presumably needed to complete the simulation
        param_info = {"names":param_names,
                      "unit_conversions":unit_conversions, 
                      "active":{name:0 for name in param_names}}
        initial_guess = {"n0":0, 
                         "p0":0, 
                         "mu_n":0, 
                         "mu_p":0, 
                         "ks":1e-11, 
                         "Sf":0, 
                         "Sb":0, 
                         "Cn":0,
                         "Cp":0,
                         "Tm":300,
                         "tauN":1e99, 
                         "tauP":1e99, 
                         "eps":10, 
                         "m":1}
        
        param_info["init_guess"] = initial_guess
        
        sim_flags = {"anneal_mode": None, # None, "exp", "log"
                     "anneal_params": [0, 1/2500*100, 1], 
                     "hmax":4, "rtol":1e-5, "atol":1e-8,
                     "measurement":"TRPL",
                     "solver":"solveivp",}
        
        p = Parameters(param_info)
        p.apply_unit_conversions(param_info)
        p2 = Parameters(param_info)
        p2.apply_unit_conversions(param_info)
        
        nt = 1000
        running_hmax = [4] * len(iniPar)
        times = [np.linspace(0, 100, nt+1), np.linspace(0, 100, nt+1)]
        vals = [np.zeros(nt+1), np.zeros(nt+1)]
        uncs = [np.ones(nt+1) * 1e-99, np.ones(nt+1) * 1e-99]
        accepted = run_iteration(p, simPar, iniPar, times, vals, uncs, 
                                 running_hmax, sim_flags, verbose=True, 
                                 logger=self.logger, prev_p=None)
        
        # First iter; auto-accept
        np.testing.assert_almost_equal(p.likelihood, [-59340.105083, -32560.139058], decimal=0) #rtol=1e-5
        self.assertTrue(accepted)
        
        # Second iter same as the first; auto-accept with likelihood ratio exactly 1
        accepted = run_iteration(p2, simPar, iniPar, times, vals, uncs, 
                                 running_hmax, sim_flags, verbose=True, 
                                 logger=self.logger, prev_p=p)
        self.assertTrue(accepted)
        # Accept should overwrite p2 (new) into p (old)
        np.testing.assert_equal(p.likelihood, p2.likelihood)
        
    def test_one_sim_ll_errata(self):
        # TODO: The next time odeint fails to do a simulation, upload it into this
        # test case
        return
        # np.random.seed(42)
        # Length  = 2000                            # Length (nm)
        # L   = 2 ** 7                                # Spatial points
        # plT = 1                                  # Set PL interval (dt)
        # pT  = (0,1,3,10,30,100)                   # Set plot intervals (%)
        # tol = 7                                   # Convergence tolerance
        # MAX = 10000                                  # Max iterations
        
        # simPar = [Length, -1, L, -1, plT, pT, tol, MAX]
        
        # iniPar = [np.logspace(20,1,L) * 1e-21, 1e16 * np.ones(L) * 1e-21]

        # param_names = ["n0", "p0", "mu_n", "mu_p", "ks", "Cn", "Cp", "Tm",
        #                "Sf", "Sb", "tauN", "tauP", "eps", "m"]
        # unit_conversions = {"n0":((1e-7) ** 3), "p0":((1e-7) ** 3), 
        #                     "mu_n":((1e7) ** 2) / (1e9), "mu_p":((1e7) ** 2) / (1e9), 
        #                     "ks":((1e7) ** 3) / (1e9), "Sf":1e-2, "Sb":1e-2}
        
        # # Iterations should proceed independent of which params are actively iterated,
        # # as all params are presumably needed to complete the simulation
        # param_info = {"names":param_names,
        #               "unit_conversions":unit_conversions, 
        #               "active":{name:0 for name in param_names}}
        # initial_guess = {"n0":1e8, 
        #                  "p0":1e12, 
        #                  "mu_n":2, 
        #                  "mu_p":2, 
        #                  "ks":1e-11, 
        #                  "Sf":1000, 
        #                  "Sb":1e4, 
        #                  "Cn":0,
        #                  "Cp":0,
        #                  "Tm":300,
        #                  "tauN":10, 
        #                  "tauP":10, 
        #                  "eps":10, 
        #                  "m":0}
        
        # sim_flags = {"hmax":MIN_HMAX * 4, "rtol":1e-10, "atol":1e-10,
        #              "measurement":"TRPL",
        #              "solver":"odeint"}
        
        # nt = 100
        # i = 0
        # times = [np.linspace(0, 10, nt+1), np.linspace(0, 10, nt+1)]
        # vals = [np.zeros(nt+1), np.zeros(nt+1)]
        
        # running_hmax = [sim_flags["hmax"]] * len(iniPar)
        # p = Parameters(param_info, initial_guess)
        # p.apply_unit_conversions(param_info)
        
        # with self.assertLogs() as captured:
        #     one_sim_likelihood(p, simPar, running_hmax, sim_flags, self.logger,
        #                        (i, iniPar[i], times[i], vals[i]))
            
        # # We can force a simulation failure by using odeint + rapid recombination
        # self.assertEqual(len(captured.records), 8) 
        # self.assertEqual(captured.records[1].getMessage(), 
        #                  f"{i}: Carriers depleted!")
        # self.assertEqual(captured.records[2].getMessage(),
        #                  f"{i}: Retrying hmax={MIN_HMAX * 2}") 

        
if __name__ == "__main__":
    t = TestUtils()
    t.setUp()
    t.test_run_iter()
