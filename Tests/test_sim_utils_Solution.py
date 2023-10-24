import unittest
import numpy as np
import sys
sys.path.append("..")
from sim_utils import Parameters, Solution, Grid
from sim_utils import check_threshold, calculate_PL, calculate_TRTS, integrate_1D, q_C

class TestUtils(unittest.TestCase):

    def test_integrate_1D(self):
        """
        Simple test integral of y=x from x=0 to x=1, which should return close to 0.5.
        Note that x is assumed to be a list of nodes with width dx - so the x
        values do not reach all the way to either 0 or 1 and must extrapolate beyond that.
        For y = x, the errors at each bound cancel each other.
        """
        dx = 0.01
        L = 1
        x = np.arange(dx / 2, L, dx)
        y = x
        expected_integral = 0.5
        np.testing.assert_almost_equal(integrate_1D(dx, y), expected_integral)
           
    def test_Solution_2D(self):
        """With 2D N and P arrays, which should return PL/TRTS arrays integrated over the space dimension"""
        q_C = 1.602e-19  # [C per carrier]
        testS = Solution()
        testG = Grid()
        testG.dx = 1
        
        num_nodes = 128
        num_tsteps = 10
        testS.N = np.ones((num_tsteps, num_nodes))
        testS.P = np.ones((num_tsteps, num_nodes))
        
        # No PL
        param_info = {'names':["ks", 'n0', 'p0'],
                      'active':{"ks":1, 'n0':1, 'p0':1}}
        initial_guesses = {"ks":0, 'n0':0, 'p0':0}
        
        param_info["init_guess"] = initial_guesses
        testP = Parameters(param_info)
        testS.calculate_PL(testG, testP)
        np.testing.assert_equal(np.zeros(num_tsteps), testS.PL)
        
        # Some PL
        initial_guesses = {"ks":1, 'n0':0, 'p0':0}
        param_info["init_guess"] = initial_guesses
        testS.P = np.ones((num_tsteps, num_nodes)) * 10
        testP = Parameters(param_info)
        testS.calculate_PL(testG, testP)
        expected_PL = initial_guesses["ks"] * (testS.N * testS.P - initial_guesses["n0"]*initial_guesses["p0"])
        # Like integrating, since dx=1
        np.testing.assert_almost_equal(expected_PL[:,0] * num_nodes, testS.PL)
        
        # Some more PL
        initial_guesses = {"ks":2, 'n0':1, 'p0':1}
        param_info["init_guess"] = initial_guesses
        testS.P = np.ones((num_tsteps, num_nodes)) * 10
        testP = Parameters(param_info)
        testS.calculate_PL(testG, testP)
        expected_PL = initial_guesses["ks"] * (testS.N * testS.P - initial_guesses["n0"]*initial_guesses["p0"])
        np.testing.assert_almost_equal(expected_PL[:,0] * num_nodes, testS.PL)
        
        # Some TRTS
        param_info = {'names':["mu_n", "mu_p", 'n0', 'p0'],
                      'active':{"mu_n":1, "mu_p":1, 'n0':1, 'p0':1}}
        initial_guesses = {"mu_n":1, "mu_p":1, 'n0':1, 'p0':1}
        param_info["init_guess"] = initial_guesses
        testS.P = np.ones((num_tsteps, num_nodes)) * 10
        testP = Parameters(param_info)
        testS.calculate_TRTS(testG, testP)
        expected_TRTS = q_C * (initial_guesses["mu_n"] * (testS.N - initial_guesses["n0"]) +
                       initial_guesses["mu_p"] * (testS.P - initial_guesses["p0"]))
        np.testing.assert_almost_equal(expected_TRTS[:,0] * num_nodes, testS.trts)
        
        return
    
    def test_solution_1D(self):
        """With 1D N and P, which should return single PL or TRTS values"""
        q_C = 1.602e-19  # [C per carrier]
        testS = Solution()
        testG = Grid()
        testG.dx = 1
        
        num_nodes = 128
        testS.N = np.ones(num_nodes)
        testS.P = np.ones(num_nodes)

        param_info = {'names':["ks", 'n0', 'p0'],
                      'active':{"ks":1, 'n0':1, 'p0':1}}

        # Some PL
        initial_guesses = {"ks":2, 'n0':1, 'p0':1}
        param_info["init_guess"] = initial_guesses
        testP = Parameters(param_info)
        testS.calculate_PL(testG, testP)
        expected_PL = initial_guesses["ks"] * (testS.N * testS.P - initial_guesses["n0"]*initial_guesses["p0"])
        np.testing.assert_almost_equal(expected_PL[0] * num_nodes, testS.PL)

        # Some TRTS
        param_info = {'names':["mu_n", "mu_p", 'n0', 'p0'],
                      'active':{"mu_n":1, "mu_p":1, 'n0':1, 'p0':1}}
        initial_guesses = {"mu_n":1, "mu_p":1, 'n0':1, 'p0':1}
        param_info["init_guess"] = initial_guesses
        testS.P = np.ones(num_nodes) * 10
        testP = Parameters(param_info)
        testS.calculate_TRTS(testG, testP)
        expected_TRTS = q_C * (initial_guesses["mu_n"] * (testS.N - initial_guesses["n0"]) +
                       initial_guesses["mu_p"] * (testS.P - initial_guesses["p0"]))
        np.testing.assert_almost_equal(expected_TRTS[0] * num_nodes, testS.trts)

    def test_check_threshold_PL(self):
        num_nodes = 100
        dx = 1
        N = np.full(num_nodes, 11)
        P = np.full(num_nodes, 11)
        t = None
        y0 = np.concatenate((N, P), axis=0)
        ks = 2
        n0 = 1
        p0 = 1
        
        # y0 -> 2 * (121 - 1) * 100 = 24000
        PL0 = calculate_PL(dx, N, P, ks, n0, p0)

        # y -> 2 * (1.21 - 1) * 100 = 42
        y = y0 / 10

        min_y = PL0 * 1e-3
        # PL, still above thr - True
        self.assertTrue(check_threshold(t, y, num_nodes, dx, min_y, mode="TRPL",
                                        ks=ks, n0=n0, p0=p0))
        
        min_y = PL0 * 1e-2
        # PL, now below thr - False
        self.assertFalse(check_threshold(t, y, num_nodes, dx, min_y, mode="TRPL",
                                         ks=ks, n0=n0, p0=p0))
        
    def test_check_threshold_TRTS(self):
        num_nodes = 100
        dx = 1
        N = np.full(num_nodes, 11)
        P = np.full(num_nodes, 11)
        t = None
        y0 = np.concatenate((N, P), axis=0)
        mu_n = 1
        mu_p = 1
        n0 = 1
        p0 = 1
        
        # y0 -> q_C * ((11 - 1) + (11 - 1)) * 100 = 2000*q_C
        TRTS0 = calculate_TRTS(dx, N, P, mu_n, mu_p, n0, p0)

        # y -> q_C * ((1.1 - 1) + (1.1 - 1)) * 100 = 20*q_C
        y = y0 / 10

        min_y = TRTS0 * 1e-2
        # PL, still at thr - True
        self.assertTrue(check_threshold(t, y, num_nodes, dx, min_y, mode="TRTS",
                                        mu_n=mu_n, mu_p=mu_p, n0=n0, p0=p0))
        
        min_y = TRTS0 * 1.001e-2
        # PL, now below thr - False
        self.assertFalse(check_threshold(t, y, num_nodes, dx, min_y, mode="TRTS",
                                         mu_n=mu_n, mu_p=mu_p, n0=n0, p0=p0))
