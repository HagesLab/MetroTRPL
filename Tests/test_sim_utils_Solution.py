import unittest
import numpy as np
import sys
sys.path.append("..")
from sim_utils import Parameters, Solution, Grid

class TestUtils(unittest.TestCase):
           
    def test_Solution(self):
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
        np.testing.assert_equal(expected_PL[:,0] * num_nodes, testS.PL)
        
        # Some more PL
        initial_guesses = {"ks":2, 'n0':1, 'p0':1}
        param_info["init_guess"] = initial_guesses
        testS.P = np.ones((num_tsteps, num_nodes)) * 10
        testP = Parameters(param_info)
        testS.calculate_PL(testG, testP)
        expected_PL = initial_guesses["ks"] * (testS.N * testS.P - initial_guesses["n0"]*initial_guesses["p0"])
        np.testing.assert_equal(expected_PL[:,0] * num_nodes, testS.PL)
        
        # Some TRTS
        param_info = {'names':["mu_n", "mu_p", 'n0', 'p0'],
                      'active':{"mu_n":1, "mu_p":1, 'n0':1, 'p0':1}}
        initial_guesses = {"mu_n":1, "mu_p":1, 'n0':1, 'p0':1}
        param_info["init_guess"] = initial_guesses
        testS.P = np.ones((num_tsteps, num_nodes)) * 10
        testP = Parameters(param_info)
        testS.calculate_TRTS(testG, testP)
        expected_TRTS = (initial_guesses["mu_n"] * (testS.N - initial_guesses["n0"]) +
                       initial_guesses["mu_p"] * (testS.P - initial_guesses["p0"]))
        np.testing.assert_equal(expected_TRTS[:,0] * num_nodes, testS.trts)
        
        return
    