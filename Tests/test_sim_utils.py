import unittest
import numpy as np
import sys
sys.path.append("..")
from sim_utils import Parameters, Solution, Grid, Covariance, MetroState

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
        
        testP = Parameters(param_info, initial_guesses)
        testS.calculate_PL(testG, testP)
        np.testing.assert_equal(np.zeros(num_tsteps), testS.PL)
        
        # Some PL
        initial_guesses = {"ks":1, 'n0':0, 'p0':0}
        testS.P = np.ones((num_tsteps, num_nodes)) * 10
        testP = Parameters(param_info, initial_guesses)
        testS.calculate_PL(testG, testP)
        np.testing.assert_equal(np.ones(num_tsteps) * 10 * num_nodes, testS.PL)
        
        # Some more PL
        initial_guesses = {"ks":2, 'n0':1, 'p0':1}
        testS.P = np.ones((num_tsteps, num_nodes)) * 10
        testP = Parameters(param_info, initial_guesses)
        testS.calculate_PL(testG, testP)
        np.testing.assert_equal(np.ones(num_tsteps) * 18 * num_nodes, testS.PL)
        
        return
    
    def test_MetroState(self):
        # Will only look for these
        dummy_names = ['mu_n', 'c', 'b', 'a']

        # Should (but not required) contain one for each name
        dummy_unitconversions = {'a':1, 'c':10, 'mu_n':0.25}
        dummy_do_log = {'a':True, 'b':0, 'c':0, 'mu_n':1, 'mu_p':True}
        dummy_active = {'a':1, 'b':1, 'c':1, 'mu_n':1}
        
        dummy_param_info = {'names':dummy_names,
                            'unit_conversions':dummy_unitconversions,
                            'do_log':dummy_do_log,
                            'active':dummy_active}
        
        dummy_initial_guesses = {'a':1, 'b':2, 'c':3, 'mu_n':4}
        dummy_initial_variance = {'a':2, 'b':1, 'c':2, 'mu_n':1}
        num_iters = 100
        ms = MetroState(dummy_param_info, dummy_initial_guesses, dummy_initial_variance,
                        num_iters)
        
        # The functionality for each of these has already been covered
        self.assertIsInstance(ms.p, Parameters)
        self.assertIsInstance(ms.prev_p, Parameters)
        self.assertIsInstance(ms.means, Parameters)
        self.assertIsInstance(ms.variances, Covariance)
        
        # Covariance scales; Should be 1D and match dummy_inital_variance
        # Note that this follows the order of dummy_names
        np.testing.assert_equal(ms.variances.little_sigma, [1,2,1,2])
        
        # Descaled Covariance matrix
        np.testing.assert_equal(ms.variances.big_sigma, np.eye(4))
        
        # Testing with samey initial variance
        samey_variance = 26
        ms = MetroState(dummy_param_info, dummy_initial_guesses, samey_variance,
                        num_iters)
        
        np.testing.assert_equal(ms.variances.little_sigma, [26,26,26,26])
        
        # Descaled Covariance matrix
        np.testing.assert_equal(ms.variances.big_sigma, np.eye(4))
        
if __name__ == "__main__":
    t = TestUtils()
    t.test_History()