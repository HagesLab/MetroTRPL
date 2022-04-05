import unittest
import numpy as np
import sys
sys.path.append("..")
from sim_utils import Parameters, History, HistoryList, Solution, Grid, Covariance

class TestUtils(unittest.TestCase):
    
    def test_Parameters(self):
        # Will only look for these
        dummy_names = ['mu_n', 'c', 'b', 'a']
        
        # Must contain one for each name
        dummy_parameters = {'a':1, 'b':2, 'c':3, 'mu_n':4}
        
        # Should (but not required) contain one for each name
        dummy_unitconversions = {'a':1, 'c':10, 'mu_n':0.25}
        dummy_do_log = {'a':True, 'b':0, 'mu_n':1, 'mu_p':True}
        
        dummy_param_info = {'names':dummy_names,
                            'unit_conversions':dummy_unitconversions,
                            'do_log':dummy_do_log}
        
        # Test initialization
        testp = Parameters(dummy_param_info, dummy_parameters)
        
        for param in dummy_names:
            self.assertEqual(getattr(testp, param), dummy_parameters[param])
            
        # Test asarray
        arr = testp.asarray(dummy_param_info)
        expected = np.array([4,3,2,1])
        np.testing.assert_equal(arr, expected)
        
        # Test init with duplicate params
        duplicate_names = ['a', 'a']
        dup_info = {'names':duplicate_names,
                    'unit_conversions':dummy_unitconversions}
        with self.assertRaises(KeyError):
            testp_dup = Parameters(dup_info, dummy_parameters)
            
        # Test init with missing params
        bad_params = {}
        dup_info = {'names':dummy_names}
        
        with self.assertRaises(KeyError):
            testp_dup = Parameters(dup_info, bad_params)
            
        # Test unit conversion
        expected_converted_values = {'a':1, 'b':2, 'c':30, 'mu_n':1.0}
        testp.apply_unit_conversions(dummy_param_info)
        for param in dummy_names:
            self.assertEqual(getattr(testp, param), expected_converted_values[param])
            
        # Test make log
        testp = Parameters(dummy_param_info, dummy_parameters)
        testp.make_log(dummy_param_info)
        expected_logged_values = {'a':0, 'b':2, 'c':3, 'mu_n':np.log10(4)}
        for param in dummy_names:
            self.assertEqual(getattr(testp, param), expected_logged_values[param])
            
    def test_Covariance(self):
        dummy_names = ['mu_n', 'c', 'b', 'a']
        d = len(dummy_names)
        
        # Must contain one for each name
        dummy_parameters = {'a':1, 'b':2, 'c':3, 'mu_n':4}
        
        # Should (but not required) contain one for each name
        dummy_unitconversions = {'a':1, 'c':10, 'mu_n':0.25}
        dummy_do_log = {'a':True, 'b':0, 'mu_n':1, 'mu_p':True}
        
        dummy_param_info = {'names':dummy_names,
                            'unit_conversions':dummy_unitconversions,
                            'do_log':dummy_do_log}
        
        # Test init
        testC = Covariance(dummy_param_info)
        np.testing.assert_equal(testC.cov, np.zeros((d,d)))
        
        # Test initial set
        testC.set_variance('c', 1)
        expected = np.zeros((d,d))
        expected[1,1] = 1
        np.testing.assert_equal(testC.cov, expected)
        
    def test_History(self):
        # Will only look for these
        dummy_names = ['mu_n', 'c', 'b', 'a']

        # Should (but not required) contain one for each name
        dummy_unitconversions = {'a':1, 'c':10, 'mu_n':0.25}
        dummy_do_log = {'a':True, 'b':0, 'mu_n':1, 'mu_p':True}
        
        dummy_param_info = {'names':dummy_names,
                            'unit_conversions':dummy_unitconversions,
                            'do_log':dummy_do_log}
        
        num_iters = 20
        
        # Test init
        testh = History(num_iters, dummy_param_info)
        self.assertEqual(sum(testh.accept), 0)
        self.assertEqual(sum(testh.ratio), 0)
        self.assertEqual(len(testh.accept), num_iters)
        self.assertEqual(len(testh.ratio), num_iters)
        
        for param in dummy_names:
            self.assertEqual(sum(getattr(testh, param)), 0)
            self.assertEqual(sum(getattr(testh, f"mean_{param}")), 0)
            self.assertEqual(len(getattr(testh, param)), num_iters)
            self.assertEqual(len(getattr(testh, f"mean_{param}")), num_iters)
            
        # Test unit conversions: note, these are converting OUT i.e. dividing
        for param in dummy_names:
            setattr(testh, param, getattr(testh, param) + 1)
            setattr(testh, f"mean_{param}", getattr(testh, f"mean_{param}") + 10)
            
        testh.apply_unit_conversions(dummy_param_info)
        expected_vals = {'a':1, 'b':1, 'c':0.1, 'mu_n':4}
        for param in dummy_names:
            np.testing.assert_equal(getattr(testh, param), np.ones(num_iters) * expected_vals[param])
            np.testing.assert_equal(getattr(testh, f"mean_{param}"), np.ones(num_iters) * expected_vals[param] * 10)
            
        # Skipping over export...
        
        # Test truncate
        truncate_at = 10
        for param in dummy_names:
            setattr(testh, param, np.arange(num_iters, dtype=float))
            setattr(testh, f"mean_{param}", np.arange(num_iters, dtype=float) + 10)
        testh.truncate(truncate_at, dummy_param_info)
        
        for param in dummy_names:
            np.testing.assert_equal(getattr(testh, param), np.arange(truncate_at))
            np.testing.assert_equal(getattr(testh, f"mean_{param}"), np.arange(truncate_at) + 10)
            
        self.assertEqual(len(testh.accept), truncate_at)
        self.assertEqual(len(testh.ratio), truncate_at)
        
        # Test KT
        R = 3
        t = 9
        kt = testh.get_KT(dummy_param_info, R, t)
        self.assertEqual(kt.shape, (R, len(dummy_names)))
        
        np.testing.assert_equal(np.mean(kt, axis=0), np.zeros(len(dummy_names)))
    
    def test_HistoryList(self):
        # Will only look for these
        dummy_names = ['mu_n', 'c']
        
        
        dummy_param_info = {'names':dummy_names}
        num_cpus = 4
        num_iters = 10
        # Test init (and join)
        histories_from_each_cpu = [History(num_iters, dummy_param_info) for i in range(num_cpus)]
        for i, h in enumerate(histories_from_each_cpu):
            for param in dummy_names:
                setattr(h, param, np.ones(num_iters) + i)
                setattr(h, f"mean_{param}", 10 * (np.ones(num_iters) + i))
        
        test_hlist = HistoryList(histories_from_each_cpu, dummy_param_info)
        
        self.assertEqual(np.sum(test_hlist.mu_n), 100)
        self.assertEqual(np.sum(test_hlist.mean_mu_n), 1000)
        self.assertEqual(test_hlist.mu_n.shape, (num_cpus, num_iters))
        self.assertEqual(test_hlist.mean_mu_n.shape, (num_cpus, num_iters))
        
        self.assertEqual(np.sum(test_hlist.c), 100)
        self.assertEqual(np.sum(test_hlist.mean_mu_n), 1000)
        
        # Not testing export
        return
    
    def test_Solution(self):
        testS = Solution()
        testG = Grid()
        testG.dx = 1
        
        num_nodes = 128
        num_tsteps = 10
        testS.N = np.ones((num_tsteps, num_nodes))
        testS.P = np.ones((num_tsteps, num_nodes))
        
        # No PL
        param_info = {'names':['B', 'n0', 'p0']}
        initial_guesses = {'B':0, 'n0':0, 'p0':0}
        testP = Parameters(param_info, initial_guesses)
        testS.calculate_PL(testG, testP)
        np.testing.assert_equal(np.zeros(num_tsteps), testS.PL)
        
        # Some PL
        param_info = {'names':['B', 'n0', 'p0']}
        initial_guesses = {'B':1, 'n0':0, 'p0':0}
        testS.P = np.ones((num_tsteps, num_nodes)) * 10
        testP = Parameters(param_info, initial_guesses)
        testS.calculate_PL(testG, testP)
        np.testing.assert_equal(np.ones(num_tsteps) * 10 * num_nodes, testS.PL)
        
        # Some more PL
        param_info = {'names':['B', 'n0', 'p0']}
        initial_guesses = {'B':2, 'n0':1, 'p0':1}
        testS.P = np.ones((num_tsteps, num_nodes)) * 10
        testP = Parameters(param_info, initial_guesses)
        testS.calculate_PL(testG, testP)
        np.testing.assert_equal(np.ones(num_tsteps) * 18 * num_nodes, testS.PL)
        
        return
    
if __name__ == "__main__":
    t = TestUtils()
    t.test_History()