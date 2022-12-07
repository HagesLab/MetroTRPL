import unittest
import numpy as np
import os

from bayes_io import get_initpoints, get_data
eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
q = 1.0 # [e]
q_C = 1.602e-19 # [C]
kB = 8.61773e-5  # [eV / K]

class TestUtils(unittest.TestCase):
    
    def test_get_initpoints(self):
        ic_flags = {'select_obs_sets':None}
        where_inits = os.path.join("Tests", "testfiles", "test_initpoints.csv")
        ic = get_initpoints(where_inits, ic_flags, scale_f=1)
       
        expected = np.array([[0,0,0,0,0], [1,2,3,4,5], [1,1,1,1,1]], dtype=float)
        np.testing.assert_equal(expected, ic)

        ic_flags = {'select_obs_sets':[1]}
        ic = get_initpoints(where_inits, ic_flags, scale_f=1)
       
        expected = np.array([[1,2,3,4,5]], dtype=float)
        np.testing.assert_equal(expected, ic)
        return
    
    def test_get_data(self):
        ic_flags = {'time_cutoff':None,
                    'select_obs_sets':None,
                    'noise_level':None}
        
        sim_flags = {'log_pl':False,
                     'self_normalize':False}
        
        where_inits = os.path.join("Tests", "testfiles", "test_data.csv")
        
        times, vals, uncs = get_data(where_inits, ic_flags, sim_flags, scale_f=1)
        expected_times = [np.array([0]), np.arange(5), np.array([0,10,20]), np.array([0]), np.array([0,1])]
        expected_vals = [np.array([0]), np.ones(5), np.array([2,2,2]), np.array([3]), np.array([4,4])]
        expected_uncs = [np.array([0]), np.array([10, 10, 10, 10, 10]), np.array([20,20,20]), np.array([30]), np.array([40,40])]
        
        np.testing.assert_equal(times, expected_times)
        np.testing.assert_equal(vals, expected_vals)
        np.testing.assert_equal(uncs, expected_uncs)
        
        ic_flags['time_cutoff'] = [-np.inf, 1]
        
        times, vals, uncs = get_data(where_inits, ic_flags, sim_flags, scale_f=1)
        expected_times = [np.array([0]), np.array([0,1]), np.array([0]), np.array([0]), np.array([0,1])]
        expected_vals = [np.array([0]), np.array([1,1]), np.array([2]), np.array([3]), np.array([4,4])]
        expected_uncs = [np.array([0]), np.array([10, 10]), np.array([20]), np.array([30]), np.array([40,40])]
        
        np.testing.assert_equal(times, expected_times)
        np.testing.assert_equal(vals, expected_vals)
        np.testing.assert_equal(uncs, expected_uncs)
        
        ic_flags['select_obs_sets'] = [0, 4]
        times, vals, uncs = get_data(where_inits, ic_flags, sim_flags, scale_f=1)
        expected_times = [np.array([0]), np.array([0,1])]
        expected_vals = [np.array([0]), np.array([4,4])]
        expected_uncs = [np.array([0]), np.array([40, 40])]
        
        np.testing.assert_equal(times, expected_times)
        np.testing.assert_equal(vals, expected_vals)
        np.testing.assert_equal(uncs, expected_uncs)
        
        sim_flags["self_normalize"] = True
        with np.errstate(divide='ignore', invalid='ignore'):
            times, vals, uncs = get_data(where_inits, ic_flags, sim_flags, scale_f=1)
        expected_times = [np.array([0]), np.array([0,1])]
        expected_vals = [np.array([np.nan]), np.array([1,1])]
        expected_uncs = [np.array([0]), np.array([40, 40])]
        
        np.testing.assert_equal(times, expected_times)
        np.testing.assert_equal(vals, expected_vals)
        np.testing.assert_equal(uncs, expected_uncs)
        
        sim_flags["log_pl"] = True
        with np.errstate(divide='ignore', invalid='ignore'):
            times, vals, uncs = get_data(where_inits, ic_flags, sim_flags, scale_f=1)
        expected_times = [np.array([0]), np.array([0,1])]
        expected_vals = [np.array([np.nan]), np.array([0,0])]
        expected_uncs = [np.array([np.nan]), np.array([40, 40]) / np.log(10)]
        
        np.testing.assert_equal(times, expected_times)
        np.testing.assert_equal(vals, expected_vals)
        np.testing.assert_equal(uncs, expected_uncs)
        
        ic_flags["noise_level"] = 1
        np.random.seed(42)
        with np.errstate(divide='ignore', invalid='ignore'):
            times, vals, uncs = get_data(where_inits, ic_flags, sim_flags, scale_f=1, verbose=True)
            
        
        ic_flags = {'time_cutoff':None,
                    'select_obs_sets':None,
                    'noise_level':None}
        
        sim_flags = {'log_pl':False,
                     'self_normalize':False}
        
        ic_flags['select_obs_sets'] = [1]
        
        ic_flags['time_cutoff'] = [1, 3]
        
        times, vals, uncs = get_data(where_inits, ic_flags, sim_flags, scale_f=1)
        expected_times = [np.array([1,2,3])]
        expected_vals = [np.array([1,1,1])]
        expected_uncs = [np.array([10, 10, 10])]
        
        np.testing.assert_equal(times, expected_times)
        np.testing.assert_equal(vals, expected_vals)
        np.testing.assert_equal(uncs, expected_uncs)