import unittest
import numpy as np
import sys
sys.path.append("..")
from sim_utils import Covariance

class TestUtils(unittest.TestCase):
    
    def setUp(self):
        self.dummy_names = ['mu_n', 'c', 'b', 'a']
        self.num_params = len(self.dummy_names)
        
        # Must contain one for each name
        
        # Should (but not required) contain one for each name
        self.dummy_unitconversions = {'a':1, 'c':10, 'mu_n':0.25}
        self.dummy_do_log = {'a':True, 'b':0, 'mu_n':1, 'mu_p':True}
        
        self.dummy_param_info = {'names':self.dummy_names,
                            'unit_conversions':self.dummy_unitconversions,
                            'do_log':self.dummy_do_log,
                            'active':{name:1 for name in self.dummy_names}}
        
        
        self.testC = Covariance(self.dummy_param_info)
           
    def test_initialization(self):
        # Test init
        np.testing.assert_equal(self.testC.cov, np.zeros((self.num_params,self.num_params)))
        
    def test_set_variance(self):
        # Edit a variance for one parameter
        expected = np.zeros((self.num_params,self.num_params))

        self.testC.set_variance('c', 1)
        expected[1,1] = 1

        self.testC.set_variance('a', {'a':26})
        expected[3,3] = 26
        
        np.testing.assert_equal(self.testC.cov, expected)
        
        np.testing.assert_equal(self.testC.trace(), [0,1,0,26])
        
    def test_apply_variances_ident(self):
        # Edit variances for multiple parameters
        
        # Regenerate test object
        self.testC = Covariance(self.dummy_param_info)
        
        initial_variance = 10
        
        self.testC.apply_values(initial_variance)
        
        # Single value initial_variance means all params have this variance
        np.testing.assert_equal(self.testC.little_sigma, [initial_variance] * self.num_params)
        np.testing.assert_equal(self.testC.big_sigma, np.eye(self.num_params))
        
        # Mask all but second param
        self.testC.mask_covariance(('c', 1))
        
        expected_cov = np.zeros((self.num_params, self.num_params))
        expected_cov[1,1] = initial_variance
        np.testing.assert_equal(expected_cov, self.testC.cov)
        
        # Unmask
        self.testC.mask_covariance(None)
        
        np.testing.assert_equal(self.testC.cov, np.eye(self.num_params) * initial_variance)
        
    def test_apply_variances_diff(self):
        # Regenerate test object
        self.testC = Covariance(self.dummy_param_info)
        
        initial_variance = {'c':10, 'b':9, 'a':8, 'mu_n':1}
        expected_cov = np.zeros((self.num_params, self.num_params))
        expected_cov[0,0] = 1
        expected_cov[1,1] = 10
        expected_cov[2,2] = 9
        expected_cov[3,3] = 8
        self.testC.apply_values(initial_variance)