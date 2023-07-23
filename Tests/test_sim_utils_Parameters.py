import unittest
import numpy as np
import sys
sys.path.append("..")
from sim_utils import Parameters

class TestUtils(unittest.TestCase):
    
    def setUp(self):
        # Will only look for these
        self.dummy_names = ['mu_n', 'c', 'b', 'a']
        
        # Must contain one for each name
        self.dummy_parameters = {'a':1, 'b':2, 'c':3, 'mu_n':4}
        
        # Should (but not required) contain one for each name
        self.dummy_unitconversions = {'a':1, 'c':10, 'mu_n':0.25}
        self.dummy_do_log = {'a':True, 'b':0, 'mu_n':1, 'mu_p':True}
        self.dummy_active = {'a':1, 'b':1, 'c':1, 'mu_n':1}
        
        self.dummy_param_info = {'names':self.dummy_names,
                            'unit_conversions':self.dummy_unitconversions,
                            'do_log':self.dummy_do_log,
                            'active':self.dummy_active,
                            'init_guess':self.dummy_parameters}
        
        # Our working Parameters object example
        self.testp = Parameters(self.dummy_param_info)
    
    def test_initialization(self):
        # Test initialization
        for param in self.dummy_names:
            self.assertEqual(getattr(self.testp, param), self.dummy_parameters[param])
            
        # THere should also be the 'm' param defined by default
        self.assertEqual(getattr(self.testp, 'm'), 1)
            
    def test_as_array(self):
        # Test asarray
        arr = self.testp.to_array(self.dummy_param_info)
        expected = np.array([4,3,2,1])
        np.testing.assert_equal(arr, expected)
        
    def test_duplicate_params(self):
        # Test init with duplicate params
        duplicate_names = ['a', 'a']
        dup_info = {'names':duplicate_names,
                    'unit_conversions':self.dummy_unitconversions,
                    'init_guess':self.dummy_parameters}
        with self.assertRaises(KeyError):
            Parameters(dup_info)
            
    def test_missing_params(self):
        # Test init with missing param values
        bad_params = {}
        dup_info = {'names':self.dummy_names,
                    'init_guess':bad_params}
        
        with self.assertRaises(KeyError):
            Parameters(dup_info)
            
    def test_unit_conversion(self):
        # Test unit conversion
        expected_converted_values = {'a':1, 'b':2, 'c':30, 'mu_n':1.0}
        self.testp.apply_unit_conversions(self.dummy_param_info)
        for param in self.dummy_names:
            self.assertEqual(getattr(self.testp, param), expected_converted_values[param])
            
    def test_do_log(self):
        # Test make log
        # Regenerate original Parmaeters() first
        self.testp = Parameters(self.dummy_param_info)
        self.testp.make_log(self.dummy_param_info)
        expected_logged_values = {'a':0, 'b':2, 'c':3, 'mu_n':np.log10(4)}
        for param in self.dummy_names:
            self.assertEqual(getattr(self.testp, param), expected_logged_values[param])
            
    def test_transfer(self):
        
        other_names = ['mu_n', 'c', 'b', 'a', "extra", "params", "we", "dont", "want"]
        
        # Must contain one for each name
        other_parameters = {'a':0, 'b':0, 'c':0, 'mu_n':0,
                            "extra":-1,"params":-1,"we":-1, "dont":-1,"want":-1}
        
        # Should (but not required) contain one for each name

        other_param_info = {"names":other_names,
                            "active":{},
                            "init_guess":other_parameters}
        other = Parameters(other_param_info)
        self.testp = Parameters(self.dummy_param_info)
        
        self.testp.transfer_from(other, self.dummy_param_info)
        
        for param in self.dummy_names:
            self.assertEqual(getattr(self.testp, param), other_parameters[param])

    def test_suppress_scale_factor(self):
        self.testP = Parameters(self.dummy_param_info)

        self.testP._s = 1000
        self.testP._s0 = 1000
        self.testP._s1 = 1000

        self.testP.suppress_scale_factor(None, 0)
        self.assertEqual(self.testP._s, 1000)
        self.assertEqual(self.testP._s0, 1000)
        self.assertEqual(self.testP._s1, 1000)

        self.testP.suppress_scale_factor(("global", 1, 1), 0)
        self.assertEqual(self.testP._s, 1)
        self.assertEqual(self.testP._s0, 1000)
        self.assertEqual(self.testP._s1, 1000)

        self.testP.suppress_scale_factor(("ind", 1, 1), 0)
        self.assertEqual(self.testP._s, 1)
        self.assertEqual(self.testP._s0, 1)
        self.assertEqual(self.testP._s1, 1000)

        self.testP.suppress_scale_factor(("ind", 1, 1), 1)
        self.assertEqual(self.testP._s, 1)
        self.assertEqual(self.testP._s0, 1)
        self.assertEqual(self.testP._s1, 1)

    def test_get_scale_factor(self):
        self.testP = Parameters(self.dummy_param_info)

        self.testP._s = 10
        self.testP._s0 = 100
        self.testP._s1 = 1000

        self.assertEqual(self.testP.get_scale_factor(None, 0), 0)
        self.assertEqual(self.testP.get_scale_factor(("global", 1, 1), 0), 1)
        self.assertEqual(self.testP.get_scale_factor(("ind", 1, 1), 0), 2)
        self.assertEqual(self.testP.get_scale_factor(("ind", 1, 1), 1), 3)
