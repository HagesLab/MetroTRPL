import unittest
import logging
import sys
sys.path.append("..")
from sim_utils import Parameters,Covariance, MetroState

class TestUtils(unittest.TestCase):       
    
    def setUp(self):
        self.logger = logging.getLogger()
        pass
    
    def test_MetroState(self):
        # Will only look for these
        dummy_names = ['mu_n', 'c', 'b', 'a']

        # Should (but not required) contain one for each name
        dummy_unitconversions = {'a':1, 'c':10, 'mu_n':0.25}
        dummy_do_log = {'a':True, 'b':0, 'c':0, 'mu_n':1, 'mu_p':True}
        dummy_active = {'a':1, 'b':1, 'c':1, 'mu_n':1}
        dummy_initial_guesses = {'a':1, 'b':2, 'c':3, 'mu_n':4}
        dummy_initial_variance = {'a':2, 'b':1, 'c':2, 'mu_n':1}
        
        dummy_param_info = {'names':dummy_names,
                            'unit_conversions':dummy_unitconversions,
                            'do_log':dummy_do_log,
                            'active':dummy_active,
                            'init_guess':dummy_initial_guesses,
                            'init_variance':dummy_initial_variance}
        
        
        dummy_sim_flags = {} # Only so this gets exported as part of the pickle
        num_iters = 100
        ms = MetroState(dummy_param_info, dummy_sim_flags, num_iters)
        
        # The functionality for each of these has already been covered
        self.assertIsInstance(ms.p, Parameters)
        self.assertIsInstance(ms.prev_p, Parameters)
        self.assertIsInstance(ms.means, Parameters)
        self.assertIsInstance(ms.variances, Covariance)
        
        with self.assertLogs() as captured:
            ms.print_status(logger=self.logger)
            
        # One message per active param
        self.assertEqual(len(captured.records), sum(dummy_active.values()))
