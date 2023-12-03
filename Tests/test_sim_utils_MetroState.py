import unittest
import logging
import sys

sys.path.append("..")
from sim_utils import Parameters, MetroState


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger()
        # Will only look for these
        dummy_names = ['mu_n', 'c', 'b', 'a']

        # Should (but not required) contain one for each name
        dummy_simPar = {"meas_types": ["TRPL", "TRTS"]}
        dummy_unitconversions = {'a': 1, 'c': 10, 'mu_n': 0.25}
        dummy_do_log = {'a': True, 'b': 0, 'c': 0, 'mu_n': 1, 'mu_p': True}
        dummy_active = {'a': 1, 'b': 1, 'c': 1, 'mu_n': 1}
        dummy_initial_guesses = {'a': 1, 'b': 2, 'c': 3, 'mu_n': 4}
        dummy_trial_move = {'a': 2, 'b': 2, 'c': 2, 'mu_n': 2}

        dummy_param_info = {'names': dummy_names,
                            'unit_conversions': dummy_unitconversions,
                            'do_log': dummy_do_log,
                            'active': dummy_active,
                            'init_guess': dummy_initial_guesses,
                            'trial_move': dummy_trial_move}

        dummy_sim_flags = {"likel2move_ratio": {"TRPL":1000, "TRTS": 2000},
                           "annealing": tuple[dict, int, dict]}
        
        num_iters = 100
        self.ms = MetroState(dummy_param_info, dummy_sim_flags, num_iters)

    def test_MetroState(self):
        # The functionality for each of these has already been covered
        self.assertIsInstance(self.ms.p, Parameters)
        self.assertIsInstance(self.ms.prev_p, Parameters)
        self.assertIsInstance(self.ms.means, Parameters)

        with self.assertLogs() as captured:
            self.ms.print_status(logger=self.logger)

        # One message per active param
        self.assertEqual(len(captured.records), sum(
            self.ms.param_info['active'].values()))
