import unittest
import logging
import sys
sys.path.append("..")

import numpy as np

from sim_utils import EnsembleTemplate
q = 1.0  # [e]
q_C = 1.602e-19  # [C]
kB = 8.61773e-5  # [eV / K]

MIN_HMAX = 0.01


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.mock_ensemble = EnsembleTemplate()
        self.mock_ensemble.logger = logging.getLogger()
        self.mock_ensemble.RNG = np.random.default_rng(1)


    def test_select_next_params(self):
        # This function assigns a set of randomly generated values
        np.random.seed(1)
        param_names = ["a", "b", "c", "d"]

        do_log = {"a": 0, "b": 1, "c": 0, "d": 0}

        prior_dist = {"a": (-np.inf, np.inf),
                      "b": (-np.inf, np.inf),
                      "c": (-np.inf, np.inf),
                      "d": (-np.inf, np.inf), }

        initial_guesses = {"a": 0,
                           "b": 100,
                           "c": 0,
                           "d": 10, }

        active_params = {"a": 0,
                         "b": 1,
                         "c": 1,
                         "d": 1, }

        trial_move = {"a": 10,
                      "b": 0.1,
                      "c": 0,
                      "d": 1}

        param_info = {"names": param_names,
                      "prior_dist": prior_dist,
                      "init_guess": initial_guesses,
                      }

        self.mock_ensemble.param_indexes = {name: param_names.index(name) for name in param_names}
        state = [initial_guesses[name] for name in param_names]
        do_log = np.array([do_log[name] for name in param_names], dtype=bool)
        active_params = np.array([active_params[name] for name in param_names], dtype=bool)
        trial_move = np.array([trial_move[name] for name in param_names], dtype=float)
        self.mock_ensemble.ensemble_fields = {"do_log": do_log, "active": active_params,
                                              "trial_move": trial_move}
        # Try box selection
        new_state = self.mock_ensemble.select_next_params(state, param_info)

        # Inactive and shouldn't change
        self.assertEqual(new_state[self.mock_ensemble.param_indexes["a"]], initial_guesses['a'])
        self.assertEqual(new_state[self.mock_ensemble.param_indexes["c"]], initial_guesses['c'])
        num_tests = 100
        for t in range(num_tests):
            new_state = self.mock_ensemble.select_next_params(state, param_info)
            self.assertTrue(np.abs(np.log10(new_state[self.mock_ensemble.param_indexes["b"]]) - np.log10(initial_guesses['b'])) <= 0.1,
                            msg=f"Uniform step #{t} failed: {new_state[self.mock_ensemble.param_indexes['b']]} from mean {initial_guesses['b']} and width 0.1")
            self.assertTrue(np.abs(new_state[self.mock_ensemble.param_indexes["d"]]-initial_guesses['d']) <= 1,
                            msg=f"Uniform step #{t} failed: {new_state[self.mock_ensemble.param_indexes['d']]} from mean {initial_guesses['d']} and width 1")


    def test_mu_constraint(self):
        # This function assigns a set of randomly generated values
        np.random.seed(1)
        param_names = ["mu_n", "mu_p"]

        do_log = {"mu_n": 1, "mu_p": 1}

        prior_dist = {"mu_n": (0.1, np.inf),
                      "mu_p": (0.1, np.inf),
                      }

        initial_guesses = {"mu_n": 20,
                           "mu_p": 20,
                           }

        active_params = {"mu_n": 1,
                         "mu_p": 1,
                         }
        
        trial_move = {"mu_n": 0.1,
                      "mu_p": 0.1}

        param_info = {"names": param_names,
                      "prior_dist": prior_dist,
                      "init_guess": initial_guesses,
                      }

        self.mock_ensemble.param_indexes = {name: param_names.index(name) for name in param_names}
        state = [initial_guesses[name] for name in param_names]
        do_log = np.array([do_log[name] for name in param_names], dtype=bool)
        active_params = np.array([active_params[name] for name in param_names], dtype=bool)
        trial_move = np.array([trial_move[name] for name in param_names], dtype=float)
        self.mock_ensemble.ensemble_fields = {"do_log": do_log, "active": active_params,
                                              "trial_move": trial_move, "do_mu_constraint": (20, 3)}
        for _ in range(10):
            new_state = self.mock_ensemble.select_next_params(state, param_info)

            self.assertTrue(2 / (new_state[self.mock_ensemble.param_indexes["mu_n"]]**-1 + new_state[self.mock_ensemble.param_indexes["mu_p"]]**-1) <= 23)
            self.assertTrue(2 / (new_state[self.mock_ensemble.param_indexes["mu_n"]]**-1 + new_state[self.mock_ensemble.param_indexes["mu_p"]]**-1) >= 17)
