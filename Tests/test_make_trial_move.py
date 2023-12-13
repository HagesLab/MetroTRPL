import unittest
import logging
import sys
sys.path.append("..")

import numpy as np

from trial_move_generation import make_trial_move
q = 1.0  # [e]
q_C = 1.602e-19  # [C]
kB = 8.61773e-5  # [eV / K]

MIN_HMAX = 0.01


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger()
        self.RNG = np.random.default_rng(1)


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

        param_indexes = {name: param_names.index(name) for name in param_names}
        state = [initial_guesses[name] for name in param_names]
        do_log = np.array([do_log[name] for name in param_names], dtype=bool)
        trial_move = np.array([trial_move[name] if active_params[name] else 0 for name in param_names], dtype=float)
        active_params = np.array([active_params[name] for name in param_names], dtype=bool)
        ensemble_fields = {"do_log": do_log, "active": active_params,
                           "prior_dist": prior_dist,
                           "_param_indexes": param_indexes, "names": param_names}
        # Try box selection
        new_state = make_trial_move(state, trial_move, ensemble_fields, self.RNG, self.logger)
        # Inactive and shouldn't change
        self.assertEqual(new_state[param_indexes["a"]], initial_guesses['a'])
        self.assertEqual(new_state[param_indexes["c"]], initial_guesses['c'])
        num_tests = 100
        for t in range(num_tests):
            new_state = make_trial_move(state, trial_move, ensemble_fields, self.RNG, self.logger)
            self.assertTrue(np.abs(np.log10(new_state[param_indexes["b"]]) - np.log10(initial_guesses['b'])) <= 0.1,
                            msg=f"Uniform step #{t} failed: {new_state[param_indexes['b']]} from mean {initial_guesses['b']} and width 0.1")
            self.assertTrue(np.abs(new_state[param_indexes["d"]]-initial_guesses['d']) <= 1,
                            msg=f"Uniform step #{t} failed: {new_state[param_indexes['d']]} from mean {initial_guesses['d']} and width 1")


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

        param_indexes = {name: param_names.index(name) for name in param_names}
        state = [initial_guesses[name] for name in param_names]
        do_log = np.array([do_log[name] for name in param_names], dtype=bool)
        trial_move = np.array([trial_move[name] if active_params[name] else 0 for name in param_names], dtype=float)
        active_params = np.array([active_params[name] for name in param_names], dtype=bool)
        ensemble_fields = {"do_log": do_log, "active": active_params,
                           "_param_indexes": param_indexes, "do_mu_constraint": (20, 3),
                           "prior_dist": prior_dist, "names": param_names}
        for _ in range(10):
            new_state = make_trial_move(state, trial_move, ensemble_fields, self.RNG, self.logger)

            self.assertTrue(2 / (new_state[param_indexes["mu_n"]]**-1 + new_state[param_indexes["mu_p"]]**-1) <= 23)
            self.assertTrue(2 / (new_state[param_indexes["mu_n"]]**-1 + new_state[param_indexes["mu_p"]]**-1) >= 17)
