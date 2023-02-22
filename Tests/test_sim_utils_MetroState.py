import unittest
import logging
import numpy as np
import sys
from sim_utils import Parameters, Covariance, MetroState

sys.path.append("..")


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger()
        # Will only look for these
        dummy_names = ['mu_n', 'c', 'b', 'a']

        # Should (but not required) contain one for each name
        dummy_unitconversions = {'a': 1, 'c': 10, 'mu_n': 0.25}
        dummy_do_log = {'a': True, 'b': 0, 'c': 0, 'mu_n': 1, 'mu_p': True}
        dummy_active = {'a': 1, 'b': 1, 'c': 1, 'mu_n': 1}
        dummy_initial_guesses = {'a': 1, 'b': 2, 'c': 3, 'mu_n': 4}
        dummy_initial_variance = {'a': 2, 'b': 2, 'c': 2, 'mu_n': 2}

        dummy_param_info = {'names': dummy_names,
                            'unit_conversions': dummy_unitconversions,
                            'do_log': dummy_do_log,
                            'active': dummy_active,
                            'init_guess': dummy_initial_guesses,
                            'init_variance': dummy_initial_variance}

        dummy_sim_flags = {"likel2variance_ratio": 1000, }
        dummy_sim_flags["annealing"] = (
            max(dummy_initial_variance.values()) *
            dummy_sim_flags["likel2variance_ratio"],
            2, 1)
        num_iters = 100
        self.ms = MetroState(dummy_param_info, dummy_sim_flags, num_iters)

        pass

    def test_MetroState(self):

        # The functionality for each of these has already been covered
        self.assertIsInstance(self.ms.p, Parameters)
        self.assertIsInstance(self.ms.prev_p, Parameters)
        self.assertIsInstance(self.ms.means, Parameters)
        self.assertIsInstance(self.ms.variances, Covariance)

        with self.assertLogs() as captured:
            self.ms.print_status(logger=self.logger)

        # One message per active param
        self.assertEqual(len(captured.records), sum(
            self.ms.param_info['active'].values()))

    def test_annealing(self):
        # Decrease these by 10x every 2 iterations
        orig_sigma = self.ms.MCMC_fields["current_sigma"]
        orig_var = np.array(self.ms.variances.trace())
        self.ms.anneal(1)
        self.assertEqual(self.ms.MCMC_fields["current_sigma"], orig_sigma)
        np.testing.assert_equal(self.ms.variances.trace(), orig_var)
        self.ms.anneal(2)
        self.assertEqual(self.ms.MCMC_fields["current_sigma"], orig_sigma * 0.1)
        np.testing.assert_equal(self.ms.variances.trace(), orig_var * 0.1)
        self.ms.anneal(3)
        self.assertEqual(self.ms.MCMC_fields["current_sigma"], orig_sigma * 0.1)
        np.testing.assert_equal(self.ms.variances.trace(), orig_var * 0.1)
        self.ms.anneal(4)
        self.assertEqual(
            self.ms.MCMC_fields["current_sigma"], orig_sigma * 0.01)
        np.testing.assert_equal(self.ms.variances.trace(), orig_var * 0.01)
        self.ms.anneal(5)
        self.assertEqual(
            self.ms.MCMC_fields["current_sigma"], orig_sigma * 0.01)
        np.testing.assert_equal(self.ms.variances.trace(), orig_var * 0.01)
        self.ms.anneal(6)
        self.assertEqual(
            self.ms.MCMC_fields["current_sigma"], orig_sigma * 0.001)
        np.testing.assert_equal(self.ms.variances.trace(), orig_var * 0.001)

        # Until sigma becomes the min_sigma (1 in this case),
        # at which point it stops, and variance will be l2v times min_sigma
        self.ms.anneal(7)
        self.assertEqual(
            self.ms.MCMC_fields["current_sigma"], orig_sigma * 0.001)
        np.testing.assert_equal(self.ms.variances.trace(), orig_var * 0.001)
        self.ms.anneal(8)
        self.assertEqual(
            self.ms.MCMC_fields["current_sigma"], 1)
        np.testing.assert_equal(
            self.ms.variances.trace(), np.ones_like(orig_var) / 1000)
        self.ms.anneal(9)
        self.assertEqual(
            self.ms.MCMC_fields["current_sigma"], 1)
        np.testing.assert_equal(
            self.ms.variances.trace(), np.ones_like(orig_var) / 1000)
        self.ms.anneal(10)
        self.assertEqual(
            self.ms.MCMC_fields["current_sigma"], 1)
        np.testing.assert_equal(
            self.ms.variances.trace(), np.ones_like(orig_var) / 1000)
