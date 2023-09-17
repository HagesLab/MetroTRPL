from sim_utils import Parameters, Covariance, MetroState
import unittest
import logging
import numpy as np
import sys

sys.path.append("..")


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
        dummy_initial_variance = {'a': 2, 'b': 2, 'c': 2, 'mu_n': 2}

        dummy_param_info = {'names': dummy_names,
                            'unit_conversions': dummy_unitconversions,
                            'do_log': dummy_do_log,
                            'active': dummy_active,
                            'init_guess': dummy_initial_guesses,
                            'init_variance': dummy_initial_variance}

        dummy_sim_flags = {"likel2variance_ratio": {"TRPL":1000, "TRTS": 2000},
                           "annealing": tuple[dict, int, dict]}
        
        annealing_step = 2
        min_sigma = 1
        dummy_sim_flags["annealing"] = ({m:max(dummy_initial_variance.values()) * dummy_sim_flags["likel2variance_ratio"][m]
                                    for m in dummy_simPar["meas_types"]},
                                    annealing_step,
                                    {m:min_sigma for m in dummy_simPar["meas_types"]})
        
        num_iters = 100
        self.ms = MetroState(dummy_param_info, dummy_sim_flags, num_iters)
        self.ms.sim_info = dummy_simPar

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

    def test_annealing_of_sigma(self):
        # Ensure that the sigma is actually becoming more selective
        # following the inputted schedule
        # Decrease these by 10x every 2 iterations
        orig_sigma = dict(self.ms.MCMC_fields["current_sigma"])
        orig_var = np.array(self.ms.variances.trace())
        self.ms.prev_p.likelihood = []  # Not testing with any measurements atm

        self.ms.anneal(1)
        for m in self.ms.sim_info["meas_types"]:
            self.assertEqual(self.ms.MCMC_fields["current_sigma"][m], orig_sigma[m])
            np.testing.assert_equal(self.ms.variances.trace(), orig_var)
        self.ms.anneal(2)
        for m in self.ms.sim_info["meas_types"]:
            self.assertEqual(self.ms.MCMC_fields["current_sigma"][m], orig_sigma[m] * 0.1)
            np.testing.assert_equal(self.ms.variances.trace(), orig_var * 0.1)
        self.ms.anneal(3)
        for m in self.ms.sim_info["meas_types"]:
            self.assertEqual(self.ms.MCMC_fields["current_sigma"][m], orig_sigma[m] * 0.1)
            np.testing.assert_equal(self.ms.variances.trace(), orig_var * 0.1)
        self.ms.anneal(4)
        for m in self.ms.sim_info["meas_types"]:
            self.assertEqual(
                self.ms.MCMC_fields["current_sigma"][m], orig_sigma[m] * 0.01)
            np.testing.assert_equal(self.ms.variances.trace(), orig_var * 0.01)
        self.ms.anneal(5)
        for m in self.ms.sim_info["meas_types"]:
            self.assertEqual(
                self.ms.MCMC_fields["current_sigma"][m], orig_sigma[m] * 0.01)
            np.testing.assert_equal(self.ms.variances.trace(), orig_var * 0.01)
        self.ms.anneal(6)
        for m in self.ms.sim_info["meas_types"]:
            self.assertEqual(
                self.ms.MCMC_fields["current_sigma"][m], orig_sigma[m] * 0.001)
            np.testing.assert_equal(self.ms.variances.trace(), orig_var * 0.001)

        # Until sigma becomes the min_sigma (1 in this case),
        # at which point it stops, and variance will be l2v times min_sigma
        self.ms.anneal(7)
        for m in self.ms.sim_info["meas_types"]:

            self.assertEqual(
                self.ms.MCMC_fields["current_sigma"][m], orig_sigma[m] * 0.001)
            np.testing.assert_equal(self.ms.variances.trace(), orig_var * 0.001)
        self.ms.anneal(8)
        for m in self.ms.sim_info["meas_types"]:
            self.assertEqual(
                self.ms.MCMC_fields["current_sigma"][m], 1)
            np.testing.assert_equal(
                self.ms.variances.trace(), np.ones_like(orig_var) / 1000)
        self.ms.anneal(9)
        for m in self.ms.sim_info["meas_types"]:
            self.assertEqual(
                self.ms.MCMC_fields["current_sigma"][m], 1)
            np.testing.assert_equal(
                self.ms.variances.trace(), np.ones_like(orig_var) / 1000)
        self.ms.anneal(10)
        for m in self.ms.sim_info["meas_types"]:
            self.assertEqual(
                self.ms.MCMC_fields["current_sigma"][m], 1)
            np.testing.assert_equal(
                self.ms.variances.trace(), np.ones_like(orig_var) / 1000)

    def test_annealing_recalculating_likel(self):
        # Ensure that when an anneal step happens, the likelihood of
        # the previous state (calculated pre-annealing) is recalculated
        # with the post-annealing uncertainty, so that it has a comparable
        # likelihood with future states
        # This supports the idea of restarting an MMC when an anneal happens
        uncs = [np.array([0, 0, 0])]
        err_sq = [np.array([1, 1, 1])]
        self.ms.prev_p.err_sq = err_sq
        self.ms.prev_p.likelihood = [-np.inf]
        self.ms.prev_p.likelihood[0] = - \
            np.sum(
                err_sq[0] / (self.ms.MCMC_fields["current_sigma"]["TRPL"]**2 + 2*uncs[0]**2))

        # sigma 10 times smaller makes likel 100 times more selective
        # but no change to underlying sum of err sq
        expected_err_sq = [[1, 1, 1]]
        expected_likelihood = self.ms.prev_p.likelihood[0] * 100

        self.ms.anneal(2, uncs)

        np.testing.assert_equal(self.ms.prev_p.err_sq, expected_err_sq)
        self.assertEqual(self.ms.prev_p.likelihood[0], expected_likelihood)

    def test_annealing_recalculating_likel_withexpunc(self):
        # Result will be a little different with nonzero exp unc
        uncs = [np.array([1000, 1000, 1000])]
        err_sq = [np.array([1, 1, 1])]
        self.ms.prev_p.err_sq = err_sq
        self.ms.prev_p.likelihood = [-np.inf]
        self.ms.prev_p.likelihood[0] = - \
            np.sum(
                err_sq[0] / (self.ms.MCMC_fields["current_sigma"]["TRPL"]**2 + 2*uncs[0]**2))

        # sigma 10 times smaller makes likel 100 times more selective
        # but no change to underlying sum of err sq
        expected_err_sq = [[1, 1, 1]]
        expected_likelihood = - \
            np.sum(
                err_sq[0] / (0.01*self.ms.MCMC_fields["current_sigma"]["TRPL"]**2 + 2*uncs[0]**2))

        self.ms.anneal(2, uncs)

        np.testing.assert_equal(self.ms.prev_p.err_sq, expected_err_sq)
        self.assertEqual(self.ms.prev_p.likelihood[0], expected_likelihood)


if __name__ == "__main__":
    t = TestUtils()
    t.setUp()
    t.test_annealing_recalculating_likel()
