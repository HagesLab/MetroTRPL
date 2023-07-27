import unittest
import numpy as np
import os

from bayes_io import get_initpoints, get_data, insert_scale_factors

class TestUtils(unittest.TestCase):

    def test_get_initpoints(self):
        ic_flags = {'select_obs_sets': None}
        where_inits = os.path.join("Tests", "testfiles", "test_initpoints.csv")
        ic = get_initpoints(where_inits, ic_flags)

        expected = np.array([[0, 0, 0, 0, 0], [1, 2, 3, 4, 5],
                             [1, 1, 1, 1, 1]], dtype=float)
        np.testing.assert_equal(expected, ic)

        ic_flags = {'select_obs_sets': [1]}
        ic = get_initpoints(where_inits, ic_flags)

        expected = np.array([[1, 2, 3, 4, 5]], dtype=float)
        np.testing.assert_equal(expected, ic)
        return
    
    def test_insert_scale_factors(self):
        # General test case with multiple measurements; mix of measurement types,
        # only necessary settings defined.
        num_meas = 6
        grid = {"meas_types": ["TRPL"] * 3 + ["TRTS"] * 3,
                "num_meas": num_meas}
        
        param_info = {"names": [],
                      "active": {},
                      "unit_conversions": {},
                      "do_log": {},
                      "prior_dist": {},
                      "init_guess": {},
                      "init_variance": {}}
        
        meas_fields = {"select_obs_sets": None}

        # 1. No scale_factor - no change to param_info
        MCMC_fields = {"self_normalize": None,
                       "scale_factor": None,
                       }

        insert_scale_factors(grid, param_info, meas_fields, MCMC_fields)

        for k in param_info:
            self.assertEqual(len(param_info[k]), 0)

        # 2. Global scale_factor - _s parameter added with desired guess and variance
        param_info = {"names": [],
                      "active": {},
                      "unit_conversions": {},
                      "do_log": {},
                      "prior_dist": {},
                      "init_guess": {},
                      "init_variance": {}}
        init_guess = 1
        init_var = 0.1
        MCMC_fields["scale_factor"] = ("global", init_guess, init_var) # type: ignore

        insert_scale_factors(grid, param_info, meas_fields, MCMC_fields)

        self.assertTrue("_s" in param_info["names"])
        self.assertEqual(param_info["active"]["_s"], 1)
        self.assertEqual(param_info["do_log"]["_s"], 1)
        self.assertEqual(param_info["prior_dist"]["_s"], (-np.inf, np.inf))
        self.assertEqual(param_info["init_guess"]["_s"], init_guess)
        self.assertEqual(param_info["init_variance"]["_s"], init_var)

        # 3. Independent scale_factors - one _s per num_meas
        param_info = {"names": [],
                      "active": {},
                      "unit_conversions": {},
                      "do_log": {},
                      "prior_dist": {},
                      "init_guess": {},
                      "init_variance": {}}
        MCMC_fields["scale_factor"] = ("ind", init_guess, init_var) # type: ignore

        insert_scale_factors(grid, param_info, meas_fields, MCMC_fields)

        for i in range(num_meas):
            self.assertTrue(f"_s{i}" in param_info["names"])
            self.assertEqual(param_info["active"][f"_s{i}"], 1)
            self.assertEqual(param_info["do_log"][f"_s{i}"], 1)
            self.assertEqual(param_info["prior_dist"][f"_s{i}"], (-np.inf, np.inf))
            self.assertEqual(param_info["init_guess"][f"_s{i}"], init_guess)
            self.assertEqual(param_info["init_variance"][f"_s{i}"], init_var)

        # 4. If select_obs_sets limits the number of measurements considered, only one _s for each active measurement
        param_info = {"names": [],
                      "active": {},
                      "unit_conversions": {},
                      "do_log": {},
                      "prior_dist": {},
                      "init_guess": {},
                      "init_variance": {}}
        meas_fields = {"select_obs_sets": [0, 4, 5]}

        insert_scale_factors(grid, param_info, meas_fields, MCMC_fields)

        for i in range(len(meas_fields["select_obs_sets"])):
            self.assertTrue(f"_s{i}" in param_info["names"])
            self.assertEqual(param_info["active"][f"_s{i}"], 1)
            self.assertEqual(param_info["do_log"][f"_s{i}"], 1)
            self.assertEqual(param_info["prior_dist"][f"_s{i}"], (-np.inf, np.inf))
            self.assertEqual(param_info["init_guess"][f"_s{i}"], init_guess)
            self.assertEqual(param_info["init_variance"][f"_s{i}"], init_var)

        self.assertFalse(f"_s{len(meas_fields['select_obs_sets'])}" in param_info["names"])

    def test_get_data_basic(self):
        # Basic selection and cutting operations on dataset
        meas_types = ["TRPL"] * 5
        ic_flags = {'time_cutoff': None,
                    'select_obs_sets': None,
                    'noise_level': None}

        sim_flags = {'log_pl': False,
                     'self_normalize': None}

        where_inits = os.path.join("Tests", "testfiles", "test_data.csv")

        times, vals, uncs = get_data(
            where_inits, meas_types, ic_flags, sim_flags)
        expected_times = [np.array([0]), np.arange(5), np.array(
            [0, 10, 20]), np.array([0]), np.array([0, 1])]
        expected_vals = [np.array([0]), np.ones(5), np.array(
            [2, 2, 2]), np.array([3]), np.array([4, 4])]
        expected_uncs = [np.array([0]), np.array([10, 10, 10, 10, 10]), np.array(
            [20, 20, 20]), np.array([30]), np.array([40, 40])]

        np.testing.assert_equal(times, expected_times)
        np.testing.assert_equal(vals, expected_vals)
        np.testing.assert_equal(uncs, expected_uncs)

        ic_flags['time_cutoff'] = [-np.inf, 1] # type: ignore

        times, vals, uncs = get_data(
            where_inits, meas_types, ic_flags, sim_flags)
        expected_times = [np.array([0]), np.array(
            [0, 1]), np.array([0]), np.array([0]), np.array([0, 1])]
        expected_vals = [np.array([0]), np.array([1, 1]), np.array(
            [2]), np.array([3]), np.array([4, 4])]
        expected_uncs = [np.array([0]), np.array([10, 10]), np.array(
            [20]), np.array([30]), np.array([40, 40])]

        np.testing.assert_equal(times, expected_times)
        np.testing.assert_equal(vals, expected_vals)
        np.testing.assert_equal(uncs, expected_uncs)

        ic_flags['select_obs_sets'] = [0, 4] # type: ignore
        times, vals, uncs = get_data(
            where_inits, meas_types, ic_flags, sim_flags)
        expected_times = [np.array([0]), np.array([0, 1])]
        expected_vals = [np.array([0]), np.array([4, 4])]
        expected_uncs = [np.array([0]), np.array([40, 40])]

        np.testing.assert_equal(times, expected_times)
        np.testing.assert_equal(vals, expected_vals)
        np.testing.assert_equal(uncs, expected_uncs)

    def test_get_data_transform(self):
        # Normalization and log operators
        meas_types = ["TRPL"] * 5
        ic_flags = {'time_cutoff': None,
                    'select_obs_sets': None,
                    'noise_level': None}

        sim_flags = {'log_pl': False,
                     'self_normalize': None}

        where_inits = os.path.join("Tests", "testfiles", "test_data.csv")
        ic_flags['time_cutoff'] = [-np.inf, 1] # type: ignore
        ic_flags['select_obs_sets'] = [0, 4] # type: ignore

        sim_flags["self_normalize"] = ["TRPL"]
        with np.errstate(divide='ignore', invalid='ignore'):
            times, vals, uncs = get_data(
                where_inits, meas_types, ic_flags, sim_flags)
        expected_times = [np.array([0]), np.array([0, 1])]
        # First curve is a single datapoint with val=0, so norm should fail
        # Second curve orig vals is 4, so should be divided by 4
        expected_vals = [np.array([np.nan]), np.array([1, 1])]
        expected_uncs = [np.array([np.nan]), np.array([10, 10])]

        np.testing.assert_equal(times, expected_times)
        np.testing.assert_equal(vals, expected_vals)
        np.testing.assert_equal(uncs, expected_uncs)

        # Order of ops: should norm then log
        sim_flags["log_pl"] = True
        with np.errstate(divide='ignore', invalid='ignore'):
            times, vals, uncs = get_data(
                where_inits, meas_types, ic_flags, sim_flags)
        expected_times = [np.array([0]), np.array([0, 1])]

        expected_vals = [np.array([np.nan]), np.array([0, 0])]
        expected_uncs = [np.array([np.nan]), np.array([10, 10]) / np.log(10)]

        np.testing.assert_equal(times, expected_times)
        np.testing.assert_equal(vals, expected_vals)
        np.testing.assert_equal(uncs, expected_uncs)

        ic_flags = {'time_cutoff': None,
                    'select_obs_sets': None,
                    'noise_level': None}

        sim_flags = {'log_pl': False,
                     'self_normalize': None}

        ic_flags['select_obs_sets'] = [1] # type: ignore

        ic_flags['time_cutoff'] = [1, 3] # type: ignore

        times, vals, uncs = get_data(
            where_inits, meas_types, ic_flags, sim_flags)
        expected_times = [np.array([1, 2, 3])]
        expected_vals = [np.array([1, 1, 1])]
        expected_uncs = [np.array([10, 10, 10])]

        np.testing.assert_equal(times, expected_times)
        np.testing.assert_equal(vals, expected_vals)
        np.testing.assert_equal(uncs, expected_uncs)
