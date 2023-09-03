import unittest
import numpy as np
import os
from bayes_io import get_initpoints, get_data, insert_param

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

    def test_insert_fluences(self):
        # Let's assume there are six measurements after select_obs_sets was applied
        # After remapping, we are guaranteed a ff[1] with at least 1 val
        # and constraint groups (if any) with vals occuring in ff[1]
        # No constraint groups
        num_meas = 6
        param_info = {"names": [],
                      "active": {},
                      "unit_conversions": {},
                      "do_log": {},
                      "prior_dist": {},
                      "init_guess": {},
                      "init_variance": {}}
        fluences = np.random.random(size=num_meas)
        meas_fields = {"fittable_fluences": (0.02, [0, 1, 2, 3, 4, 5], None, fluences)}
        expected_ffs = [0, 1, 2, 3, 4, 5]
        
        insert_param(param_info, meas_fields)

        for i in expected_ffs:
            self.assertTrue(f"_f{i}" in param_info["names"])
            self.assertEqual(param_info["active"][f"_f{i}"], 1)
            self.assertEqual(param_info["do_log"][f"_f{i}"], 1)
            self.assertEqual(param_info["prior_dist"][f"_f{i}"], (0, np.inf))
            self.assertEqual(param_info["init_guess"][f"_f{i}"], fluences[i])
            self.assertEqual(param_info["init_variance"][f"_f{i}"], 0.02)

        # Null constraint groups
        param_info = {"names": [],
                      "active": {},
                      "unit_conversions": {},
                      "do_log": {},
                      "prior_dist": {},
                      "init_guess": {},
                      "init_variance": {}}
        fluences = np.random.random(size=num_meas)
        meas_fields = {"fittable_fluences": (0.02, [0, 1, 2, 3, 4, 5], [], fluences)}
        expected_ffs = [0, 1, 2, 3, 4, 5]
        insert_param(param_info, meas_fields)

        for i in expected_ffs:
            self.assertTrue(f"_f{i}" in param_info["names"])

        # One constraint group
        param_info = {"names": [],
                      "active": {},
                      "unit_conversions": {},
                      "do_log": {},
                      "prior_dist": {},
                      "init_guess": {},
                      "init_variance": {}}
        fluences = np.random.random(size=num_meas)
        meas_fields = {"fittable_fluences": (0.02, [0, 1, 2, 3, 4, 5], [(1, 2)], fluences)}
        expected_ffs = [0, 1, 3, 4, 5]
        insert_param(param_info, meas_fields)

        for i in expected_ffs:
            self.assertTrue(f"_f{i}" in param_info["names"])
            self.assertEqual(param_info["init_guess"][f"_f{i}"], fluences[i])

        param_info = {"names": [],
                      "active": {},
                      "unit_conversions": {},
                      "do_log": {},
                      "prior_dist": {},
                      "init_guess": {},
                      "init_variance": {}}
        fluences = np.random.random(size=num_meas)
        meas_fields = {"fittable_fluences": (0.02, [0, 1, 2, 3, 4, 5], [(0, 1, 2, 3, 4, 5)], fluences)}
        expected_ffs = [0]
        insert_param(param_info, meas_fields)

        for i in expected_ffs:
            self.assertTrue(f"_f{i}" in param_info["names"])
            self.assertEqual(param_info["init_guess"][f"_f{i}"], fluences[i])

        # Two constraint grp
        param_info = {"names": [],
                      "active": {},
                      "unit_conversions": {},
                      "do_log": {},
                      "prior_dist": {},
                      "init_guess": {},
                      "init_variance": {}}
        fluences = np.random.random(size=num_meas)
        meas_fields = {"fittable_fluences": (0.02, [0, 1, 2, 3, 4, 5], [(0, 2), (3, 4, 5)], fluences)}
        expected_ffs = [0, 1, 3]
        insert_param(param_info, meas_fields)

        for i in expected_ffs:
            self.assertTrue(f"_f{i}" in param_info["names"])
            self.assertEqual(param_info["init_guess"][f"_f{i}"], fluences[i])

        # Three constraint grp
        param_info = {"names": [],
                      "active": {},
                      "unit_conversions": {},
                      "do_log": {},
                      "prior_dist": {},
                      "init_guess": {},
                      "init_variance": {}}
        fluences = np.random.random(size=num_meas)
        meas_fields = {"fittable_fluences": (0.02, [0, 1, 2, 3, 4, 5], [(0, 2), (1, 4), (3, 5)], fluences)}
        expected_ffs = [0, 1, 3]
        insert_param(param_info, meas_fields)

        for i in expected_ffs:
            self.assertTrue(f"_f{i}" in param_info["names"])
            self.assertEqual(param_info["init_guess"][f"_f{i}"], fluences[i])

    def test_insert_absorption(self):
        # Three constraint grp
        # Should function the same as insert_fluences, but params are _a instead of _f
        num_meas = 6
        param_info = {"names": [],
                      "active": {},
                      "unit_conversions": {},
                      "do_log": {},
                      "prior_dist": {},
                      "init_guess": {},
                      "init_variance": {}}
        alphas = np.random.random(size=num_meas)
        meas_fields = {"fittable_absps": (0.02, [0, 1, 2, 3, 4, 5], [(0, 2), (1, 4), (3, 5)], alphas)}
        expected_ffs = [0, 1, 3]
        insert_param(param_info, meas_fields, mode="absorptions")

        for i in expected_ffs:
            self.assertTrue(f"_a{i}" in param_info["names"])
            self.assertEqual(param_info["init_guess"][f"_a{i}"], alphas[i])

    def test_insert_scale_factors(self):
        # Three constraint grp
        # Should function the same as insert_fluences, but params are _s instead of _f
        num_meas = 6
        param_info = {"names": [],
                      "active": {},
                      "unit_conversions": {},
                      "do_log": {},
                      "prior_dist": {},
                      "init_guess": {},
                      "init_variance": {}}
        scale_fs = np.random.random(size=num_meas)
        meas_fields = {"scale_factor": (0.02, [0, 1, 2, 3, 4, 5], [(0, 2), (1, 4), (3, 5)], scale_fs)}
        expected_ffs = [0, 1, 3]
        insert_param(param_info, meas_fields, mode="scale_f")

        for i in expected_ffs:
            self.assertTrue(f"_s{i}" in param_info["names"])
            self.assertEqual(param_info["init_guess"][f"_s{i}"], scale_fs[i])

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

if __name__ == "__main__":
    t = TestUtils()
    t.test_insert_fluences()