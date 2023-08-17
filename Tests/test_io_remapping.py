import unittest
import numpy as np
from bayes_io import remap_constraint_grps, remap_fittable_inds

class TestUtils(unittest.TestCase):

    def test_remap_fittable_inds(self):
        # Let's assume there are six measurements
        # All fittable and all selected - no change
        fittables = [0, 1, 2, 3, 4, 5]
        select_obs_sets = [0, 1, 2, 3, 4, 5]
        expected_fittables = fittables

        new_fittables = remap_fittable_inds(fittables, select_obs_sets)
        np.testing.assert_equal(expected_fittables, new_fittables)

        # Equal fittables and selected - re-numbered as if only those selected exist
        fittables = [0, 2, 4]
        select_obs_sets = [0, 2, 4]
        expected_fittables = [0, 1, 2]

        new_fittables = remap_fittable_inds(fittables, select_obs_sets)
        np.testing.assert_equal(expected_fittables, new_fittables)

        # More fittables than selected - only those selected,
        # and they should be re-numbered as if only those selected exist
        fittables = [0, 1, 2, 3, 4, 5]
        select_obs_sets = [0, 2, 4]
        expected_fittables = [0, 1, 2]

        new_fittables = remap_fittable_inds(fittables, select_obs_sets)
        np.testing.assert_equal(expected_fittables, new_fittables)

        fittables = [0, 1, 2, 3, 4, 5]
        select_obs_sets = [0, 2]
        expected_fittables = [0, 1]

        new_fittables = remap_fittable_inds(fittables, select_obs_sets)
        np.testing.assert_equal(expected_fittables, new_fittables)

        fittables = [0, 1, 2, 3, 4, 5]
        select_obs_sets = [0]
        expected_fittables = [0]

        new_fittables = remap_fittable_inds(fittables, select_obs_sets)
        np.testing.assert_equal(expected_fittables, new_fittables)

        # Fewer fittables than selected - only those  both fittable and selected
        fittables = [0, 2, 3]
        select_obs_sets = [0, 2, 4]
        expected_fittables = [0, 1]

        new_fittables = remap_fittable_inds(fittables, select_obs_sets)
        np.testing.assert_equal(expected_fittables, new_fittables)

        fittables = [0, 1, 3]
        select_obs_sets = [0, 2, 4]
        expected_fittables = [0]

        new_fittables = remap_fittable_inds(fittables, select_obs_sets)
        np.testing.assert_equal(expected_fittables, new_fittables)

        fittables = [5, 1, 3]
        select_obs_sets = [0, 2, 4]
        expected_fittables = []

        new_fittables = remap_fittable_inds(fittables, select_obs_sets)
        np.testing.assert_equal(expected_fittables, new_fittables)

    def test_remap_constraint_grps(self):
        # Let's assume there are six measurements
        # Every meas in its own grp - all grps will be removed
        # (one should simply specify c_grps as None if all meas are unique)
        c_grps = [(0,), (1,), (2,), (3,), (4,), (5,)]
        select_obs_sets = [0, 1, 2, 3, 4, 5]
        expected_c_grps = []

        new_c_grps = remap_constraint_grps(c_grps, select_obs_sets)
        self.assertEqual(len(new_c_grps), len(expected_c_grps))

        # All meas sharing one c_grp - only those selected,
        # and they should be renumbered as if only those selected exist
        c_grps = [(0, 1, 2, 3, 4, 5,)]
        select_obs_sets = [0, 1, 2, 3, 4, 5]
        expected_c_grps = [(0, 1, 2, 3, 4, 5)]

        new_c_grps = remap_constraint_grps(c_grps, select_obs_sets)
        for i, expected_c_grp in enumerate(expected_c_grps):
            np.testing.assert_equal(new_c_grps[i], expected_c_grp)

        c_grps = [(0, 1, 2, 3, 4, 5,)]
        select_obs_sets = [0, 1, 2]
        expected_c_grps = [(0, 1, 2)]

        new_c_grps = remap_constraint_grps(c_grps, select_obs_sets)
        for i, expected_c_grp in enumerate(expected_c_grps):
            np.testing.assert_equal(new_c_grps[i], expected_c_grp)

        c_grps = [(0, 1, 2, 3, 4, 5,)]
        select_obs_sets = [0, 2, 4]
        expected_c_grps = [(0, 1, 2)]

        new_c_grps = remap_constraint_grps(c_grps, select_obs_sets)
        for i, expected_c_grp in enumerate(expected_c_grps):
            np.testing.assert_equal(new_c_grps[i], expected_c_grp)

        # c_grps of size 1 are removed
        c_grps = [(0, 1, 2, 3, 4, 5,)]
        select_obs_sets = [0]
        expected_c_grps = []

        new_c_grps = remap_constraint_grps(c_grps, select_obs_sets)
        self.assertEqual(len(new_c_grps), len(expected_c_grps))

        # Two c_grps, selecting all
        c_grps = [(0, 1, 2), (3, 4, 5,)]
        select_obs_sets = [0, 1, 2, 3, 4, 5]
        expected_c_grps = [(0, 1, 2), (3, 4, 5)]

        new_c_grps = remap_constraint_grps(c_grps, select_obs_sets)
        for i, expected_c_grp in enumerate(expected_c_grps):
            np.testing.assert_equal(new_c_grps[i], expected_c_grp)

        # Subsetting
        c_grps = [(0, 1, 2), (3, 4, 5,)]
        select_obs_sets = [0, 1, 4, 5]
        expected_c_grps = [(0, 1), (2, 3)]

        new_c_grps = remap_constraint_grps(c_grps, select_obs_sets)
        for i, expected_c_grp in enumerate(expected_c_grps):
            np.testing.assert_equal(new_c_grps[i], expected_c_grp)

        # Subsetting so much that one c_grp gets size 1
        c_grps = [(0, 1, 2), (3, 4, 5,)]
        select_obs_sets = [0, 4, 5]
        expected_c_grps = [(1, 2)]

        new_c_grps = remap_constraint_grps(c_grps, select_obs_sets)
        for i, expected_c_grp in enumerate(expected_c_grps):
            np.testing.assert_equal(new_c_grps[i], expected_c_grp)

        # Non-ascending grouping
        c_grps = [(0, 2, 5), (1, 3, 4)]
        select_obs_sets = [0, 1, 4, 5]
        expected_c_grps = [(0, 3), (1, 2)]

        new_c_grps = remap_constraint_grps(c_grps, select_obs_sets)
        for i, expected_c_grp in enumerate(expected_c_grps):
            np.testing.assert_equal(new_c_grps[i], expected_c_grp)

        # 3 groups
        c_grps = [(0, 2), (1, 5), (3, 4)]
        select_obs_sets = [0, 1, 4, 5]
        expected_c_grps = [(1, 3)]

        new_c_grps = remap_constraint_grps(c_grps, select_obs_sets)
        for i, expected_c_grp in enumerate(expected_c_grps):
            np.testing.assert_equal(new_c_grps[i], expected_c_grp)

if __name__ == "__main__":
    t = TestUtils()
    t.test_remap_constraint_grps()
