import unittest
import numpy as np
import os
import sys
sys.path.append("..")
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

if __name__ == "__main__":
    t = TestUtils()
    t.test_remap_fittable_inds()