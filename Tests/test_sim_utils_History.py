import unittest
import numpy as np
import sys
sys.path.append("..")
from sim_utils import History

class TestUtils(unittest.TestCase):

    def setUp(self):
        # Will only look for these
        self.dummy_names = ['mu_n', 'c', 'b', 'a']

        # Should (but not required) contain one for each name
        dummy_unitconversions = {'a': 1, 'c': 10, 'mu_n': 0.25}
        dummy_do_log = {'a': True, 'b': 0, 'c': 0, 'mu_n': 1, 'mu_p': True}

        self.dummy_param_info = {'names': self.dummy_names,
                                 'unit_conversions': dummy_unitconversions,
                                 'do_log': dummy_do_log}

        self.num_iters = 20

        self.tasth = History(self.num_iters, self.dummy_param_info)

    def test_initialization(self):
        # Test init
        self.assertEqual(np.sum(self.tasth.accept), 0)
        self.assertEqual(np.sum(self.tasth.loglikelihood), 0)
        self.assertEqual(len(self.tasth.accept[0]), self.num_iters)
        self.assertEqual(len(self.tasth.loglikelihood[0]), self.num_iters)
        
        for param in self.dummy_names:
            self.assertEqual(np.sum(getattr(self.tasth, f"mean_{param}")), 0)
            self.assertEqual(
                len(getattr(self.tasth, f"mean_{param}")[0]), self.num_iters)

    # Skipping over export...

    def test_truncate(self):
        self.tasth = History(self.num_iters, self.dummy_param_info)
        # for param in self.dummy_names:
        #     setattr(self.tasth, param, getattr(self.tasth, param) + 1)
        #     setattr(self.tasth, f"mean_{param}", getattr(self.tasth, f"mean_{param}") + 10)

        # Test truncate
        truncate_at = 10
        for param in self.dummy_names:
            setattr(self.tasth, f"mean_{param}", np.arange(
                self.num_iters, dtype=float).reshape((1, self.num_iters)) + 10)
        self.tasth.truncate(truncate_at, self.dummy_param_info)

        for param in self.dummy_names:
            np.testing.assert_equal(
                getattr(self.tasth, f"mean_{param}")[0], np.arange(truncate_at) + 10)

        self.assertEqual(len(self.tasth.accept[0]), truncate_at)
        self.assertEqual(len(self.tasth.loglikelihood[0]), truncate_at)

    def test_extend(self):
        self.tasth = History(self.num_iters, self.dummy_param_info)
        # for param in self.dummy_names:
        #     setattr(self.tasth, param, getattr(self.tasth, param) + 1)
        #     setattr(self.tasth, f"mean_{param}", getattr(self.tasth, f"mean_{param}") + 10)

        # Test extend from 20 iters to 19 iters, which should result in a contraction
        extend_to = 19
        for param in self.dummy_names:
            setattr(self.tasth, f"mean_{param}", np.arange(
                self.num_iters, dtype=float).reshape((1, self.num_iters)) + 10)
        self.tasth.extend(extend_to, self.dummy_param_info)
        for param in self.dummy_names:
            np.testing.assert_equal(getattr(self.tasth, f"mean_{param}")[0], np.arange(
                extend_to, dtype=float)+10)

        self.assertEqual(len(self.tasth.accept[0]), extend_to)
        self.assertEqual(len(self.tasth.loglikelihood[0]), extend_to)

        # Test extend from 20 iters to 20 iters, which should result in no changes
        self.setUp()
        extend_to = 20
        for param in self.dummy_names:
            setattr(self.tasth, f"mean_{param}", np.arange(
                self.num_iters, dtype=float).reshape((1, self.num_iters)) + 10)
        self.tasth.extend(extend_to, self.dummy_param_info)
        for param in self.dummy_names:
            np.testing.assert_equal(getattr(self.tasth, f"mean_{param}")[0], np.arange(
                self.num_iters, dtype=float)+10)

        self.assertEqual(len(self.tasth.accept[0]), self.num_iters)
        self.assertEqual(len(self.tasth.loglikelihood[0]), self.num_iters)

        # Test extend from 20 iters to 100 iters
        self.setUp()
        extend_to = 100
        for param in self.dummy_names:
            setattr(self.tasth, f"mean_{param}", np.arange(
                self.num_iters, dtype=float).reshape((1, self.num_iters)) + 10)
        self.tasth.extend(extend_to, self.dummy_param_info)

        expected_means = np.concatenate(
            (np.arange(self.num_iters, dtype=float) + 10, np.zeros(extend_to - self.num_iters)))
        for param in self.dummy_names:
            np.testing.assert_equal(
                getattr(self.tasth, f"mean_{param}")[0], expected_means)

        self.assertEqual(len(self.tasth.accept[0]), extend_to)
        self.assertEqual(len(self.tasth.loglikelihood[0]), extend_to)

    def test_update(self):
        self.tasth = History(self.num_iters, self.dummy_param_info)

        # Update only the 3rd history entry
        k = 2

        class DummyParameters():
            def __init__(self):
                self.mu_n = 0
                self.a = 0
                self.b = 0
                self.c = 0
                return

        p = DummyParameters()
        p.c = 5
        p.likelihood = [1, 1, 1]
        means = DummyParameters()
        means.c = 50

        self.tasth.update(k, p, means, self.dummy_param_info)

        self.assertEqual(self.tasth.mean_c[0, k], means.c)
        self.assertEqual(np.sum(self.tasth.mean_c), means.c)

    def test_best_LL(self):
        self.tasth = History(self.num_iters, self.dummy_param_info)

        # Update only the 3rd history entry
        k = 2

        class DummyParameters():
            def __init__(self):
                self.mu_n = 0
                return

        prev_p = DummyParameters()
        prev_p.likelihood = [1, 1, 1]

        self.tasth.record_best_logll(k, prev_p)

        expected_loglikelihood = np.zeros((1, self.num_iters))
        expected_loglikelihood[0, k] = np.sum(prev_p.likelihood)
        np.testing.assert_equal(self.tasth.loglikelihood,
                                expected_loglikelihood)

if __name__ == "__main__":
    t = TestUtils()
    t.setUp()
    t.test_extend()