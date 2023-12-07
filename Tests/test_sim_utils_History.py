from sim_utils import History
import unittest
import numpy as np
import sys
sys.path.append("..")


class TestUtils(unittest.TestCase):

    def setUp(self):
        # Will only look for these
        self.dummy_names = ['mu_n', 'c', 'b', 'a']
        self.indexes = {"mu_n": 0,
                        "c": 1,
                        "b": 2,
                        "a": 3}

        self.num_iters = 20

    def test_initialization(self):
        # Test init
        testh = History(self.num_iters, self.dummy_names)
        self.assertEqual(np.sum(testh.accept), 0)
        self.assertEqual(np.sum(testh.loglikelihood), 0)
        self.assertEqual(len(testh.accept[0]), self.num_iters)
        self.assertEqual(len(testh.loglikelihood[0]), self.num_iters)
        self.assertEqual(testh.states.shape, (len(self.dummy_names), self.num_iters))
        self.assertEqual(np.sum(testh.states), 0)

    # Skipping over export...

    def test_truncate(self):
        testh = History(self.num_iters, self.dummy_names)
        # for param in self.dummy_names:
        #     setattr(self.tasth, param, getattr(self.tasth, param) + 1)
        #     setattr(self.tasth, f"mean_{param}", getattr(self.tasth, f"mean_{param}") + 10)

        # Test truncate
        truncate_at = 10
        testh.truncate(truncate_at)

        self.assertEqual(testh.states.shape[1], truncate_at)
        self.assertEqual(testh.accept.shape[1], truncate_at)
        self.assertEqual(testh.loglikelihood.shape[1], truncate_at)

    def test_extend(self):
        testh = History(self.num_iters, self.dummy_names)
        # for param in self.dummy_names:
        #     setattr(self.tasth, param, getattr(self.tasth, param) + 1)
        #     setattr(self.tasth, f"mean_{param}", getattr(self.tasth, f"mean_{param}") + 10)

        # Test extend from 20 iters to 19 iters, which should result in a contraction
        extend_to = 19
        testh.extend(extend_to)
        self.assertEqual(testh.states.shape[1], extend_to)
        self.assertEqual(testh.accept.shape[1], extend_to)
        self.assertEqual(testh.loglikelihood.shape[1], extend_to)

        # Test extend from 20 iters to 20 iters, which should result in no changes
        self.setUp()
        extend_to = 20
        testh.extend(extend_to)
        self.assertEqual(testh.states.shape[1], self.num_iters)
        self.assertEqual(testh.accept.shape[1], self.num_iters)
        self.assertEqual(testh.loglikelihood.shape[1], self.num_iters)

        # Test extend from 20 iters to 100 iters
        self.setUp()
        extend_to = 100
        testh.extend(extend_to)
        self.assertEqual(testh.states.shape[1], extend_to)
        self.assertEqual(testh.accept.shape[1], extend_to)
        self.assertEqual(testh.loglikelihood.shape[1], extend_to)

    def test_update(self):
        testh = History(self.num_iters, self.dummy_names)
        k = 0
        testh.states[self.indexes["c"], k] = 50

        # Should split the states array into new attributes for each parameter
        testh.update(self.dummy_names)

        self.assertEqual(testh.mean_c[k], 50)
        self.assertEqual(np.sum(testh.mean_c), 50)
        self.assertEqual(np.sum(testh.mean_a), 0)

if __name__ == "__main__":
    t = TestUtils()
    t.setUp()
    t.test_extend()
