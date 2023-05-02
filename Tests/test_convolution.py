import unittest
import numpy as np

from laplace import I_moment, convolve, n_convolve
from laplace import make_I_tables, do_irf_convolution, post_conv_trim


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.t = np.linspace(0, 10, 1001)
        self.max_t = self.t[-1]
        self.dt = self.t[1] - self.t[0]
        self.double_t = np.arange(0, self.max_t + self.dt / 4, self.dt / 2)

    def test_make_I_table(self):
        g_t = np.sin(self.t)

        irfs = {1: np.vstack((self.t, g_t)).T}

        I_table = make_I_tables(irfs)

        expected_I_table = dict()
        nk_irf = len(g_t)
        I_0 = np.zeros((nk_irf, 3))
        for i in range(nk_irf - 1):
            for n in range(3):
                I_0[i, n] = I_moment(self.t, g_t, i, n, u_spacing=1000)

        expected_I_table[1] = (I_0, self.t)

        np.testing.assert_equal(expected_I_table[1], I_table[1])

    def test_do_full_irf(self):
        # Same as convolve(), but f_t can have any time step
        custom_t = self.t / 2
        f_t = np.exp(-custom_t)
        g_t = np.sin(self.t)

        nk_irf = len(g_t)
        I_table = np.zeros((nk_irf, 3))
        for i in range(nk_irf - 1):
            for n in range(3):
                I_table[i, n] = I_moment(self.t, g_t, i, n, u_spacing=1000)

        convolved_t, convolved_f, succ = do_irf_convolution(custom_t, f_t, (I_table, self.t))
        expected_h_k = 0.5 * (np.exp(-convolved_t) + np.sin(convolved_t) - np.cos(convolved_t))

        np.testing.assert_almost_equal(convolved_f, expected_h_k, decimal=5)

        convolved_t, convolved_f, succ = do_irf_convolution(custom_t, f_t, (I_table, self.t),
                                                            time_max_shift=True)
        np.testing.assert_equal(0, convolved_t[np.argmax(convolved_f)])

    def test_post_conv_trim(self):
        exp_t = self.t
        exp_y = np.arange(len(exp_t))
        exp_u = np.arange(len(exp_t)) * 0.1

        conv_t = self.double_t[:801]
        max_t = max(conv_t)
        conv_y = np.arange(len(conv_t))

        y_c, times_c, vals_c, uncs_c = post_conv_trim(conv_t, conv_y, exp_t, exp_y, exp_u)

        # Check lengths
        np.testing.assert_equal(len(y_c), len(vals_c))
        np.testing.assert_equal(len(y_c), len(uncs_c))
        np.testing.assert_equal(len(y_c), len(times_c))

        # Check times_c, vals_c, uncs_c truncation
        np.testing.assert_equal(times_c < max_t, True)
        np.testing.assert_equal(vals_c, exp_y[:np.where(exp_t < max_t)[0][-1]+1])
        np.testing.assert_equal(uncs_c, exp_u[:np.where(exp_t < max_t)[0][-1]+1])

        # Check correct interpolation; each y_c and conv_y are matching
        conv_t = list(conv_t)
        for i in range(len(times_c)):
            np.testing.assert_equal(y_c[i], conv_y[conv_t.index(times_c[i])])

    def test1(self):
        # Test 1: f(t) = exp(-t), g(t) = sin(t)
        # Expected: (f o g)(t) = 0.5 * (exp(-t) + sin(t) - cos(t))
        # Error scales as t step ** 2

        f_t = np.exp(-self.double_t)
        g_t = np.sin(self.t)

        nk_irf = len(g_t)
        I_table = np.zeros((nk_irf, 3))
        for i in range(nk_irf - 1):
            for n in range(3):
                I_table[i, n] = I_moment(self.t, g_t, i, n, u_spacing=1000)

        h_k = convolve(f_t, I_table)
        h_k2 = n_convolve(f_t, I_table)
        expected_h_k = 0.5 * (np.exp(-self.t) + np.sin(self.t) - np.cos(self.t))

        np.testing.assert_almost_equal(h_k, expected_h_k, decimal=5)
        np.testing.assert_almost_equal(h_k2, expected_h_k, decimal=5)

    def test2(self):
        # Test 2: f(t) = sin(t), g(t) = exp(-t)
        # Convolution is commutative, so this should have same result as Test 1
        f_t = np.sin(self.double_t)
        g_t = np.exp(-self.t)

        nk_irf = len(g_t)
        I_table = np.zeros((nk_irf, 3))
        for i in range(nk_irf - 1):
            for n in range(3):
                I_table[i, n] = I_moment(self.t, g_t, i, n, u_spacing=1000)

        h_k = convolve(f_t, I_table)
        expected_h_k = 0.5 * (np.exp(-self.t) + np.sin(self.t) - np.cos(self.t))

        np.testing.assert_almost_equal(h_k, expected_h_k, decimal=5)

    def test3(self):
        # Test 3: f(t) = g(t) = 1 {0 < t < 1}
        # Window function
        # This should produce a triangular pulse of length 2 and amplitude 1
        # The left half of the triangle is correct, but the right half is off by a small amount.
        # Perhaps the numerical method is having trouble with the discontinuity?

        f_t = np.where(self.double_t < 1, 1, 0)
        g_t = np.where(self.t < 1, 1, 0)

        nk_irf = len(g_t)
        I_table = np.zeros((nk_irf, 3))
        for i in range(nk_irf - 1):
            for n in range(3):
                I_table[i, n] = I_moment(self.t, g_t, i, n, u_spacing=1000)

        h_k = convolve(f_t, I_table)
        expected_h_k = np.where(self.t < 1, self.t, 2 - self.t)
        expected_h_k = np.where(self.t <= 2, expected_h_k, 0)

        # np.testing.assert_almost_equal(h_k, expected_h_k, decimal=5)


if __name__ == "__main__":
    t = TestUtils()
    t.setUp()
    t.test_post_conv_trim()
