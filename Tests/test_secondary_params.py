import unittest
import sys
import os
import numpy as np
sys.path.append(os.path.join("GUI"))
from secondary_parameters import SecondaryParameters
class TestUtils(unittest.TestCase):

    def setUp(self):
        self.SP = SecondaryParameters()

    def test_tau_rad(self):
        p = {"ks": 1e-11, "p0": 1e15}
        self.assertEqual(self.SP.t_rad(p), 1e5)

        p = {"ks": 0, "p0": 1}
        with self.assertRaises(ZeroDivisionError):
            self.SP.t_rad(p)

        with np.errstate(divide='ignore'):
            p = {"ks": np.ones(1), "p0": np.zeros(1)}
            self.assertEqual(self.SP.t_rad(p), np.full(1, np.inf))

        return

    def test_tau_auger(self):
        p = {"Cp": 1e-29, "p0": 1e15}
        self.assertEqual(self.SP.t_auger(p), 1e8)

        # Allow "infinite" lifetime for np arrays
        p = {"Cp": 0, "p0": 1}
        with self.assertRaises(ZeroDivisionError):
            self.SP.t_auger(p)

        with np.errstate(divide='ignore'):
            p = {"Cp": np.ones(1), "p0": np.zeros(1)}
            self.assertEqual(self.SP.t_auger(p), np.full(1, np.inf))

    def test_s_eff(self):
        p = {"Sf": 1, "Sb": 0}
        self.assertEqual(self.SP.s_eff(p), 1)

        p = {"Sf": np.ones(1), "Sb": np.zeros(1)}
        self.assertEqual(self.SP.s_eff(p), np.ones(1))
        return

    def test_mu_eff(self):
        p = {"mu_n": 12, "mu_p": 60}
        self.assertEqual(self.SP.mu_eff(p), 20)

        p = {"mu_n": 20, "mu_p": 20}
        self.assertEqual(self.SP.mu_eff(p), 20)

        p = {"mu_n": np.zeros(1), "mu_p": np.ones(1)}
        with np.errstate(divide='ignore'):
            self.assertEqual(self.SP.mu_eff(p), np.zeros(1))
        return

    def test_epsilon(self):
        p = {"lambda": 2}
        self.assertEqual(self.SP.epsilon(p), 0.5)

        p = {"lambda": np.zeros(1)}
        with np.errstate(divide='ignore'):
            self.assertEqual(self.SP.epsilon(p), np.full(1, np.inf))
        return

    def test_LI_tau_eff(self):
        # tau_n only = tau_n
        p = {"ks": np.zeros(1), "p0": 1, "tauN": 450.0, "Sf": np.zeros(1), "Sb": 0,
             "Cp": np.zeros(1), "thickness": 1, "mu_n": np.zeros(1), "mu_p": np.zeros(1)}
        with np.errstate(divide='ignore'):
            self.assertEqual(self.SP.li_tau_eff(p), 450)

        # Rad only = 1/(B*p0)
        p = {"ks": np.ones(1), "p0": 1e9, "tauN": np.full(1, np.inf), "Sf": np.zeros(1), "Sb": 0,
             "Cp": np.zeros(1), "thickness": 1, "mu_n": np.zeros(1), "mu_p": np.zeros(1)}
        with np.errstate(divide='ignore'):
            self.assertAlmostEqual(self.SP.li_tau_eff(p)[0], 1)

        # Surf only = d / (Sf+Sb)
        p = {"ks": np.zeros(1), "p0": 1, "tauN": np.full(1, np.inf), "Sf": np.ones(1), "Sb": 0,
             "Cp": np.zeros(1), "thickness": 1, "mu_n": np.full(1, np.inf), "mu_p": np.full(1, np.inf)}
        with np.errstate(divide='ignore'):
            self.assertAlmostEqual(self.SP.li_tau_eff(p)[0], 100)

        # Diff only = (d^2 / (2pi*kT*mu)) if Sf ~ Sb
        p = {"ks": np.zeros(1), "p0": 1, "tauN": np.full(1, np.inf), "Sf": np.full(1, np.inf), "Sb": 0,
             "Cp": np.zeros(1), "thickness": 1, "mu_n": np.ones(1), "mu_p": np.ones(1)}
        with np.errstate(divide='ignore'):
            self.assertAlmostEqual(self.SP.li_tau_eff(p)[0], 3.9424585074839604e-05)

        # Realism
        p = {"ks": np.full(1, 4.8e-11), "p0": 3e15, "tauN": np.full(1, 511.0), "Sf": np.full(1, 10.0), "Sb": 10,
             "Cp": np.zeros(1), "thickness": 2000, "mu_n": np.full(1, 20.0), "mu_p": np.full(1, 20.0)}
        with np.errstate(divide='ignore'):
            self.assertAlmostEqual(self.SP.li_tau_eff(p)[0], 454.36610893)

    def test_LI_tau_srh(self):
        # LI_tau_eff but with only SRH terms
        # tau_n only
        p = {"tauN": 450.0, "Sf": np.zeros(1), "Sb": 0,
             "thickness": 1, "mu_n": np.zeros(1), "mu_p": np.zeros(1)}
        with np.errstate(divide='ignore'):
            self.assertEqual(self.SP.li_tau_srh(p), 450)

        # Surf only
        p = {"tauN": np.full(1, np.inf), "Sf": np.ones(1), "Sb": 0,
             "thickness": 1, "mu_n": np.full(1, np.inf), "mu_p": np.full(1, np.inf)}
        with np.errstate(divide='ignore'):
            self.assertAlmostEqual(self.SP.li_tau_srh(p)[0], 100)

        # Realism
        p = {"tauN": np.full(1, 511.0), "Sf": np.full(1, 10), "Sb": 10,
             "thickness": 2000, "mu_n": np.full(1, 20.0), "mu_p": np.full(1, 20.0)}
        with np.errstate(divide='ignore'):
            self.assertAlmostEqual(self.SP.li_tau_srh(p)[0], 486.1759809086)

    def test_HI_tau_srh(self):
        # HI_tau_srh but with only SRH terms
        # Rad and Auger don't have asymptotic values
        # Bulk only = tau_n + tau_p
        p = {"tauN": 450.0, "tauP": 500.0, "Sf": np.zeros(1), "Sb": 0,
             "thickness": 1, "mu_n": np.zeros(1), "mu_p": np.zeros(1)}
        with np.errstate(divide='ignore'):
            self.assertEqual(self.SP.hi_tau_srh(p), 950)

        # Surf only = 2d/(Sf+Sb)
        p = {"tauN": np.full(1, np.inf), "tauP": np.full(1, np.inf), "Sf": np.ones(1), "Sb": 0,
             "thickness": 1, "mu_n": np.full(1, np.inf), "mu_p": np.full(1, np.inf)}
        with np.errstate(divide='ignore'):
            self.assertAlmostEqual(self.SP.hi_tau_srh(p)[0], 200)

        # Realism
        p = {"tauN": np.full(1, 511.0), "tauP": np.full(1, 871.0), "Sf": np.full(1, 10), "Sb": 10,
             "thickness": 2000, "mu_n": np.full(1, 20.0), "mu_p": np.full(1, 20.0)}
        with np.errstate(divide='ignore'):
            self.assertAlmostEqual(self.SP.hi_tau_srh(p)[0], 1292.7090100)

if __name__ == "__main__":
    tests = TestUtils()
    tests.setUp()
    tests.test_LI_tau_eff()