import unittest
import numpy as np

from secondary_parameters import LI_tau_eff, t_rad, s_eff, mu_eff, epsilon

class TestUtils(unittest.TestCase):
    
    def test_tau_rad(self):
        B = 1e-11
        p0 = 1e15
        self.assertEqual(t_rad(B, p0), 1e5)
        
        B = 0
        p0 = 1
        with self.assertRaises(ZeroDivisionError):
            t_rad(B, p0)
            
            
        with np.errstate(divide='ignore'):
            B = np.ones(1)
            p0 = np.zeros(1)
            self.assertEqual(t_rad(B, p0), np.ones(1) * np.inf)
            
        return
    
    def test_s_eff(self):
        sf = 1
        sb = 0
        self.assertEqual(s_eff(sf, sb), 1)
        self.assertEqual(s_eff(sb, sf), 1)
        
        sf = np.ones(1)
        sb = np.zeros(1)
        self.assertEqual(s_eff(sb, sf), np.ones(1))
        return
    
    def test_mu_eff(self):
        mu_n = 12
        mu_p = 60
        self.assertEqual(mu_eff(mu_n, mu_p), 20)
        self.assertEqual(mu_eff(mu_p, mu_n), 20)
        
        mu_n = 20
        mu_p = 20
        self.assertEqual(mu_eff(mu_n, mu_p), 20)
        
        mu_n = np.zeros(1)
        mu_p = np.ones(1)
        with np.errstate(divide='ignore'):
            self.assertEqual(mu_eff(mu_n, mu_p), np.zeros(1))
        return
    
    def test_epsilon(self):
        lamb = 2
        self.assertEqual(epsilon(lamb), 0.5)
        
        lamb = np.zeros(1)
        with np.errstate(divide='ignore'):
            self.assertEqual(epsilon(lamb), np.ones(1)*np.inf)
        return
    
    def test_tau_eff(self):
        # tau_n only
        B = np.zeros(1)
        p0 = 1
        tau_n = 450
        Sf = np.zeros(1)
        Sb = 0
        thickness = 1
        mu = np.zeros(1)
        with np.errstate(divide='ignore'):
            self.assertEqual(LI_tau_eff(B, p0, tau_n, Sf, Sb, thickness, mu), 450)
            
        # Rad only
        B = np.ones(1)
        p0 = 1e9
        tau_n = np.ones(1) * np.inf
        Sf = np.zeros(1)
        Sb = 0
        thickness = 1
        mu = np.zeros(1)
        with np.errstate(divide='ignore'):
            self.assertAlmostEqual(LI_tau_eff(B, p0, tau_n, Sf, Sb, thickness, mu)[0], 1)
            
        # Surf only
        B = np.zeros(1)
        p0 = 1
        tau_n = np.ones(1) * np.inf
        Sf = np.ones(1)
        Sb = 0
        thickness = 1
        mu = np.ones(1) * np.inf
        with np.errstate(divide='ignore'):
            self.assertAlmostEqual(LI_tau_eff(B, p0, tau_n, Sf, Sb, thickness, mu)[0], 100)
            
        # Diff only
        B = np.zeros(1)
        p0 = 1
        tau_n = np.ones(1) * np.inf
        Sf = np.ones(1) * np.inf
        Sb = 0
        thickness = 1
        mu = np.ones(1)
        with np.errstate(divide='ignore'):
            self.assertAlmostEqual(LI_tau_eff(B, p0, tau_n, Sf, Sb, thickness, mu)[0], 3.9424585074839604e-05)
            
        # Realism
        B = np.ones(1) * 4.8e-11
        p0 = 3e15
        tau_n = np.ones(1) * 511
        Sf = np.ones(1) * 10
        Sb = 10
        thickness = 2000
        mu = np.ones(1) * 20
        with np.errstate(divide='ignore'):
            self.assertAlmostEqual(LI_tau_eff(B, p0, tau_n, Sf, Sb, thickness, mu)[0], 454.36610893)