import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.join("Visualization"))
from secondary_parameters import LI_tau_eff, t_rad, s_eff, mu_eff, epsilon
from secondary_parameters import t_auger, LI_tau_srh, HI_tau_srh
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
    
    def test_tau_auger(self):
        Cp = 1e-29
        p0 = 1e15
        self.assertEqual(t_auger(Cp, p0), 1e8)
        
        # Allow "infinite" lifetime for np arrays
        Cp = 0
        p0 = 1
        with self.assertRaises(ZeroDivisionError):
            t_auger(Cp, p0)
            
        with np.errstate(divide='ignore'):
            Cp = np.ones(1)
            p0 = np.zeros(1)
            self.assertEqual(t_auger(Cp, p0), np.ones(1) * np.inf)
    
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
    
    def test_LI_tau_eff(self):
        # tau_n only = tau_n
        B = np.zeros(1)
        p0 = 1
        tau_n = 450
        Sf = np.zeros(1)
        Sb = 0
        Cp = np.zeros(1)
        thickness = 1
        mu = np.zeros(1)
        with np.errstate(divide='ignore'):
            self.assertEqual(LI_tau_eff(B, p0, tau_n, Sf, Sb, Cp, thickness, mu), 450)
            
        # Rad only = 1/(B*p0)
        B = np.ones(1)
        p0 = 1e9
        tau_n = np.ones(1) * np.inf
        Sf = np.zeros(1)
        Sb = 0
        Cp = np.zeros(1)
        thickness = 1
        mu = np.zeros(1)
        with np.errstate(divide='ignore'):
            self.assertAlmostEqual(LI_tau_eff(B, p0, tau_n, Sf, Sb, Cp, thickness, mu)[0], 1)
            
        # Surf only = d / (Sf+Sb)
        B = np.zeros(1)
        p0 = 1
        tau_n = np.ones(1) * np.inf
        Sf = np.ones(1)
        Sb = 0
        Cp = np.zeros(1)
        thickness = 1
        mu = np.ones(1) * np.inf
        with np.errstate(divide='ignore'):
            self.assertAlmostEqual(LI_tau_eff(B, p0, tau_n, Sf, Sb, Cp, thickness, mu)[0], 100)
            
        # Diff only = (d^2 / (2pi*kT*mu)) if Sf ~ Sb
        B = np.zeros(1)
        p0 = 1
        tau_n = np.ones(1) * np.inf
        Sf = np.ones(1) * np.inf
        Sb = 0
        Cp = np.zeros(1)
        thickness = 1
        mu = np.ones(1)
        with np.errstate(divide='ignore'):
            self.assertAlmostEqual(LI_tau_eff(B, p0, tau_n, Sf, Sb, Cp, thickness, mu)[0], 3.9424585074839604e-05)
            
        # Realism
        B = np.ones(1) * 4.8e-11
        p0 = 3e15
        tau_n = np.ones(1) * 511
        Sf = np.ones(1) * 10
        Sb = 10
        Cp = np.zeros(1)
        thickness = 2000
        mu = np.ones(1) * 20
        with np.errstate(divide='ignore'):
            self.assertAlmostEqual(LI_tau_eff(B, p0, tau_n, Sf, Sb, Cp, thickness, mu)[0], 454.36610893)
            
    def test_LI_tau_srh(self):
        # LI_tau_eff but with only SRH terms
        # tau_n only
        tau_n = 450
        Sf = np.zeros(1)
        Sb = 0
        thickness = 1
        mu = np.zeros(1)
        with np.errstate(divide='ignore'):
            self.assertEqual(LI_tau_srh(tau_n, Sf, Sb, thickness, mu), 450)
            
        # Surf only
        tau_n = np.ones(1) * np.inf
        Sf = np.ones(1)
        Sb = 0
        thickness = 1
        mu = np.ones(1) * np.inf
        with np.errstate(divide='ignore'):
            self.assertAlmostEqual(LI_tau_srh(tau_n, Sf, Sb, thickness, mu)[0], 100)
               
        # Realism
        tau_n = np.ones(1) * 511
        Sf = np.ones(1) * 10
        Sb = 10
        thickness = 2000
        mu = np.ones(1) * 20
        with np.errstate(divide='ignore'):
            self.assertAlmostEqual(LI_tau_srh(tau_n, Sf, Sb, thickness, mu)[0], 486.1759809086)
            
    def test_HI_tau_srh(self):
        # HI_tau_srh but with only SRH terms
        # Rad and Auger don't have asymptotic values
        # Bulk only = tau_n + tau_p
        tau_n = 450
        tau_p = 500
        Sf = np.zeros(1)
        Sb = 0
        thickness = 1
        mu = np.zeros(1)
        with np.errstate(divide='ignore'):
            self.assertEqual(HI_tau_srh(tau_n, tau_p, Sf, Sb, thickness, mu), 950)
            
        # Surf only = 2d/(Sf+Sb)
        tau_n = np.ones(1) * np.inf
        tau_p = np.ones(1) * np.inf
        Sf = np.ones(1)
        Sb = 0
        thickness = 1
        mu = np.ones(1) * np.inf
        with np.errstate(divide='ignore'):
            self.assertAlmostEqual(HI_tau_srh(tau_n, tau_p, Sf, Sb, thickness, mu)[0], 200)
               
        # Realism
        tau_n = np.ones(1) * 511
        tau_p = np.ones(1) * 871
        Sf = np.ones(1) * 10
        Sb = 10
        thickness = 2000
        mu = np.ones(1) * 20
        with np.errstate(divide='ignore'):
            self.assertAlmostEqual(HI_tau_srh(tau_n, tau_p, Sf, Sb, thickness, mu)[0], 1292.7090100)
            