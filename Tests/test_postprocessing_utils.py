import unittest
import numpy as np
import os
import sys
sys.path.append(os.path.join("Visualization"))

from postprocessing_utils import recommend_logscale, calc_contours, fetch_param
from postprocessing_utils import ASJD, ESS, binned_stderr
from postprocessing_utils import load_all_accepted, package_all_accepted
from secondary_parameters import LI_tau_eff, LI_tau_srh, HI_tau_srh

class TestUtils(unittest.TestCase):
    
    def test_rec_logscale(self):
        # Does it work on arbitrary named parameters?
        # Missing parameters?
        do_log = {"a":1,
                  "b":True,
                  "c":0,
                  "d":False,}
        
        self.assertEqual(recommend_logscale("a", do_log), True)
        self.assertEqual(recommend_logscale("b", do_log), True)
        self.assertEqual(recommend_logscale("c", do_log), False)
        self.assertEqual(recommend_logscale("d", do_log), False)
        self.assertEqual(recommend_logscale("not in do_log", do_log), False)
        
        # Preference testing Sf/Sb relationships
        # "Log if any log"
        do_log = {"Sf":1,"Sb":1}
        self.assertEqual(recommend_logscale("Sf+Sb", do_log),1) 
        
        do_log = {"Sf":0,"Sb":1}
        self.assertEqual(recommend_logscale("Sf+Sb", do_log),1) 
        
        do_log = {"Sf":1,"Sb":0}
        self.assertEqual(recommend_logscale("Sf+Sb", do_log),1) 
        
        do_log = {"Sf":0,"Sb":0}
        self.assertEqual(recommend_logscale("Sf+Sb", do_log),0) 
        
        # Preference testing mu relationships
        do_log = {"mu_n":1, "mu_p":1}
        self.assertEqual(recommend_logscale("mu_eff", do_log),1) 
        
        do_log = {"mu_n":0, "mu_p":1}
        self.assertEqual(recommend_logscale("mu_eff", do_log),1) 
        
        do_log = {"mu_n":1, "mu_p":0}
        self.assertEqual(recommend_logscale("mu_eff", do_log),1) 
        
        do_log = {"mu_n":0, "mu_p":0}
        self.assertEqual(recommend_logscale("mu_eff", do_log),0) 
        
        # Preference testing tau_eff relationships
        do_log = {"tauN":1, "Sf":1,"Sb":1}
        self.assertEqual(recommend_logscale("tau_eff", do_log),1) 
        
        do_log = {"tauN":1,"Sf":0,"Sb":0}
        self.assertEqual(recommend_logscale("tau_eff", do_log),1) 
        
        do_log = {"tauN":0,"Sf":1,"Sb":1}
        self.assertEqual(recommend_logscale("tau_eff", do_log),1) 
        
        do_log = {"tauN":0,"Sf":0,"Sb":0}
        self.assertEqual(recommend_logscale("tau_eff", do_log),0) 
        
        # Preference testing HI_tau_eff relationships
        do_log = {"tauN":1, "tauP":0, "Sf":1,"Sb":1}
        self.assertEqual(recommend_logscale("HI_tau_srh", do_log),1) 
        
        do_log = {"tauN":1, "tauP":1, "Sf":1,"Sb":1}
        self.assertEqual(recommend_logscale("HI_tau_srh", do_log),1) 
        
        do_log = {"tauN":1, "tauP":0, "Sf":0,"Sb":0}
        self.assertEqual(recommend_logscale("HI_tau_srh", do_log),1) 
        
        do_log = {"tauN":0, "tauP":0, "Sf":1,"Sb":1}
        self.assertEqual(recommend_logscale("HI_tau_srh", do_log),1) 
        
        do_log = {"tauN":0, "tauP":1, "Sf":1,"Sb":1}
        self.assertEqual(recommend_logscale("HI_tau_srh", do_log),1) 
        
        do_log = {"tauN":0, "tauP":0, "Sf":0,"Sb":0}
        self.assertEqual(recommend_logscale("HI_tau_srh", do_log),0) 
        
        # PReference testing tn/tp
        do_log = {"tauN":1,"tauP":1}
        self.assertEqual(recommend_logscale("tauN+tauP", do_log),1) 
        
        do_log = {"tauN":0,"tauP":1}
        self.assertEqual(recommend_logscale("tauN+tauP", do_log),1) 
        
        do_log = {"tauN":1,"tauP":0}
        self.assertEqual(recommend_logscale("tauN+tauP", do_log),1) 
        
        do_log = {"tauN":0,"tauP":0}
        self.assertEqual(recommend_logscale("tauN+tauP", do_log),0) 
        
        # Preference testing Cn/Cp
        do_log = {"Cn":1,"Cp":1}
        self.assertEqual(recommend_logscale("Cn+Cp", do_log),1) 

        do_log = {"Cn":0,"Cp":1}
        self.assertEqual(recommend_logscale("Cn+Cp", do_log),1) 
        
        do_log = {"Cn":1,"Cp":0}
        self.assertEqual(recommend_logscale("Cn+Cp", do_log),1) 
        
        do_log = {"Cn":0,"Cp":0}
        self.assertEqual(recommend_logscale("Cn+Cp", do_log),0) 
        return
    
    def test_calc_contours(self):
        size = 1000
        x = np.linspace(0,100, size)
        y = np.linspace(0,100, size)
        cx, cy = np.meshgrid(x, y)
        clevels = [1,2,3,4,5]
        
        # Default case
        with self.assertRaises(NotImplementedError):
            calc_contours(x, y, clevels, ("no contours here", "none here either"))
            
        # Sf + Sb works?
        expected = (cx, cy, cx+cy, clevels)
        output = calc_contours(x, y, clevels, ("Sf", "Sb"), size=size)
        self.assertEqual(len(output), len(expected))
        for i in range(len(output)):
            self.assertTrue(np.array_equal(output[i], expected[i]), msg=f"Output {i} mismatch")
            
        # Sf + Sb should be commutative
        cx, cy = np.meshgrid(y, x)
        expected = (cx, cy, cx+cy, clevels)
        output = calc_contours(y, x, clevels, ("Sf", "Sb"), size=size)
        for i in range(len(output)):
            self.assertTrue(np.array_equal(output[i], expected[i]), msg=f"Output {i} mismatch")
               
        # How about over log space?
        x = np.geomspace(0.1,100, size)
        y = np.geomspace(0.1,100, size)
        logx = True
        logy = True
        cx, cy = np.meshgrid(x, y)
        expected = (cx, cy, cx+cy, clevels)
        output = calc_contours(x, y, clevels, ("Sf", "Sb"), size=size, do_logx=logx, do_logy=logy)
        for i in range(len(output)):
            self.assertTrue(np.array_equal(output[i], expected[i]), msg=f"Output {i} mismatch")
            
        # Ambipolar mu works?
        size = 1000
        x = np.geomspace(1e-1,100, size)
        y = np.geomspace(1e-1,100, size)
        logx = True
        logy = True
        cx, cy = np.meshgrid(x, y)
        clevels = [1e-5, 1e-4, 1e-3]
        expected = (cx, cy, 2 / (cx**-1 + cy**-1), clevels)
        output = calc_contours(x, y, clevels, ("mu_n", "mu_p"), size=size, do_logx=logx, do_logy=logy)
        self.assertEqual(len(output), len(expected))
        for i in range(len(output)):
            self.assertTrue(np.array_equal(output[i], expected[i]), msg=f"Output {i} mismatch Expected {expected[i]}\n Output {output[i]}")
        
        return
    
    
    def test_fetch_param(self):
        class dummy_MS():
            def __init__(self):
                return
            
        class dummy_H():
            def __init__(self):
                return
            
        MS = dummy_MS()
        MS.H = dummy_H()
        
        # Test Sf+Sb
        MS.H.Sf = np.zeros(10)
        MS.H.Sb = np.ones(10)
        MS.H.mean_Sf = np.linspace(0,10,11)
        MS.H.mean_Sb = np.linspace(0,10,11)
        
        expected_proposed = MS.H.Sf + MS.H.Sb
        expected_accepted = MS.H.mean_Sf + MS.H.mean_Sb
        output_proposed, output_accepted = fetch_param(MS, "Sf+Sb")
        
        self.assertTrue(np.array_equal(expected_proposed, output_proposed))
        self.assertTrue(np.array_equal(expected_accepted, output_accepted))
        
        # Test tn+tp
        MS.H.tauN = np.zeros(10)
        MS.H.tauP = np.ones(10)
        MS.H.mean_tauN = np.linspace(0,10,11)
        MS.H.mean_tauP = np.linspace(0,10,11)
        
        expected_proposed = MS.H.tauN + MS.H.tauP
        expected_accepted = MS.H.mean_tauN + MS.H.mean_tauP
        output_proposed, output_accepted = fetch_param(MS, "tauN+tauP")
        
        self.assertTrue(np.array_equal(expected_proposed, output_proposed))
        self.assertTrue(np.array_equal(expected_accepted, output_accepted))
        
        # Test Cn+Cp
        MS.H.Cn = np.zeros(10)
        MS.H.Cp = np.ones(10)
        MS.H.mean_Cn = np.linspace(0,10,11)
        MS.H.mean_Cp = np.linspace(0,10,11)
        
        expected_proposed = MS.H.Cn + MS.H.Cp
        expected_accepted = MS.H.mean_Cn + MS.H.mean_Cp
        output_proposed, output_accepted = fetch_param(MS, "Cn+Cp")
        
        self.assertTrue(np.array_equal(expected_proposed, output_proposed))
        self.assertTrue(np.array_equal(expected_accepted, output_accepted))
        
        # Test mu_eff
        MS = dummy_MS()
        MS.H = dummy_H()
        MS.H.mu_n = np.ones(10)
        MS.H.mu_p = np.ones(10)
        MS.H.mean_mu_n = np.geomspace(0.1,100,31) 
        MS.H.mean_mu_p = np.geomspace(0.1,100,31)
        
        expected_proposed = 2 / (MS.H.mu_n**-1 + MS.H.mu_p**-1)
        expected_accepted = 2 / (MS.H.mean_mu_n**-1 + MS.H.mean_mu_p**-1)
        output_proposed, output_accepted = fetch_param(MS, "mu_eff")
        
        self.assertTrue(np.array_equal(expected_proposed, output_proposed))
        self.assertTrue(np.array_equal(expected_accepted, output_accepted))
        
        # Test tau_eff
        MS.H.Sf = np.zeros(10)
        MS.H.Sb = np.ones(10)
        MS.H.mu_n = np.ones(10)
        MS.H.mu_p = np.ones(10)
        MS.H.ks = np.ones(10)
        MS.H.p0 = np.ones(10)
        MS.H.tauN = np.ones(10)
        MS.H.Cp = np.ones(10)
        
        MS.H.mean_Sf = np.geomspace(0.1,10,31)
        MS.H.mean_Sb = np.geomspace(0.1,10,31)
        MS.H.mean_mu_n = np.geomspace(0.1,100,31)
        MS.H.mean_mu_p = np.geomspace(0.1,100,31)
        MS.H.mean_ks =  np.geomspace(0.1,100,31)
        MS.H.mean_p0 = np.geomspace(0.1,100,31)
        MS.H.mean_tauN = np.geomspace(0.1,1000,31)
        MS.H.mean_Cp = np.geomspace(0.1,100,31)
        thickness = 1e3
        
        expected_raw_mu = 2 / (MS.H.mu_n**-1 + MS.H.mu_p**-1)
        expected_mean_mu = 2 / (MS.H.mean_mu_n**-1 + MS.H.mean_mu_p**-1)
        
        expected_proposed = LI_tau_eff(MS.H.ks, MS.H.p0, MS.H.tauN, MS.H.Sf, MS.H.Sb, MS.H.Cp, thickness, expected_raw_mu)
        expected_accepted = LI_tau_eff(MS.H.mean_ks, MS.H.mean_p0, MS.H.mean_tauN, MS.H.mean_Sf, MS.H.mean_Sb, MS.H.mean_Cp, thickness, expected_mean_mu)
        output_proposed, output_accepted = fetch_param(MS, "tau_eff", thickness=thickness)
        self.assertTrue(np.array_equal(expected_proposed, output_proposed))
        self.assertTrue(np.array_equal(expected_accepted, output_accepted))
        
        # Test tau_srh
        MS.H.Sf = np.zeros(10)
        MS.H.Sb = np.ones(10)
        MS.H.mu_n = np.ones(10)
        MS.H.mu_p = np.ones(10)
        MS.H.tauN = np.ones(10)
        
        MS.H.mean_Sf = np.geomspace(0.1,10,31)
        MS.H.mean_Sb = np.geomspace(0.1,10,31)
        MS.H.mean_mu_n = np.geomspace(0.1,100,31)
        MS.H.mean_mu_p = np.geomspace(0.1,100,31)
        MS.H.mean_tauN = np.geomspace(0.1,1000,31)
        thickness = 1e3
        
        expected_raw_mu = 2 / (MS.H.mu_n**-1 + MS.H.mu_p**-1)
        expected_mean_mu = 2 / (MS.H.mean_mu_n**-1 + MS.H.mean_mu_p**-1)
        
        expected_proposed = LI_tau_srh(MS.H.tauN, MS.H.Sf, MS.H.Sb, thickness, expected_raw_mu)
        expected_accepted = LI_tau_srh(MS.H.mean_tauN, MS.H.mean_Sf, MS.H.mean_Sb, thickness, expected_mean_mu)
        output_proposed, output_accepted = fetch_param(MS, "tau_srh", thickness=thickness)
        self.assertTrue(np.array_equal(expected_proposed, output_proposed))
        self.assertTrue(np.array_equal(expected_accepted, output_accepted))
        
        # Test HI_tau_srh
        MS.H.Sf = np.zeros(10)
        MS.H.Sb = np.ones(10)
        MS.H.mu_n = np.ones(10)
        MS.H.mu_p = np.ones(10)
        MS.H.tauN = np.ones(10)
        MS.H.tauP = np.ones(10)
        
        MS.H.mean_Sf = np.geomspace(0.1,10,31)
        MS.H.mean_Sb = np.geomspace(0.1,10,31)
        MS.H.mean_mu_n = np.geomspace(0.1,100,31)
        MS.H.mean_mu_p = np.geomspace(0.1,100,31)
        MS.H.mean_tauN = np.geomspace(0.1,1000,31)
        MS.H.mean_tauP = np.geomspace(0.1,1000,31)
        thickness = 1e3
        
        expected_raw_mu = 2 / (MS.H.mu_n**-1 + MS.H.mu_p**-1)
        expected_mean_mu = 2 / (MS.H.mean_mu_n**-1 + MS.H.mean_mu_p**-1)
        
        expected_proposed = HI_tau_srh(MS.H.tauN, MS.H.tauP, MS.H.Sf, MS.H.Sb, thickness, expected_raw_mu)
        expected_accepted = HI_tau_srh(MS.H.mean_tauN, MS.H.mean_tauP, MS.H.mean_Sf, MS.H.mean_Sb, thickness, expected_mean_mu)
        output_proposed, output_accepted = fetch_param(MS, "HI_tau_srh", thickness=thickness)
        self.assertTrue(np.array_equal(expected_proposed, output_proposed))
        self.assertTrue(np.array_equal(expected_accepted, output_accepted))
        return
    
    def test_ASJD(self):
        test1 = np.linspace(0,100,101)
        test2 = np.linspace(0, 50, 101)
        accepted = np.array([test1, test2])
        accepted = np.expand_dims(accepted, 1)
        accepted = np.repeat(accepted, 3, axis=1)
        
        # Vary the chains - vary the ASJD
        accepted[:,0] *= 0.5
        accepted[:,2] *= 2
        
        diffs = ASJD(accepted)
        np.testing.assert_equal(diffs, [1.25 * 0.5 ** 2, 1.25, 1.25 * 2**2])
        
    def test_ESS(self):
        test1 = np.expand_dims(np.arange(100), 0)
        test2 = np.expand_dims(np.sin(np.arange(100)), 0)
        
        avg_ess = ESS(test1, False, verbose=False)
        self.assertEqual(avg_ess, 0)
        avg_ess = ESS(test2, False, verbose=False)
        self.assertAlmostEqual(avg_ess, 3494.367866841)
        
    def test_binned_stderr(self):
        test_chain = np.arange(100)
        binning = 10
        bins = np.arange(0, len(test_chain), int(binning))[1:]
        
        # Auto promotes 1D to 2D
        expected_out_submeans = [[4.5,14.5,24.5,34.5,44.5,54.5,64.5,74.5,84.5,94.5]]
        

        expected_out_stderr = np.std(expected_out_submeans, ddof=1)
        
        out_subs, out_stderr = binned_stderr(test_chain, bins)
        np.testing.assert_equal(out_subs, expected_out_submeans)
        self.assertEqual(out_stderr, expected_out_stderr)
        
        
        # Disallow uneven binning
        test_chain = np.arange(101)
        with self.assertRaises(ValueError):
            binned_stderr(test_chain, bins)
        
    def test_package_all_accepted(self):
        class dummy_MS():
            def __init__(self):
                return
            
        class dummy_H():
            def __init__(self):
                return
            
        MS = dummy_MS()
        MS.H = dummy_H()
        names = ["mu_n", "p0", "tauN"]
        is_active = {"mu_n":1, "p0":1, "tauN":0}
        do_log = {"mu_n":0, "p0":1, "tauN":0}
        
        mu_n = [1,2,3,4,5]
        p0 = [1e12, 1e13, 1e14, 1e15, 1e16]
        tauN = [511, 511, 511, 511, 511]
        
        MS.H.mean_mu_n = mu_n
        MS.H.mean_p0 = p0
        MS.H.mean_tauN = tauN
        
        expected_out = [mu_n, np.log10(p0)]
        
        out = package_all_accepted(MS, names, is_active, do_log)
        np.testing.assert_equal(out, expected_out)
        
    def test_load_all_accepted(self):
        # Deprecated
        path = os.path.join("Tests", "testfiles")
        names = ["mu_n", "p0", "tauN"]
        is_active = {"mu_n":1, "p0":1, "tauN":0}
        do_log = {"mu_n":0, "p0":1, "tauN":0}
        allll = load_all_accepted(path, names, is_active, do_log)
        
        expected = np.array([[20] * 10, [12.36172784, 12.36172784, 12.36172784, 12.36172784, 12.36172784,
                                         12.36172784, 11.85057902, 11.85057902, 11.85057902, 11.85057902]])
        
        np.testing.assert_almost_equal(allll, expected)
        
if __name__ == "__main__":
    t = TestUtils()
    t.test_ESS()