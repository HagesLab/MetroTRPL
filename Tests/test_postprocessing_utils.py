import unittest
import numpy as np
import os
import shutil

from postprocessing_utils import recommend_logscale, calc_contours, fetch_param, fetch
from postprocessing_utils import ASJD, ESS, binned_stderr
from postprocessing_utils import load_all_accepted
from secondary_parameters import LI_tau_eff
import math

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
    
    def test_fetch(self):
        tests = {"Test3":(["mu_n", "mu_p"], "mu_eff"),
                 "Test4":(["Sf", "Sb"], "Sf+Sb"),
                 "Test5":(["Sf", "Sb", "tauN", "mu_n", "mu_p", "ks", "p0"], "tau_eff")}
        # Sf+Sb
        for testname, testitems in tests.items():
            if not os.path.exists(testname):
                os.mkdir(testname)
            
            path = testname
            params = testitems[0]
            expected_raw = {}
            expected_mean = {}
            for param in params:
                np.save(os.path.join(path, f"{param}.npy"), np.zeros(10))
                np.save(os.path.join(path, f"mean_{param}.npy"), np.linspace(0,10,11))
                expected_raw[param] = np.zeros(10)
                expected_mean[param] = np.linspace(0,10,11)
    
            which_param = testitems[1]
            
            output_raw, output_mean = fetch(path, which_param)
            
            shutil.rmtree(path)
            
            for param in expected_raw:
                self.assertTrue(np.array_equal(expected_raw[param], output_raw[param]), msg=param)
                self.assertTrue(np.array_equal(expected_mean[param], output_mean[param]), msg=param)
        
        with self.assertRaises(KeyError):
            fetch("any path whatsoever", "not an add_param")
        
        return
    
    def test_fetch_param(self):
        # Test Sf+Sb
        raw = {"Sf":np.zeros(10), "Sb":np.ones(10)}
        mean = {"Sf":np.linspace(0,10,11), "Sb":np.linspace(0,10,11)}
        
        expected_proposed = raw["Sf"] + raw["Sb"]
        expected_accepted = mean["Sf"] + mean["Sb"]
        output_proposed, output_accepted = fetch_param(raw, mean, "Sf+Sb")
        
        self.assertTrue(np.array_equal(expected_proposed, output_proposed))
        self.assertTrue(np.array_equal(expected_accepted, output_accepted))
        
        # Test mu_eff
        raw = {"mu_n":np.ones(10), "mu_p":np.ones(10)}
        mean = {"mu_n":np.geomspace(0.1,100,31), "mu_p":np.geomspace(0.1,100,31)}
        
        expected_proposed = 2 / (raw["mu_n"]**-1 + raw["mu_p"]**-1)
        expected_accepted = 2 / (mean["mu_n"]**-1 + mean["mu_p"]**-1)
        output_proposed, output_accepted = fetch_param(raw, mean, "mu_eff")
        
        self.assertTrue(np.array_equal(expected_proposed, output_proposed))
        self.assertTrue(np.array_equal(expected_accepted, output_accepted))
        
        # Test tau_eff
        raw = {"Sf":np.zeros(10), "Sb":np.ones(10),
               "mu_n":np.ones(10), "mu_p":np.ones(10),
               "ks": np.ones(10), "p0":np.ones(10),
               "tauN":np.ones(10)}
        mean = {"Sf":np.geomspace(0.1,10,31), "Sb":np.geomspace(0.1,10,31),
               "mu_n":np.geomspace(0.1,100,31), "mu_p":np.geomspace(0.1,100,31),
               "ks": np.geomspace(0.1,100,31), "p0":np.geomspace(0.1,100,31),
               "tauN":np.geomspace(0.1,1000,31)}
        thickness = 1e3
        
        expected_raw_mu = 2 / (raw["mu_n"]**-1 + raw["mu_p"]**-1)
        expected_mean_mu = 2 / (mean["mu_n"]**-1 + mean["mu_p"]**-1)
        
        expected_proposed = LI_tau_eff(raw["ks"], raw["p0"], raw["tauN"], raw["Sf"], raw["Sb"], thickness, expected_raw_mu)
        expected_accepted = LI_tau_eff(mean["ks"], mean["p0"], mean["tauN"], mean["Sf"], mean["Sb"], thickness, expected_mean_mu)
        output_proposed, output_accepted = fetch_param(raw, mean, "tau_eff", thickness=thickness)
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
        
        diffs = ASJD(accepted, [None, None])
        np.testing.assert_equal(diffs, [1.25 * 0.5 ** 2, 1.25, 1.25 * 2**2])
        
    def test_ESS(self):
        test1 = np.expand_dims(np.arange(100), 0)
        test2 = np.expand_dims(np.sin(np.arange(100)), 0)
        
        avg_ess = ESS(test1, [None, None], do_log=False, verbose=False)
        self.assertTrue(math.isnan(avg_ess))
        avg_ess = ESS(test2, [None, None], do_log=False, verbose=False)
        self.assertAlmostEqual(avg_ess, 3494.367866841)
        
    def test_binned_stderr(self):
        test_chain = np.arange(100)
        binning = 10
        bins = np.arange(0, len(test_chain), int(binning))[1:]
        
        expected_out_submeans = [4.5,14.5,24.5,34.5,44.5,54.5,64.5,74.5,84.5,94.5]
        
        # There should be 10 subgroups - hence sqrt(10)
        expected_out_stderr = np.std(expected_out_submeans, ddof=1) / np.sqrt(10)
        
        out_subs, out_stderr = binned_stderr(test_chain, bins)
        np.testing.assert_equal(out_subs, expected_out_submeans)
        self.assertEqual(out_stderr, expected_out_stderr)
        
        
        # Disallow uneven binning
        test_chain = np.arange(101)
        with self.assertRaises(ValueError):
            binned_stderr(test_chain, bins)
        
        
    def test_load_all_accepted(self):
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
    t.test_load_all_accepted()