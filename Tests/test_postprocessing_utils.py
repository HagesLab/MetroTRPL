import unittest
import numpy as np
import os
import shutil

from postprocessing_utils import recommend_logscale, calc_contours, fetch_param, fetch
from secondary_parameters import LI_tau_eff

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
                 "Test5":(["Sf", "Sb", "tauN", "mu_n", "mu_p", "B", "p0"], "tau_eff")}
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
               "B": np.ones(10), "p0":np.ones(10),
               "tauN":np.ones(10)}
        mean = {"Sf":np.geomspace(0.1,10,31), "Sb":np.geomspace(0.1,10,31),
               "mu_n":np.geomspace(0.1,100,31), "mu_p":np.geomspace(0.1,100,31),
               "B": np.geomspace(0.1,100,31), "p0":np.geomspace(0.1,100,31),
               "tauN":np.geomspace(0.1,1000,31)}
        thickness = 1e3
        
        expected_raw_mu = 2 / (raw["mu_n"]**-1 + raw["mu_p"]**-1)
        expected_mean_mu = 2 / (mean["mu_n"]**-1 + mean["mu_p"]**-1)
        
        expected_proposed = LI_tau_eff(raw["B"], raw["p0"], raw["tauN"], raw["Sf"], raw["Sb"], thickness, expected_raw_mu)
        expected_accepted = LI_tau_eff(mean["B"], mean["p0"], mean["tauN"], mean["Sf"], mean["Sb"], thickness, expected_mean_mu)
        output_proposed, output_accepted = fetch_param(raw, mean, "tau_eff", thickness=thickness)
        return