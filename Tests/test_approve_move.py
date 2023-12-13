import unittest
import sys
sys.path.append("..")

import numpy as np

from sim_utils import EnsembleTemplate
from trial_move_generation import approve_move


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.mock_ensemble = EnsembleTemplate()

    def test_tauNP(self):
        info = {'names': ['tauP', 'tauN', 'somethingelse'],
                'prior_dist': {'tauP': (0.1, np.inf), 'tauN': (0.1, np.inf),
                               'somethingelse': (-np.inf, np.inf)},
                'do_log': {'tauP': 1, 'tauN': 1, 'somethingelse': 1},
                "active": {'tauP': 1, 'tauN': 1, 'somethingelse': 1}}
        param_indexes = {name: info["names"].index(name) for name in info["names"]}
        do_log = np.array([info["do_log"][param] for param in info["names"]], dtype=bool)
        active = np.array([info["active"][name] for name in info["names"]], dtype=bool)
        ensemble_fields = {"do_log": do_log, "active": active, "prior_dist": info["prior_dist"],
                           "_param_indexes": param_indexes, "names": info["names"]}
        # taun, taup must be within 2 OM
        # Accepts new_p as log10
        # [n0, p0, mu_n, mu_p, ks, sf, sb, taun, taup, eps, m]
        new_p = np.log10([511, 511e2, 1])
        self.assertTrue(len(approve_move(new_p, ensemble_fields)) == 0)
        new_p = np.log10([511, 511e2+1,  1])
        self.assertTrue("tn_tp_close" in approve_move(new_p, ensemble_fields))

        # tn, tp size limit
        new_p = np.log10([0.11, 0.11, 1])
        self.assertTrue(len(approve_move(new_p, ensemble_fields)) == 0)
        new_p = np.log10([0.1, 0.11, 1])
        self.assertTrue("tauP_size" in approve_move(new_p, ensemble_fields))
        new_p = np.log10([0.11, 0.1, 1])
        self.assertTrue("tauN_size" in approve_move(new_p, ensemble_fields))
    
    def test_inactive(self):
        # If params are inactive, they should not be checked
        info = {'names': ['tauP', 'tauN', 'somethingelse'],
                'prior_dist': {'tauP': (0.1, np.inf), 'tauN': (0.1, np.inf),
                               'somethingelse': (-np.inf, np.inf)},
                'do_log': {'tauP': 1, 'tauN': 1, 'somethingelse': 1},
                "active": {'tauP': 0, 'tauN': 0, 'somethingelse': 1}}
        param_indexes = {name: info["names"].index(name) for name in info["names"]}
        do_log = np.array([info["do_log"][param] for param in info["names"]], dtype=bool)
        active = np.array([info["active"][name] for name in info["names"]], dtype=bool)
        ensemble_fields = {"do_log": do_log, "active": active, "prior_dist": info["prior_dist"],
                           "_param_indexes": param_indexes, "names": info["names"]}
        new_p = np.log10([0.11, 0.1, 1])
        self.assertTrue(len(approve_move(new_p, ensemble_fields)) == 0)

    def test_nologscale(self):
        # These should still work if p is not logscaled
        info = {'names': ['tauP', 'tauN', 'somethingelse'],
                'prior_dist': {'tauP': (0.1, np.inf), 'tauN': (0.1, np.inf),
                               'somethingelse': (-np.inf, np.inf)},
                'do_log': {'tauP': 0, 'tauN': 0, 'somethingelse': 1},
                "active": {'tauP': 1, 'tauN': 1, 'somethingelse': 1}}
        param_indexes = {name: info["names"].index(name) for name in info["names"]}
        do_log = np.array([info["do_log"][param] for param in info["names"]], dtype=bool)
        active = np.array([info["active"][name] for name in info["names"]], dtype=bool)
        ensemble_fields = {"do_log": do_log, "active": active, "prior_dist": info["prior_dist"],
                           "_param_indexes": param_indexes, "names": info["names"]}
        new_p = np.array([511, 511e2, 1])
        self.assertTrue(len(approve_move(new_p, ensemble_fields)) == 0)
        new_p = np.array([511, 511e2+1,  1])
        self.assertTrue("tn_tp_close" in approve_move(new_p, ensemble_fields))

        new_p = np.array([0.11, 0.11, 1])
        self.assertTrue(len(approve_move(new_p, ensemble_fields)) == 0)
        new_p = np.array([0.1, 0.11, 1])
        self.assertTrue("tauP_size" in approve_move(new_p, ensemble_fields))
        new_p = np.array([0.11, 0.1, 1])
        self.assertTrue("tauN_size" in approve_move(new_p, ensemble_fields))

    def test_musurface(self):
        # Check mu_n, mu_p, Sf, and Sb size limits
        info = {"names": ["mu_n", "mu_p", "Sf", "Sb"],
                'prior_dist': {'mu_n': (0.1, 1e6), 'mu_p': (0.1, 1e6),
                               'Sf': (0, 1e7), 'Sb': (0, 1e7)},
                'do_log': {"mu_n": 1, "mu_p": 1, "Sf": 1, "Sb": 1},
                "active": {"mu_n": 1, "mu_p": 1, "Sf": 1, "Sb": 1}}
        param_indexes = {name: info["names"].index(name) for name in info["names"]}
        do_log = np.array([info["do_log"][param] for param in info["names"]], dtype=bool)
        active = np.array([info["active"][name] for name in info["names"]], dtype=bool)
        ensemble_fields = {"do_log": do_log, "active": active, "prior_dist": info["prior_dist"],
                           "_param_indexes": param_indexes, "names": info["names"]}
        new_p = np.log10([1e6-1, 1e6-1, 1e7-1, 1e7-1])
        self.assertTrue(len(approve_move(new_p, ensemble_fields)) == 0)

        new_p = np.log10([1e6, 1e6-1, 1e7-1, 1e7-1])
        self.assertTrue("mu_n_size" in approve_move(new_p, ensemble_fields))
        new_p = np.log10([1e6-1, 1e6, 1e7-1, 1e7-1])
        self.assertTrue("mu_p_size" in approve_move(new_p, ensemble_fields))
        new_p = np.log10([1e6-1, 1e6-1, 1e7, 1e7-1])
        self.assertTrue("Sf_size" in approve_move(new_p, ensemble_fields))
        new_p = np.log10([1e6-1, 1e6-1, 1e7-1, 1e7])
        self.assertTrue("Sb_size" in approve_move(new_p, ensemble_fields))

    def test_musurface_nolog(self):
        # These should still work if p is not logscaled
        info = {"names": ["mu_n", "mu_p", "Sf", "Sb"],
                'prior_dist': {'mu_n': (0.1, 1e6), 'mu_p': (0.1, 1e6),
                               'Sf': (0, 1e7), 'Sb': (0, 1e7)},
                'do_log': {"mu_n": 0, "mu_p": 0, "Sf": 0, "Sb": 0},
                "active": {"mu_n": 1, "mu_p": 1, "Sf": 1, "Sb": 1}}
        param_indexes = {name: info["names"].index(name) for name in info["names"]}
        do_log = np.array([info["do_log"][param] for param in info["names"]], dtype=bool)
        active = np.array([info["active"][name] for name in info["names"]], dtype=bool)
        ensemble_fields = {"do_log": do_log, "active": active, "prior_dist": info["prior_dist"],
                           "_param_indexes": param_indexes, "names": info["names"]}
        new_p = np.array([1e6-1, 1e6-1, 1e7-1, 1e7-1])
        self.assertTrue(len(approve_move(new_p, ensemble_fields)) == 0)

        new_p = np.array([1e6, 1e6-1, 1e7-1, 1e7-1])
        self.assertTrue("mu_n_size" in approve_move(new_p, ensemble_fields))
        new_p = np.array([1e6-1, 1e6, 1e7-1, 1e7-1])
        self.assertTrue("mu_p_size" in approve_move(new_p, ensemble_fields))
        new_p = np.array([1e6-1, 1e6-1, 1e7, 1e7-1])
        self.assertTrue("Sf_size" in approve_move(new_p, ensemble_fields))
        new_p = np.array([1e6-1, 1e6-1, 1e7-1, 1e7])
        self.assertTrue("Sb_size" in approve_move(new_p, ensemble_fields))

    def test_highorderrec(self):
        # Check ks, Cn, Cp size limits
        info = {"names": ["ks", "Cn", "Cp"],
                'prior_dist': {'ks': (0, 1e-7), 'Cn': (0, 1e-21),
                               'Cp': (0, 1e-21)},
                "do_log": {"ks": 1, "Cn": 1, "Cp": 1},
                "active": {"ks": 1, "Cn": 1, "Cp": 1}}
        param_indexes = {name: info["names"].index(name) for name in info["names"]}
        do_log = np.array([info["do_log"][param] for param in info["names"]], dtype=bool)
        active = np.array([info["active"][name] for name in info["names"]], dtype=bool)
        ensemble_fields = {"do_log": do_log, "active": active, "prior_dist": info["prior_dist"],
                           "_param_indexes": param_indexes, "names": info["names"]}
        new_p = np.log10([1e-7*0.9, 1e-21*0.9, 1e-21*0.9])
        self.assertTrue(len(approve_move(new_p, ensemble_fields)) == 0)

        new_p = np.log10([1e-7, 1e-21*0.9, 1e-21*0.9])
        self.assertTrue("ks_size" in approve_move(new_p, ensemble_fields))
        new_p = np.log10([1e-7*0.9, 1e-21, 1e-21*0.9])
        self.assertTrue("Cn_size" in approve_move(new_p, ensemble_fields))
        new_p = np.log10([1e-7*0.9, 1e-21*0.9, 1e-21])
        self.assertTrue("Cp_size" in approve_move(new_p, ensemble_fields))

    def test_highorderrec_nolog(self):
        # Should work without log
        info = {"names": ["ks", "Cn", "Cp"],
                'prior_dist': {'ks': (0, 1e-7), 'Cn': (0, 1e-21),
                               'Cp': (0, 1e-21)},
                "do_log": {"ks": 0, "Cn": 0, "Cp": 0},
                "active": {"ks": 1, "Cn": 1, "Cp": 1}}
        param_indexes = {name: info["names"].index(name) for name in info["names"]}
        do_log = np.array([info["do_log"][param] for param in info["names"]], dtype=bool)
        active = np.array([info["active"][name] for name in info["names"]], dtype=bool)
        ensemble_fields = {"do_log": do_log, "active": active, "prior_dist": info["prior_dist"],
                           "_param_indexes": param_indexes, "names": info["names"]}
        new_p = np.array([1e-7*0.9, 1e-21*0.9, 1e-21*0.9])
        self.assertTrue(len(approve_move(new_p, ensemble_fields)) == 0)

        new_p = np.array([1e-7, 1e-21*0.9, 1e-21*0.9])
        self.assertTrue("ks_size" in approve_move(new_p, ensemble_fields))
        new_p = np.array([1e-7*0.9, 1e-21, 1e-21*0.9])
        self.assertTrue("Cn_size" in approve_move(new_p, ensemble_fields))
        new_p = np.array([1e-7*0.9, 1e-21*0.9, 1e-21])
        self.assertTrue("Cp_size" in approve_move(new_p, ensemble_fields))

    def test_p0(self):
        # Check p0, which has a size limit and must also be larger than n0
        info = {"names": ["n0", "p0"],
                'prior_dist': {'n0': (0, 1e19), 'p0': (0, 1e19)},
                "do_log": {"n0": 1, "p0": 1},
                "active": {"n0": 1, "p0": 1}}
        param_indexes = {name: info["names"].index(name) for name in info["names"]}
        do_log = np.array([info["do_log"][param] for param in info["names"]], dtype=bool)
        active = np.array([info["active"][name] for name in info["names"]], dtype=bool)
        ensemble_fields = {"do_log": do_log, "active": active, "prior_dist": info["prior_dist"],
                           "_param_indexes": param_indexes, "names": info["names"]}
        new_p = np.log10([1e19 * 0.8, 1e19 * 0.9])
        self.assertTrue(len(approve_move(new_p, ensemble_fields)) == 0)

        new_p = np.log10([1e19 * 0.8, 1e19])
        self.assertTrue("p0_size" in approve_move(new_p, ensemble_fields))
        new_p = np.log10([1e19, 1e19 * 0.9])
        self.assertTrue("p0_greater" in approve_move(new_p, ensemble_fields))

    def test_p0_nolog(self):
        # Should work without log
        info = {"names": ["n0", "p0"],
                'prior_dist': {'n0': (0, 1e19), 'p0': (0, 1e19)},
                "do_log": {"n0": 0, "p0": 0},
                "active": {"n0": 1, "p0": 1}}
        param_indexes = {name: info["names"].index(name) for name in info["names"]}
        do_log = np.array([info["do_log"][param] for param in info["names"]], dtype=bool)
        active = np.array([info["active"][name] for name in info["names"]], dtype=bool)
        ensemble_fields = {"do_log": do_log, "active": active, "prior_dist": info["prior_dist"],
                           "_param_indexes": param_indexes, "names": info["names"]}
        new_p = np.array([1e19 * 0.8, 1e19 * 0.9])
        self.assertTrue(len(approve_move(new_p, ensemble_fields)) == 0)

        new_p = np.array([1e19 * 0.8, 1e19])  # p0 too large
        self.assertTrue("p0_size" in approve_move(new_p, ensemble_fields))
        new_p = np.array([1e19, 1e19 * 0.9])  # p0 smaller than n0
        self.assertTrue("p0_greater" in approve_move(new_p, ensemble_fields))

    def test_custom_parameter(self):
        info_without_taus = {'names': ['tauQ', 'somethingelse'],
                             "do_log": {'tauQ': 1, 'somethingelse': 1},
                             "active": {'tauQ': 1, 'somethingelse': 1},
                             'prior_dist': {'tauQ': (-np.inf, np.inf),
                                            'somethingelse': (-np.inf, np.inf)}}
        param_indexes = {name: info_without_taus["names"].index(name) for name in info_without_taus["names"]}
        do_log = np.array([info_without_taus["do_log"][param] for param in info_without_taus["names"]], dtype=bool)
        active = np.array([info_without_taus["active"][name] for name in info_without_taus["names"]], dtype=bool)
        ensemble_fields = {"do_log": do_log, "active": active,
                           "prior_dist": info_without_taus["prior_dist"],
                           "_param_indexes": param_indexes, "names": info_without_taus["names"]}
        # No failures if criteria do not cover params
        new_p = np.log10([1, 1e10])
        self.assertTrue(
            len(approve_move(new_p, ensemble_fields)) == 0)
