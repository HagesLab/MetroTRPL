import unittest
import logging
import sys
sys.path.append("..")

import numpy as np

from sim_utils import EnsembleTemplate
q = 1.0  # [e]
q_C = 1.602e-19  # [C]
kB = 8.61773e-5  # [eV / K]

MIN_HMAX = 0.01


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.mock_ensemble = EnsembleTemplate()
        self.mock_ensemble.logger = logging.getLogger()


    def test_run_iter(self):
        # The bare minimum needed to run a Monte Carlo iteration
        np.random.seed(42)
        Length = [2000, 2000]                            # Length (nm)
        L = [2 ** 7, 2 ** 7]                                # Spatial point
        mtype = ["TRPL", "TRPL"]
        simPar = {"lengths": Length,
                  "nx": L,
                  "meas_types": mtype,
                  "num_meas": 2}

        iniPar = np.array([1e15 * np.ones(L[0]), 1e16 * np.ones(L[1])])

        param_names = ["n0", "p0", "mu_n", "mu_p", "ks", "Cn", "Cp", "Tm",
                       "Sf", "Sb", "tauN", "tauP", "eps", "m"]
        unit_conversions = {"n0": ((1e-7) ** 3), "p0": ((1e-7) ** 3),
                            "mu_n": ((1e7) ** 2) / (1e9), "mu_p": ((1e7) ** 2) / (1e9),
                            "ks": ((1e7) ** 3) / (1e9), "Sf": 1e-2, "Sb": 1e-2}

        # Iterations should proceed independent of which params are actively iterated,
        # as all params are presumably needed to complete the simulation
        param_info = {"names": param_names,
                      "unit_conversions": unit_conversions,
                      "active": {name: 0 for name in param_names}}
        initial_guess = {"n0": 0,
                         "p0": 0,
                         "mu_n": 0,
                         "mu_p": 0,
                         "ks": 1e-11,
                         "Sf": 0,
                         "Sb": 0,
                         "Cn": 0,
                         "Cp": 0,
                         "Tm": 300,
                         "tauN": 1e99,
                         "tauP": 1e99,
                         "eps": 10,
                         "m": 1}
        param_info["init_guess"] = initial_guess

        sim_flags = {"current_sigma": {"TRPL": 1},
                     }

        indexes = {name: param_names.index(name) for name in param_names}
        state = [param_info["init_guess"][name] for name in param_names]
        units = np.array([unit_conversions.get(name, 1) for name in param_names], dtype=float)

        nt = 1000
        times = [np.linspace(0, 100, nt+1), np.linspace(0, 100, nt+1)]
        vals = [np.ones(nt+1) * 23, np.ones(nt+1) * 23]
        uncs = [np.ones(nt+1) * 1e-99, np.ones(nt+1) * 1e-99]
        self.mock_ensemble.sim_info = simPar
        self.mock_ensemble.iniPar = iniPar
        self.mock_ensemble.times = times
        self.mock_ensemble.param_indexes = indexes
        self.mock_ensemble.ensemble_fields = {"units": units, "solver": ("solveivp",),
                     "model": "std", "hmax": 4, "rtol": 1e-5, "atol": 1e-8}
        self.mock_ensemble.vals = vals
        self.mock_ensemble.uncs = uncs
        logll, _ = self.mock_ensemble.eval_trial_move(state, sim_flags)

        np.testing.assert_almost_equal(
            logll, np.sum([-59340.105083, -32560.139058]), decimal=0)  # rtol=1e-5

    def test_run_iter_depletion(self):
        """Prove that the truncation allows likelihood of two carrier-depleting simulations to be reliably determined."""
        np.random.seed(42)
        Length = [2000]                            # Length (nm)
        L = [2 ** 7]                                # Spatial point
        mtype = ["TRPL"]
        simPar = {"lengths": Length, "nx": L, "meas_types": mtype, "num_meas": 1}

        iniPar = np.array([1e15 * np.ones(L[0])]) # PL(t=0) ~ 2e15

        param_names = ["n0", "p0", "mu_n", "mu_p", "ks", "Cn", "Cp", "Tm",
                       "Sf", "Sb", "tauN", "tauP", "eps", "m"]
        unit_conversions = {"n0": ((1e-7) ** 3), "p0": ((1e-7) ** 3),
                            "mu_n": ((1e7) ** 2) / (1e9), "mu_p": ((1e7) ** 2) / (1e9),
                            "ks": ((1e7) ** 3) / (1e9), "Sf": 1e-2, "Sb": 1e-2}

        param_info = {"names": param_names,
                      "unit_conversions": unit_conversions,
                      "active": {name: 0 for name in param_names}}
        initial_guess = {"n0": 1e8,
                         "p0": 1e17,
                         "mu_n": 0,
                         "mu_p": 0,
                         "ks": 1e-13,
                         "Sf": 0,
                         "Sb": 0,
                         "Cn": 0,
                         "Cp": 0,
                         "Tm": 300,
                         "tauN": 4, # Fast enough to deplete carriers
                         "tauP": 4,
                         "eps": 10,
                         "m": 1}
        param_info["init_guess"] = initial_guess

        sim_flags = {"current_sigma": {"TRPL": 1},
                     }

        indexes = {name: param_names.index(name) for name in param_names}
        state = [param_info["init_guess"][name] for name in param_names]
        units = np.array([unit_conversions.get(name, 1) for name in param_names], dtype=float)

        nt = 1000
        times = [np.linspace(0, 100, nt+1)]
        vals = [np.log10(2e14 * np.exp(-times[0] / 8))]
        uncs = [np.ones(nt+1) * 1e-99]
        self.mock_ensemble.sim_info = simPar
        self.mock_ensemble.iniPar = iniPar
        self.mock_ensemble.times = times
        self.mock_ensemble.param_indexes = indexes
        self.mock_ensemble.ensemble_fields = {"units": units, "solver": ("solveivp",),
                     "model": "std", "hmax": 4, "rtol": 1e-5, "atol": 1e-8, "force_min_y": True}
        self.mock_ensemble.vals = vals
        self.mock_ensemble.uncs = uncs
        logll1, _ = self.mock_ensemble.eval_trial_move(state, sim_flags)

        # A small move toward the true lifetime of 10 makes the likelihood better
        # Without min_y truncation, the likelihoods were so small they weren't even comparable
        param_info["init_guess"]["tauN"] = 4.01
        param_info["init_guess"]["tauP"] = 4.01

        state = [param_info["init_guess"][name] for name in param_names]
        logll2, _ = self.mock_ensemble.eval_trial_move(state, sim_flags)
        self.assertTrue(logll2 > logll1)

    def test_run_iter_cutoff(self):
        # Same as test_run_iter, only "experimental" data is
        # truncated at [0,50] instead of [0,100].
        # Half as many points means the likelihood should be reduced to about half.
        np.random.seed(42)
        Length = [2000, 2000]                            # Length (nm)
        L = [2 ** 7, 2 ** 7]                                # Spatial point
        mtype = ["TRPL", "TRPL"]
        simPar = {"lengths": Length,
                  "nx": L,
                  "meas_types": mtype,
                  "num_meas": 2}

        iniPar = np.array([1e15 * np.ones(L[0]), 1e16 * np.ones(L[1])])

        param_names = ["n0", "p0", "mu_n", "mu_p", "ks", "Cn", "Cp", "Tm",
                       "Sf", "Sb", "tauN", "tauP", "eps", "m"]
        unit_conversions = {"n0": ((1e-7) ** 3), "p0": ((1e-7) ** 3),
                            "mu_n": ((1e7) ** 2) / (1e9), "mu_p": ((1e7) ** 2) / (1e9),
                            "ks": ((1e7) ** 3) / (1e9), "Sf": 1e-2, "Sb": 1e-2}

        # Iterations should proceed independent of which params are actively iterated,
        # as all params are presumably needed to complete the simulation
        param_info = {"names": param_names,
                      "unit_conversions": unit_conversions,
                      "active": {name: 0 for name in param_names}}
        initial_guess = {"n0": 0,
                         "p0": 0,
                         "mu_n": 0,
                         "mu_p": 0,
                         "ks": 1e-11,
                         "Sf": 0,
                         "Sb": 0,
                         "Cn": 0,
                         "Cp": 0,
                         "Tm": 300,
                         "tauN": 1e99,
                         "tauP": 1e99,
                         "eps": 10,
                         "m": 1}
        param_info["init_guess"] = initial_guess

        sim_flags = {"current_sigma": {"TRPL": 1},}

        indexes = {name: param_names.index(name) for name in param_names}
        state = [param_info["init_guess"][name] for name in param_names]
        units = np.array([unit_conversions.get(name, 1) for name in param_names], dtype=float)

        nt = 500
        times = [np.linspace(0, 50, nt+1), np.linspace(0, 50, nt+1)]
        vals = [np.ones(nt+1) * 23, np.ones(nt+1) * 23]
        uncs = [np.ones(nt+1) * 1e-99, np.ones(nt+1) * 1e-99]

        self.mock_ensemble.sim_info = simPar
        self.mock_ensemble.iniPar = iniPar
        self.mock_ensemble.times = times
        self.mock_ensemble.param_indexes = indexes
        self.mock_ensemble.ensemble_fields = {"units": units, "solver": ("solveivp",),
                     "model": "std", "hmax": 4, "rtol": 1e-5, "atol": 1e-8}
        self.mock_ensemble.vals = vals
        self.mock_ensemble.uncs = uncs
        logll, _ = self.mock_ensemble.eval_trial_move(state, sim_flags)

        # First iter; auto-accept
        np.testing.assert_almost_equal(
            logll, -45982, decimal=0)  # rtol=1e-5

    def test_run_iter_scale(self):
        # Same as test_run_iter, except global scale factor, which will be chosen to match
        # the first measurement (but off by one order of magnitude from the second)
        # likelihood should be small but nonzero as a result
        np.random.seed(42)
        Length = [2000, 2000]                            # Length (nm)
        L = [2 ** 7, 2 ** 7]                                # Spatial point
        mtype = ["TRPL", "TRPL"]
        simPar = {"lengths": Length,
                  "nx": L,
                  "meas_types": mtype,
                  "num_meas": 2}

        iniPar = np.array([1e15 * np.ones(L[0]), 1e16 * np.ones(L[1])])

        param_names = ["n0", "p0", "mu_n", "mu_p", "ks", "Cn", "Cp", "Tm",
                       "Sf", "Sb", "tauN", "tauP", "eps"]
        unit_conversions = {"n0": ((1e-7) ** 3), "p0": ((1e-7) ** 3),
                            "mu_n": ((1e7) ** 2) / (1e9), "mu_p": ((1e7) ** 2) / (1e9),
                            "ks": ((1e7) ** 3) / (1e9), "Sf": 1e-2, "Sb": 1e-2}

        # Iterations should proceed independent of which params are actively iterated,
        # as all params are presumably needed to complete the simulation
        param_info = {"names": param_names,
                      "unit_conversions": unit_conversions,
                      "active": {name: 0 for name in param_names}}
        initial_guess = {"n0": 0,
                         "p0": 0,
                         "mu_n": 0,
                         "mu_p": 0,
                         "ks": 1e-20,
                         "Sf": 0,
                         "Sb": 0,
                         "Cn": 0,
                         "Cp": 0,
                         "Tm": 300,
                         "tauN": 1e99,
                         "tauP": 1e99,
                         "eps": 10}
        param_info["init_guess"] = initial_guess

        sim_flags = {"current_sigma": {"TRPL": 1},
                     }

        # These would normally be inserted when the script file is read by bayes_io
        param_info["names"].append("_s0")
        param_info["names"].append("_s1")
        # By setting individual scale factors in this simple case the likelihood can be made perfect
        param_info["init_guess"]["_s0"] = 2e-17 ** -1
        param_info["init_guess"]["_s1"] = 2e-15 ** -1
        indexes = {name: param_names.index(name) for name in param_names}
        state = [param_info["init_guess"][name] for name in param_names]
        units = np.array([unit_conversions.get(name, 1) for name in param_names], dtype=float)

        nt = 1000
        times = [np.linspace(0, 100, nt+1), np.linspace(0, 100, nt+1)]
        vals = [np.ones(nt+1) * 23, np.ones(nt+1) * 23]
        uncs = [np.ones(nt+1) * 1e-99, np.ones(nt+1) * 1e-99]

        self.mock_ensemble.sim_info = simPar
        self.mock_ensemble.iniPar = iniPar
        self.mock_ensemble.times = times
        self.mock_ensemble.param_indexes = indexes
        self.mock_ensemble.ensemble_fields = {"units": units, "solver": ("solveivp",),
                     "model": "std", "hmax": 4, "rtol": 1e-5, "atol": 1e-8,
                     "scale_factor": (0.02, [0, 1, 2, 3, 4, 5], [(0, 2, 4), (1, 3, 5)])}
        self.mock_ensemble.vals = vals
        self.mock_ensemble.uncs = uncs
        logll, _ = self.mock_ensemble.eval_trial_move(state, sim_flags)

        np.testing.assert_almost_equal(
            logll, 0, decimal=0)  # rtol=1e-5

    def test_run_iter_mixed_types(self):
        # Will basically need to set up a full simulation for this
        np.random.seed(42)
        Length = [2000, 2000]                            # Length (nm)
        L = [2 ** 7, 2 ** 7]                                # Spatial point
        mtype = ["TRPL", "TRTS"]
        simPar = {"lengths": Length,
                  "nx": L,
                  "meas_types": mtype,
                  "num_meas": 2}

        iniPar = np.array([1e15 * np.ones(L[0]), 1e15 * np.ones(L[1])])

        param_names = ["n0", "p0", "mu_n", "mu_p", "ks", "Cn", "Cp", "Tm",
                       "Sf", "Sb", "tauN", "tauP", "eps", "m"]
        unit_conversions = {"n0": ((1e-7) ** 3), "p0": ((1e-7) ** 3),
                            "mu_n": ((1e7) ** 2) / (1e9), "mu_p": ((1e7) ** 2) / (1e9),
                            "ks": ((1e7) ** 3) / (1e9), "Sf": 1e-2, "Sb": 1e-2}

        # Iterations should proceed independent of which params are actively iterated,
        # as all params are presumably needed to complete the simulation
        param_info = {"names": param_names,
                      "unit_conversions": unit_conversions,
                      "active": {name: 0 for name in param_names}}
        initial_guess = {"n0": 0,
                         "p0": 0,
                         "mu_n": 0.01,
                         "mu_p": 0.01,
                         "ks": 1e-11,
                         "Sf": 0,
                         "Sb": 0,
                         "Cn": 0,
                         "Cp": 0,
                         "Tm": 300,
                         "tauN": 1e99,
                         "tauP": 1e99,
                         "eps": 10,
                         "m": 1}
        param_info["init_guess"] = initial_guess

        sim_flags = {"current_sigma": {"TRPL": 1, "TRTS": 10},}

        indexes = {name: param_names.index(name) for name in param_names}
        state = [param_info["init_guess"][name] for name in param_names]
        units = np.array([unit_conversions.get(name, 1) for name in param_names], dtype=float)

        nt = 1000
        times = [np.linspace(0, 100, nt+1), np.linspace(0, 100, nt+1)]
        vals = [np.ones(nt+1) * 23, np.ones(nt+1) * -2]
        uncs = [np.ones(nt+1) * 1e-99, np.ones(nt+1) * 1e-99]
        self.mock_ensemble.sim_info = simPar
        self.mock_ensemble.iniPar = iniPar
        self.mock_ensemble.times = times
        self.mock_ensemble.param_indexes = indexes
        self.mock_ensemble.ensemble_fields = {"units": units, "solver": ("solveivp",),
                     "model": "std", "hmax": 4, "rtol": 1e-5, "atol": 1e-8}
        self.mock_ensemble.vals = vals
        self.mock_ensemble.uncs = uncs
        logll, _ = self.mock_ensemble.eval_trial_move(state, sim_flags)

        # First iter; auto-accept
        np.testing.assert_almost_equal(
            logll, np.sum([-59340.105083, -517.98]), decimal=0)  # rtol=1e-5
