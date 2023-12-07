import unittest
import logging
import sys
sys.path.append("..")
import numpy as np
from scipy.integrate import trapz
from metropolis import all_signal_handler
from metropolis import select_next_params, check_approved_param
from forward_solver import E_field, solve, calculate_PL, calculate_TRTS
from metropolis import roll_acceptance
from utils import unpack_simpar, set_min_y, search_c_grps
from metropolis import almost_equal

from sim_utils import Grid
q = 1.0  # [e]
q_C = 1.602e-19  # [C]
kB = 8.61773e-5  # [eV / K]

MIN_HMAX = 0.01


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger()

    def test_search_c_grps(self):
        c_grps = [(0, 1, 2), (3, 4)]
        self.assertEqual(search_c_grps(c_grps, 0), 0)
        self.assertEqual(search_c_grps(c_grps, 1), 0)
        self.assertEqual(search_c_grps(c_grps, 2), 0)
        self.assertEqual(search_c_grps(c_grps, 3), 3)
        self.assertEqual(search_c_grps(c_grps, 4), 3)

    def test_all_signal(self):
        def f(x): return x
        all_signal_handler(f)

    def test_E_field(self):
        eps0 = 8.854 * 1e-12 * 1e-9  # [C / V m] to {C / V nm}
        # Test 1D
        vals = {'n0': 0,
                'p0': 0,
                'eps': 1}

        nx = 10
        dx = 1
        # Test N=P
        N = np.zeros(nx)
        P = np.zeros(nx)
        E = E_field(N, P, vals["n0"], vals["p0"], vals["eps"], dx)

        np.testing.assert_equal(E, np.zeros_like(E))

        # Test N>P
        P = np.ones(nx) * 2
        N = np.ones(nx)
        E = E_field(N, P, vals["n0"], vals["p0"], vals["eps"], dx)

        np.testing.assert_equal(E[1:], np.ones_like(
            E[1:]) * q_C/eps0 * np.cumsum(N))

        # Test N<P
        P = np.ones(nx) * -1
        E = E_field(N, P, vals["n0"], vals["p0"], vals["eps"], dx)

        np.testing.assert_equal(
            E[1:], -2 * np.ones_like(E[1:]) * q_C/eps0 * np.cumsum(N))

        # Test corner_E != 0
        corner_E = 24
        E = E_field(N, P, vals["n0"], vals["p0"], vals["eps"], dx, corner_E=corner_E)

        np.testing.assert_equal(
            E[1:], -2 * np.ones_like(E[1:]) * q_C/eps0 * np.cumsum(N) + corner_E)

        # Test 2D
        N = np.ones((nx, nx+1))
        P = np.ones((nx, nx+1)) * -1
        E = E_field(N, P, vals["n0"], vals["p0"], vals["eps"], dx, corner_E=corner_E)
        np.testing.assert_equal(
            E[:, 1:], -2 * np.ones_like(E[:, 1:]) * q_C/eps0 * np.cumsum(N, axis=1) + corner_E)

        # Test n0, p0
        vals = {'n0': 1,
                'p0': 1,
                'eps': 1}
        N = np.ones((nx, nx+1))
        P = np.ones((nx, nx+1))
        E = E_field(N, P, vals["n0"], vals["p0"], vals["eps"], dx, corner_E=corner_E)
        np.testing.assert_equal(E[1:], np.zeros_like(E[1:]) + corner_E)

        return

    def test_solve(self):
        # A high-inj, rad-only sample problem
        g = Grid(thickness=1000, nx=100, tSteps=np.linspace(0, 100, 1001), hmax=4)

        unit_conversions = {"n0": ((1e-7) ** 3), "p0": ((1e-7) ** 3),
                            "mu_n": ((1e7) ** 2) / (1e9),
                            "mu_p": ((1e7) ** 2) / (1e9),
                            "ks": ((1e7) ** 3) / (1e9), "Sf": 1e-2, "Sb": 1e-2}

        param_info = {"names": ["n0", "p0", "mu_n", "mu_p",
                                "ks", "tauN", "tauP",
                                "Cn", "Cp", "Sf", "Sb", "eps", "Tm"],
                      "active": {"n0": 0, "p0": 1,
                                 "mu_n": 0, "mu_p": 0,
                                 "Cn": 0, "Cp": 0,
                                 "ks": 1, "Sf": 1, "Sb": 1,
                                 "tauN": 1, "tauP": 1, "eps": 0,
                                 "Tm": 0},
                      "unit_conversions": unit_conversions}
        vals = {'n0': 0,
                'p0': 0,
                'mu_n': 0,
                'mu_p': 0,
                "ks": 1e-11,
                "Cn": 0, "Cp": 0,
                'tauN': 1e99,
                'tauP': 1e99,
                'Sf': 0,
                'Sb': 0,
                "Tm": 300,
                'eps': 1}

        param_info["init_guess"] = vals
        indexes = {name: i for i, name in enumerate(param_info["names"])}
        state = [param_info["init_guess"][name] for name in param_info["names"]]
        units = np.array([unit_conversions.get(name, 1) for name in param_info["names"]], dtype=float)
        init_dN = 1e20 * np.ones(g.nx) # [cm^-3]
        out_dN = np.full_like(init_dN, 0.0009900990095719482)
        # with solveivp
        test_PL = solve(init_dN, g, state, indexes, meas="TRPL", units=units, solver=("solveivp",),
                                RTOL=1e-10, ATOL=1e-14)
        # Calculate expected output in simulation units
        for name in indexes:
            state[indexes[name]] *= unit_conversions.get(name, 1)
        rr = state[indexes["ks"]] * (out_dN * out_dN - state[indexes["n0"]] * state[indexes["p0"]])
        expected_out = trapz(rr, dx=g.dx) + rr[0]*g.dx/2 + rr[-1]*g.dx/2
        expected_out *= 1e23
        for name in indexes:
            state[indexes[name]] /= unit_conversions.get(name, 1)
        self.assertAlmostEqual(test_PL[-1] / np.amax(test_PL[-1]), expected_out / np.amax(test_PL[-1]))

        # with odeint
        test_PL = solve(init_dN, g, state, indexes, meas="TRPL", units=units, solver=("odeint",),
                                RTOL=1e-10, ATOL=1e-14)
        for name in indexes:
            state[indexes[name]] *= unit_conversions.get(name, 1)
        rr = state[indexes["ks"]] * (out_dN * out_dN - state[indexes["n0"]] * state[indexes["p0"]])
        expected_out = trapz(rr, dx=g.dx) + rr[0]*g.dx/2 + rr[-1]*g.dx/2
        expected_out *= 1e23
        for name in indexes:
            state[indexes[name]] /= unit_conversions.get(name, 1)
        self.assertAlmostEqual(test_PL[-1] / np.amax(test_PL[-1]), expected_out / np.amax(test_PL[-1]),
            places=6)

        # try a trts

        vals = {'n0': 0,
                'p0': 0,
                'mu_n': 10,
                'mu_p': 10,
                "ks": 1e-11,
                "Cn": 0, "Cp": 0,
                'tauN': 1e99,
                'tauP': 1e99,
                'Sf': 0,
                'Sb': 0,
                "Tm": 300,
                'eps': 1}
        param_info["init_guess"] = vals
        indexes = {name: i for i, name in enumerate(param_info["names"])}
        state = [param_info["init_guess"][name] for name in param_info["names"]]
        out_dN = np.full_like(init_dN, 0.0009900986886696803)
        test_TRTS = solve(
            init_dN, g, state, indexes, meas="TRTS", units=units, solver=("solveivp",))
        for name in indexes:
            state[indexes[name]] *= unit_conversions.get(name, 1)
        trts = q_C * (state[indexes["mu_n"]] * out_dN + state[indexes["mu_p"]] * out_dN)
        expected_out = trapz(trts, dx=g.dx) + trts[0]*g.dx/2 + trts[-1]*g.dx/2
        expected_out *= 1e9
        for name in indexes:
            state[indexes[name]] /= unit_conversions.get(name, 1)
        self.assertAlmostEqual(test_TRTS[-1] / np.amax(test_TRTS[-1]), expected_out / np.amax(test_TRTS[-1]))

        for n in param_info["names"]:
            self.assertEqual(state[indexes[n]], vals[n])

        # try an undefined measurement
        with self.assertRaises(NotImplementedError):
            solve(init_dN, g, state, indexes, meas="something else")

        # try an undefined solver
        with self.assertRaises(NotImplementedError):
            solve(init_dN, g, state, indexes, meas="TRPL", solver=("somethign else",))

        return

    def test_solve_depletion(self):
        """ A high-inj, rad-only sample problem. Parameters are chosen such that
        the simulated PL will decay by about 2.5 orders of magnitude. """
        g = Grid(thickness=1000, nx=100, tSteps=np.linspace(0, 100, 1001), hmax=4)

        unit_conversions = {"n0": ((1e-7) ** 3), "p0": ((1e-7) ** 3),
                            "mu_n": ((1e7) ** 2) / (1e9),
                            "mu_p": ((1e7) ** 2) / (1e9),
                            "ks": ((1e7) ** 3) / (1e9), "Sf": 1e-2, "Sb": 1e-2}

        param_info = {"names": ["n0", "p0", "mu_n", "mu_p",
                                "ks", "tauN", "tauP",
                                "Cn", "Cp", "Sf", "Sb", "eps", "Tm"],
                      "active": {"n0": 0, "p0": 1,
                                 "mu_n": 1, "mu_p": 1,
                                 "Cn": 0, "Cp": 0,
                                 "ks": 1, "Sf": 1, "Sb": 1,
                                 "tauN": 1, "tauP": 1, "eps": 0,
                                 "Tm": 0},
                      "unit_conversions": unit_conversions}
        vals = {'n0': 0,
                'p0': 0,
                'mu_n': 1,
                'mu_p': 1,
                "ks": 2e-10,
                "Cn": 0, "Cp": 0,
                'tauN': 1e99,
                'tauP': 1e99,
                'Sf': 0,
                'Sb': 0,
                "Tm": 300,
                'eps': 1}

        param_info["init_guess"] = vals
        indexes = {name: i for i, name in enumerate(param_info["names"])}
        state = [param_info["init_guess"][name] for name in param_info["names"]]
        units = np.array([unit_conversions.get(name, 1) for name in param_info["names"]], dtype=float)
        init_dN = 1e18 * np.ones(g.nx) # [cm^-3]

        PL0 = calculate_PL(g.dx * 1e-7, init_dN, init_dN, vals["ks"], vals["n0"], vals["p0"]) # in cm/s units
        # No or large range - PL runs to conclusion
        test_PL = solve(init_dN, g, state, indexes, meas="TRPL", units=units, solver=("solveivp",),
                        RTOL=1e-10, ATOL=1e-14)
        self.assertEqual(len(g.tSteps), len(test_PL))

        # Smaller range - PL decay stops early at PL_final, and signal over remaining time is set to PL_final
        g.min_y = PL0 * 1e-2
        test_PL = solve(init_dN, g, state, indexes, meas="TRPL", units=units, solver=("solveivp",),
                        RTOL=1e-10, ATOL=1e-14)

        self.assertTrue(min(test_PL) >= g.min_y)
        np.testing.assert_equal(test_PL[-10:], g.min_y)

        # Try a TRTS
        TRTS0 = calculate_TRTS(g.dx * 1e-7, init_dN, init_dN, vals["mu_n"], vals["mu_p"], vals["n0"], vals["p0"])
        # No or large range - PL runs to conclusion
        g.min_y = TRTS0 * 1e-10
        test_TRTS = solve(init_dN, g, state, indexes, meas="TRTS", units=units, solver=("solveivp",),
                          RTOL=1e-10, ATOL=1e-14)
        self.assertEqual(len(g.tSteps), len(test_TRTS))


        # Smaller range - TRTS is truncated
        g.min_y = TRTS0 * 1e-1
        test_TRTS = solve(init_dN, g, state, indexes, meas="TRTS", units=units, solver=("solveivp",),
                          RTOL=1e-10, ATOL=1e-14)
        self.assertTrue(min(test_TRTS) >= g.min_y)
        np.testing.assert_equal(test_TRTS[-10:], g.min_y)

        return

    def test_solve_traps(self):
        # A high-inj, rad-only sample problem using null parameters for the trap model
        # which should be equivalent to the std model
        g = Grid(thickness=1000, nx=100, tSteps=np.linspace(0, 100, 1001), hmax=4)

        unit_conversions = {"n0": ((1e-7) ** 3), "p0": ((1e-7) ** 3),
                            "mu_n": ((1e7) ** 2) / (1e9),
                            "mu_p": ((1e7) ** 2) / (1e9),
                            "ks": ((1e7) ** 3) / (1e9), "Sf": 1e-2, "Sb": 1e-2,
                            "kC": ((1e7) ** 3) / (1e9),
                            "Nt": ((1e-7) ** 3)}

        param_info = {"names": ["n0", "p0", "mu_n", "mu_p",
                                "ks", "tauN", "tauP",
                                "Cn", "Cp", "Sf", "Sb", "eps", "Tm",
                                "kC", "Nt", "tauE"],
                      "active": {"n0": 0, "p0": 1,
                                 "mu_n": 0, "mu_p": 0,
                                 "Cn": 0, "Cp": 0,
                                 "ks": 1, "Sf": 1, "Sb": 1,
                                 "tauN": 1, "tauP": 1, "eps": 0,
                                 "Tm": 0,"kC": 1, "Nt": 1, "tauE": 1},
                      "unit_conversions": unit_conversions}
        vals = {"n0": 0,
                "p0": 0,
                "mu_n": 0,
                "mu_p": 0,
                "ks": 1e-11,
                "Cn": 0, "Cp": 0,
                "tauN": 1e99,
                "tauP": 1e99,
                "Sf": 0,
                "Sb": 0,
                "Tm": 300,
                "eps": 1,
                "kC": 0,
                "Nt": 0,
                "tauE": 1e99}

        param_info["init_guess"] = vals
        indexes = {name: i for i, name in enumerate(param_info["names"])}
        state = [param_info["init_guess"][name] for name in param_info["names"]]
        units = np.array([unit_conversions.get(name, 1) for name in param_info["names"]], dtype=float)
        init_dN = 1e20 * np.ones(g.nx) # [cm^-3]
        out_dN = np.full_like(init_dN, 0.0009900990095719482)
        # with solveivp
        test_PL= solve(init_dN, g, state, indexes, meas="TRPL", units=units, solver=("solveivp",), model="traps",
                       RTOL=1e-10, ATOL=1e-14)
        # Calculate expected output in simulation units
        for name in indexes:
            state[indexes[name]] *= unit_conversions.get(name, 1)
        rr = state[indexes["ks"]] * (out_dN * out_dN - state[indexes["n0"]] * state[indexes["p0"]])
        expected_out = trapz(rr, dx=g.dx) + rr[0]*g.dx/2 + rr[-1]*g.dx/2
        expected_out *= 1e23
        for name in indexes:
            state[indexes[name]] /= unit_conversions.get(name, 1)
        self.assertAlmostEqual(test_PL[-1] / np.amax(test_PL[-1]), expected_out / np.amax(test_PL[-1]))

        return

    def test_solve_iniPar(self):
        # A high-inj, rad-only sample problem
        g = Grid(thickness=1000, nx=100, tSteps=np.linspace(0, 100, 1001), hmax=4)

        unit_conversions = {"n0": ((1e-7) ** 3), "p0": ((1e-7) ** 3),
                            "mu_n": ((1e7) ** 2) / (1e9),
                            "mu_p": ((1e7) ** 2) / (1e9),
                            "ks": ((1e7) ** 3) / (1e9), "Sf": 1e-2, "Sb": 1e-2}

        param_info = {"names": ["n0", "p0", "mu_n", "mu_p",
                                "ks", "tauN", "tauP",
                                "Cn", "Cp", "Sf", "Sb", "eps", "Tm"],
                      "active": {"n0": 0, "p0": 1,
                                 "mu_n": 0, "mu_p": 0,
                                 "Cn": 0, "Cp": 0,
                                 "ks": 1, "Sf": 1, "Sb": 1,
                                 "tauN": 1, "tauP": 1, "eps": 0,
                                 "Tm": 0},
                      "unit_conversions": unit_conversions}
        vals = {'n0': 0,
                'p0': 0,
                'mu_n': 0,
                'mu_p': 0,
                "ks": 1e-11,
                "Cn": 0, "Cp": 0,
                'tauN': 1e99,
                'tauP': 1e99,
                'Sf': 0,
                'Sb': 0,
                "Tm": 300,
                'eps': 1}

        param_info["init_guess"] = vals
        indexes = {name: i for i, name in enumerate(param_info["names"])}
        state = [param_info["init_guess"][name] for name in param_info["names"]]
        units = np.array([unit_conversions.get(name, 1) for name in param_info["names"]], dtype=float)

        fluence = 1e15 # Fluence, alpha in cm units
        alpha = 6e4
        g.xSteps = np.linspace(g.dx / 2, g.thickness - g.dx/2, g.nx)

        init_dN = fluence * alpha * np.exp(-alpha * g.xSteps * 1e-7)  # In cm units

        PL_by_initvals = solve(init_dN, g, state, indexes, meas="TRPL", units=units, solver=("solveivp",),
                               RTOL=1e-10, ATOL=1e-14)

        PL_by_initparams = solve([fluence, alpha], g, state, indexes, meas="TRPL", units=units, solver=("solveivp",),
                                 RTOL=1e-10, ATOL=1e-14)

        np.testing.assert_almost_equal(PL_by_initvals / np.amax(PL_by_initvals), PL_by_initparams / np.amax(PL_by_initvals))

    def test_approve_param(self):
        info = {'names': ['tauP', 'tauN', 'somethingelse'],
                'prior_dist': {'tauP': (0.1, np.inf), 'tauN': (0.1, np.inf),
                               'somethingelse': (-np.inf, np.inf)},
                'do_log': {'tauP': 1, 'tauN': 1, 'somethingelse': 1},
                "active": {'tauP': 1, 'tauN': 1, 'somethingelse': 1}}
        indexes = {name: info["names"].index(name) for name in info["names"]}
        do_log = np.array([info["do_log"][param] for param in info["names"]], dtype=bool)
        active = np.array([info["active"][name] for name in info["names"]], dtype=bool)
        # taun, taup must be within 2 OM
        # Accepts new_p as log10
        # [n0, p0, mu_n, mu_p, ks, sf, sb, taun, taup, eps, m]
        new_p = np.log10([511, 511e2, 1])
        self.assertTrue(len(check_approved_param(new_p, info, indexes, active, do_log)) == 0)
        new_p = np.log10([511, 511e2+1,  1])
        self.assertTrue("tn_tp_close" in check_approved_param(new_p, info, indexes, active, do_log))

        # tn, tp size limit
        new_p = np.log10([0.11, 0.11, 1])
        self.assertTrue(len(check_approved_param(new_p, info, indexes, active, do_log)) == 0)
        new_p = np.log10([0.1, 0.11, 1])
        self.assertTrue("tauP_size" in check_approved_param(new_p, info, indexes, active, do_log))
        new_p = np.log10([0.11, 0.1, 1])
        self.assertTrue("tauN_size" in check_approved_param(new_p, info, indexes, active, do_log))

        # If params are inactive, they should not be checked
        info = {'names': ['tauP', 'tauN', 'somethingelse'],
                'prior_dist': {'tauP': (0.1, np.inf), 'tauN': (0.1, np.inf),
                               'somethingelse': (-np.inf, np.inf)},
                'do_log': {'tauP': 1, 'tauN': 1, 'somethingelse': 1},
                "active": {'tauP': 0, 'tauN': 0, 'somethingelse': 1}}
        indexes = {name: info["names"].index(name) for name in info["names"]}
        do_log = np.array([info["do_log"][param] for param in info["names"]], dtype=bool)
        active = np.array([info["active"][name] for name in info["names"]], dtype=bool)
        new_p = np.log10([0.11, 0.1, 1])
        self.assertTrue(len(check_approved_param(new_p, info, indexes, active, do_log)) == 0)

        # These should still work if p is not logscaled
        info = {'names': ['tauP', 'tauN', 'somethingelse'],
                'prior_dist': {'tauP': (0.1, np.inf), 'tauN': (0.1, np.inf),
                               'somethingelse': (-np.inf, np.inf)},
                'do_log': {'tauP': 0, 'tauN': 0, 'somethingelse': 1},
                "active": {'tauP': 1, 'tauN': 1, 'somethingelse': 1}}
        indexes = {name: info["names"].index(name) for name in info["names"]}
        do_log = np.array([info["do_log"][param] for param in info["names"]], dtype=bool)
        active = np.array([info["active"][name] for name in info["names"]], dtype=bool)
        new_p = np.array([511, 511e2, 1])
        self.assertTrue(len(check_approved_param(new_p, info, indexes, active, do_log)) == 0)
        new_p = np.array([511, 511e2+1,  1])
        self.assertTrue("tn_tp_close" in check_approved_param(new_p, info, indexes, active, do_log))

        new_p = np.array([0.11, 0.11, 1])
        self.assertTrue(len(check_approved_param(new_p, info, indexes, active, do_log)) == 0)
        new_p = np.array([0.1, 0.11, 1])
        self.assertTrue("tauP_size" in check_approved_param(new_p, info, indexes, active, do_log))
        new_p = np.array([0.11, 0.1, 1])
        self.assertTrue("tauN_size" in check_approved_param(new_p, info, indexes, active, do_log))

        # Check mu_n, mu_p, Sf, and Sb size limits
        info = {"names": ["mu_n", "mu_p", "Sf", "Sb"],
                'prior_dist': {'mu_n': (0.1, 1e6), 'mu_p': (0.1, 1e6),
                               'Sf': (0, 1e7), 'Sb': (0, 1e7)},
                'do_log': {"mu_n": 1, "mu_p": 1, "Sf": 1, "Sb": 1},
                "active": {"mu_n": 1, "mu_p": 1, "Sf": 1, "Sb": 1}}
        indexes = {name: info["names"].index(name) for name in info["names"]}
        do_log = np.array([info["do_log"][param] for param in info["names"]], dtype=bool)
        active = np.array([info["active"][name] for name in info["names"]], dtype=bool)
        new_p = np.log10([1e6-1, 1e6-1, 1e7-1, 1e7-1])
        self.assertTrue(len(check_approved_param(new_p, info, indexes, active, do_log)) == 0)

        new_p = np.log10([1e6, 1e6-1, 1e7-1, 1e7-1])
        self.assertTrue("mu_n_size" in check_approved_param(new_p, info, indexes, active, do_log))
        new_p = np.log10([1e6-1, 1e6, 1e7-1, 1e7-1])
        self.assertTrue("mu_p_size" in check_approved_param(new_p, info, indexes, active, do_log))
        new_p = np.log10([1e6-1, 1e6-1, 1e7, 1e7-1])
        self.assertTrue("Sf_size" in check_approved_param(new_p, info, indexes, active, do_log))
        new_p = np.log10([1e6-1, 1e6-1, 1e7-1, 1e7])
        self.assertTrue("Sb_size" in check_approved_param(new_p, info, indexes, active, do_log))

        # These should still work if p is not logscaled
        info = {"names": ["mu_n", "mu_p", "Sf", "Sb"],
                'prior_dist': {'mu_n': (0.1, 1e6), 'mu_p': (0.1, 1e6),
                               'Sf': (0, 1e7), 'Sb': (0, 1e7)},
                'do_log': {"mu_n": 0, "mu_p": 0, "Sf": 0, "Sb": 0},
                "active": {"mu_n": 1, "mu_p": 1, "Sf": 1, "Sb": 1}}
        indexes = {name: info["names"].index(name) for name in info["names"]}
        do_log = np.array([info["do_log"][param] for param in info["names"]], dtype=bool)
        active = np.array([info["active"][name] for name in info["names"]], dtype=bool)
        new_p = np.array([1e6-1, 1e6-1, 1e7-1, 1e7-1])
        self.assertTrue(len(check_approved_param(new_p, info, indexes, active, do_log)) == 0)

        new_p = np.array([1e6, 1e6-1, 1e7-1, 1e7-1])
        self.assertTrue("mu_n_size" in check_approved_param(new_p, info, indexes, active, do_log))
        new_p = np.array([1e6-1, 1e6, 1e7-1, 1e7-1])
        self.assertTrue("mu_p_size" in check_approved_param(new_p, info, indexes, active, do_log))
        new_p = np.array([1e6-1, 1e6-1, 1e7, 1e7-1])
        self.assertTrue("Sf_size" in check_approved_param(new_p, info, indexes, active, do_log))
        new_p = np.array([1e6-1, 1e6-1, 1e7-1, 1e7])
        self.assertTrue("Sb_size" in check_approved_param(new_p, info, indexes, active, do_log))

        # Check ks, Cn, Cp size limits
        info = {"names": ["ks", "Cn", "Cp"],
                'prior_dist': {'ks': (0, 1e-7), 'Cn': (0, 1e-21),
                               'Cp': (0, 1e-21)},
                "do_log": {"ks": 1, "Cn": 1, "Cp": 1},
                "active": {"ks": 1, "Cn": 1, "Cp": 1}}
        indexes = {name: info["names"].index(name) for name in info["names"]}
        do_log = np.array([info["do_log"][param] for param in info["names"]], dtype=bool)
        active = np.array([info["active"][name] for name in info["names"]], dtype=bool)
        new_p = np.log10([1e-7*0.9, 1e-21*0.9, 1e-21*0.9])
        self.assertTrue(len(check_approved_param(new_p, info, indexes, active, do_log)) == 0)

        new_p = np.log10([1e-7, 1e-21*0.9, 1e-21*0.9])
        self.assertTrue("ks_size" in check_approved_param(new_p, info, indexes, active, do_log))
        new_p = np.log10([1e-7*0.9, 1e-21, 1e-21*0.9])
        self.assertTrue("Cn_size" in check_approved_param(new_p, info, indexes, active, do_log))
        new_p = np.log10([1e-7*0.9, 1e-21*0.9, 1e-21])
        self.assertTrue("Cp_size" in check_approved_param(new_p, info, indexes, active, do_log))

        # Should work without log
        info = {"names": ["ks", "Cn", "Cp"],
                'prior_dist': {'ks': (0, 1e-7), 'Cn': (0, 1e-21),
                               'Cp': (0, 1e-21)},
                "do_log": {"ks": 0, "Cn": 0, "Cp": 0},
                "active": {"ks": 1, "Cn": 1, "Cp": 1}}
        indexes = {name: info["names"].index(name) for name in info["names"]}
        do_log = np.array([info["do_log"][param] for param in info["names"]], dtype=bool)
        active = np.array([info["active"][name] for name in info["names"]], dtype=bool)
        new_p = np.array([1e-7*0.9, 1e-21*0.9, 1e-21*0.9])
        self.assertTrue(len(check_approved_param(new_p, info, indexes, active, do_log)) == 0)

        new_p = np.array([1e-7, 1e-21*0.9, 1e-21*0.9])
        self.assertTrue("ks_size" in check_approved_param(new_p, info, indexes, active, do_log))
        new_p = np.array([1e-7*0.9, 1e-21, 1e-21*0.9])
        self.assertTrue("Cn_size" in check_approved_param(new_p, info, indexes, active, do_log))
        new_p = np.array([1e-7*0.9, 1e-21*0.9, 1e-21])
        self.assertTrue("Cp_size" in check_approved_param(new_p, info, indexes, active, do_log))

        # Check p0, which has a size limit and must also be larger than n0
        info = {"names": ["n0", "p0"],
                'prior_dist': {'n0': (0, 1e19), 'p0': (0, 1e19)},
                "do_log": {"n0": 1, "p0": 1},
                "active": {"n0": 1, "p0": 1}}
        indexes = {name: info["names"].index(name) for name in info["names"]}
        do_log = np.array([info["do_log"][param] for param in info["names"]], dtype=bool)
        active = np.array([info["active"][name] for name in info["names"]], dtype=bool)
        new_p = np.log10([1e19 * 0.8, 1e19 * 0.9])
        self.assertTrue(len(check_approved_param(new_p, info, indexes, active, do_log)) == 0)

        new_p = np.log10([1e19 * 0.8, 1e19])
        self.assertTrue("p0_size" in check_approved_param(new_p, info, indexes, active, do_log))
        new_p = np.log10([1e19, 1e19 * 0.9])
        self.assertTrue("p0_greater" in check_approved_param(new_p, info, indexes, active, do_log))

        # Should work without log
        info = {"names": ["n0", "p0"],
                'prior_dist': {'n0': (0, 1e19), 'p0': (0, 1e19)},
                "do_log": {"n0": 0, "p0": 0},
                "active": {"n0": 1, "p0": 1}}
        indexes = {name: info["names"].index(name) for name in info["names"]}
        do_log = np.array([info["do_log"][param] for param in info["names"]], dtype=bool)
        active = np.array([info["active"][name] for name in info["names"]], dtype=bool)
        new_p = np.array([1e19 * 0.8, 1e19 * 0.9])
        self.assertTrue(len(check_approved_param(new_p, info, indexes, active, do_log)) == 0)

        new_p = np.array([1e19 * 0.8, 1e19])  # p0 too large
        self.assertTrue("p0_size" in check_approved_param(new_p, info, indexes, active, do_log))
        new_p = np.array([1e19, 1e19 * 0.9])  # p0 smaller than n0
        self.assertTrue("p0_greater" in check_approved_param(new_p, info, indexes, active, do_log))

        info_without_taus = {'names': ['tauQ', 'somethingelse'],
                             "do_log": {'tauQ': 1, 'somethingelse': 1},
                             "active": {'tauQ': 1, 'somethingelse': 1},
                             'prior_dist': {'tauQ': (-np.inf, np.inf),
                                            'somethingelse': (-np.inf, np.inf)}}
        indexes = {name: info_without_taus["names"].index(name) for name in info_without_taus["names"]}
        do_log = np.array([info_without_taus["do_log"][param] for param in info_without_taus["names"]], dtype=bool)
        active = np.array([info_without_taus["active"][name] for name in info_without_taus["names"]], dtype=bool)
        # No failures if criteria do not cover params
        new_p = np.log10([1, 1e10])
        self.assertTrue(
            len(check_approved_param(new_p, info_without_taus, indexes, active, do_log)) == 0)

    def test_select_next_params(self):
        # This function assigns a set of randomly generated values
        np.random.seed(1)
        param_names = ["a", "b", "c", "d"]

        do_log = {"a": 0, "b": 1, "c": 0, "d": 0}

        prior_dist = {"a": (-np.inf, np.inf),
                      "b": (-np.inf, np.inf),
                      "c": (-np.inf, np.inf),
                      "d": (-np.inf, np.inf), }

        initial_guesses = {"a": 0,
                           "b": 100,
                           "c": 0,
                           "d": 10, }

        active_params = {"a": 0,
                         "b": 1,
                         "c": 1,
                         "d": 1, }

        trial_move = {"a": 10,
                      "b": 0.1,
                      "c": 0,
                      "d": 1}

        param_info = {"names": param_names,
                      "prior_dist": prior_dist,
                      "init_guess": initial_guesses,
                      }

        indexes = {name: param_names.index(name) for name in param_names}
        state = [initial_guesses[name] for name in param_names]
        do_log = np.array([do_log[name] for name in param_names], dtype=bool)
        active_params = np.array([active_params[name] for name in param_names], dtype=bool)
        trial_move = np.array([trial_move[name] for name in param_names], dtype=float)
        # Try box selection
        new_state = select_next_params(state, param_info, indexes, active_params, trial_move, do_log, logger=self.logger)

        # Inactive and shouldn't change
        self.assertEqual(new_state[indexes["a"]], initial_guesses['a'])
        self.assertEqual(new_state[indexes["c"]], initial_guesses['c'])
        num_tests = 100
        for t in range(num_tests):
            new_state = select_next_params(state, param_info, indexes, active_params, trial_move, do_log, logger=self.logger)
            self.assertTrue(np.abs(np.log10(new_state[indexes["b"]]) - np.log10(initial_guesses['b'])) <= 0.1,
                            msg="Uniform step #{} failed: {} from mean {} and width 0.1".format(t, new_state[indexes["b"]], initial_guesses['b']))
            self.assertTrue(np.abs(new_state[indexes["d"]]-initial_guesses['d']) <= 1,
                            msg="Uniform step #{} failed: {} from mean {} and width 1".format(t, new_state[indexes["d"]], initial_guesses['d']))

        return

    def test_mu_constraint(self):
        # This function assigns a set of randomly generated values
        np.random.seed(1)
        param_names = ["mu_n", "mu_p"]

        do_log = {"mu_n": 1, "mu_p": 1}

        prior_dist = {"mu_n": (0.1, np.inf),
                      "mu_p": (0.1, np.inf),
                      }

        initial_guesses = {"mu_n": 20,
                           "mu_p": 20,
                           }

        active_params = {"mu_n": 1,
                         "mu_p": 1,
                         }
        
        trial_move = {"mu_n": 0.1,
                      "mu_p": 0.1}

        param_info = {"names": param_names,
                      "do_mu_constraint": (20, 3),
                      "prior_dist": prior_dist,
                      "init_guess": initial_guesses,
                      }

        indexes = {name: param_names.index(name) for name in param_names}
        state = [initial_guesses[name] for name in param_names]
        do_log = np.array([do_log[name] for name in param_names], dtype=bool)
        active_params = np.array([active_params[name] for name in param_names], dtype=bool)
        trial_move = np.array([trial_move[name] for name in param_names], dtype=float)

        for _ in range(10):
            new_state = select_next_params(state, param_info, indexes, active_params, trial_move, do_log, logger=self.logger)

            self.assertTrue(2 / (new_state[indexes["mu_n"]]**-1 + new_state[indexes["mu_p"]]**-1) <= 23)
            self.assertTrue(2 / (new_state[indexes["mu_n"]]**-1 + new_state[indexes["mu_p"]]**-1) >= 17)

        return

    def test_almost_equal(self):
        threshold = 1e-7

        # One element just too large
        a = np.array([1.0, 1.0])
        b = np.array([1.0, 1.0+threshold])
        self.assertFalse(almost_equal(b, a, threshold=threshold))

        # All elements just close enough
        b = np.array([1.0, 1.0+0.999*threshold])
        self.assertTrue(almost_equal(b, a, threshold=threshold))

        wrong_shape = np.array([1.0, 1.0, 1.0])
        self.assertFalse(almost_equal(a, wrong_shape))

    def test_roll_acceptance(self):
        np.random.seed(1)

        logratio = 1
        accept = np.zeros(100, dtype=bool)
        for i in range(len(accept)):
            accept[i] = roll_acceptance(logratio)

        self.assertTrue(all(accept))

        logratio = -1
        accept = np.zeros(10000, dtype=bool)
        for i in range(len(accept)):
            accept[i] = roll_acceptance(logratio)

        # Should accept around np.exp(-1) ~ 3679
        self.assertEqual(accept.sum(), 3695)
        return

    def test_unpack_simpar(self):
        # Length = [311,2000,311,2000, 311, 2000]
        Length = [2000]                            # Length (nm)
        L = [2 ** 7]                                # Spatial points
        meas_type = ["TRPL"]                          # measurement type

        simPar = {"lengths": Length,
                  "nx": L,
                  "meas_types": meas_type,
                  "num_meas": 1}

        thickness, nx, mtype = unpack_simpar(simPar, 0)
        self.assertEqual(Length[0], thickness)
        self.assertEqual(L[0], nx)
        self.assertEqual(meas_type[0], mtype)

        Length = np.array([311, 2000, 311, 2000, 311, 2000])
        L = [10, 20, 30, 40, 50, 60]
        meas_type = ["TRPL", "TRTS", "TRPL", "TRPL", "TRTS", "TRPL"]
        simPar = {"lengths": Length,
                  "nx": L,
                  "meas_types": meas_type,
                  "num_meas": 6}

        thickness, nx, mtype = unpack_simpar(simPar, 2)
        self.assertEqual(Length[2], thickness)
        self.assertEqual(L[2], nx)
        self.assertEqual(meas_type[2], mtype)
        return

    def test_set_min_y(self):
        t = np.linspace(0, 100, 100)
        vals = np.log10(np.exp(-t / 2)) # A slow decay
        sol = np.exp(-t) # A faster decay
        scale_shift = np.log10(0.1)
        sol, min_y, n_set = set_min_y(sol, vals, scale_shift)

        # Show that the min_y accounts for scale_shift
        np.testing.assert_equal(sol[len(sol)-n_set:], min_y)
        self.assertEqual(np.log10(min_y), min(vals) - scale_shift)

    def test_one_sim_ll_errata(self):
        # TODO: The next time odeint fails to do a simulation, upload it into this
        # test case
        return

if __name__ == "__main__":
    t = TestUtils()
    t.setUp()
    t.test_solve_traps()
