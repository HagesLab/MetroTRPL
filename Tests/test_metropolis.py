import unittest
import logging
import sys
sys.path.append("..")
import numpy as np
from scipy.integrate import trapz
from metropolis import all_signal_handler
from metropolis import E_field, solve, select_next_params
from metropolis import do_simulation, roll_acceptance, unpack_simpar
from metropolis import almost_equal
from metropolis import check_approved_param
from metropolis import run_iteration
from metropolis import search_c_grps
from metropolis import set_min_y
from sim_utils import Parameters, Grid, calculate_PL, calculate_TRTS
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
        param_info = {"names": ["n0", "p0", "eps"],
                      "active": {"n0": 1, "p0": 1, "eps": 1},
                      "init_guess": vals}

        pa = Parameters(param_info)
        nx = 10
        dx = 1
        # Test N=P
        N = np.zeros(nx)
        P = np.zeros(nx)
        E = E_field(N, P, pa, dx)

        np.testing.assert_equal(E, np.zeros_like(E))

        # Test N>P
        P = np.ones(nx) * 2
        N = np.ones(nx)
        E = E_field(N, P, pa, dx)

        np.testing.assert_equal(E[1:], np.ones_like(
            E[1:]) * q_C/eps0 * np.cumsum(N))

        # Test N<P
        P = np.ones(nx) * -1
        E = E_field(N, P, pa, dx)

        np.testing.assert_equal(
            E[1:], -2 * np.ones_like(E[1:]) * q_C/eps0 * np.cumsum(N))

        # Test corner_E != 0
        corner_E = 24
        E = E_field(N, P, pa, dx, corner_E=corner_E)

        np.testing.assert_equal(
            E[1:], -2 * np.ones_like(E[1:]) * q_C/eps0 * np.cumsum(N) + corner_E)

        # Test 2D
        N = np.ones((nx, nx+1))
        P = np.ones((nx, nx+1)) * -1
        E = E_field(N, P, pa, dx, corner_E=corner_E)
        np.testing.assert_equal(
            E[:, 1:], -2 * np.ones_like(E[:, 1:]) * q_C/eps0 * np.cumsum(N, axis=1) + corner_E)

        # Test n0, p0
        vals = {'n0': 1,
                'p0': 1,
                'eps': 1}
        param_info["init_guess"] = vals
        pa = Parameters(param_info)
        N = np.ones((nx, nx+1))
        P = np.ones((nx, nx+1))
        E = E_field(N, P, pa, dx, corner_E=corner_E)
        np.testing.assert_equal(E[1:], np.zeros_like(E[1:]) + corner_E)

        return

    def test_solve(self):
        # A high-inj, rad-only sample problem
        g = Grid()
        g.nx = 100
        g.dx = 10
        g.start_time = 0
        g.time = 100
        g.nt = 1000
        g.hmax = 4
        g.tSteps = np.linspace(g.start_time, g.time, g.nt+1)

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
        pa = Parameters(param_info)
        init_dN = 1e20 * np.ones(g.nx) # [cm^-3]

        # with solveivp
        test_PL, out_dN = solve(init_dN, g, pa, meas="TRPL", solver=("solveivp",),
                                RTOL=1e-10, ATOL=1e-14)
        # Calculate expected output in simulation units
        pa.apply_unit_conversions()
        rr = pa.ks * (out_dN * out_dN - pa.n0 * pa.p0)
        expected_out = trapz(rr, dx=g.dx) + rr[0]*g.dx/2 + rr[-1]*g.dx/2
        expected_out *= 1e23
        pa.apply_unit_conversions(reverse=True)
        self.assertAlmostEqual(test_PL[-1] / np.amax(test_PL[-1]), expected_out / np.amax(test_PL[-1]))

        # with odeint
        test_PL, out_DN = solve(init_dN, g, pa, meas="TRPL", solver=("odeint",),
                                RTOL=1e-10, ATOL=1e-14)
        pa.apply_unit_conversions()
        rr = pa.ks * (out_dN * out_dN - pa.n0 * pa.p0)
        expected_out = trapz(rr, dx=g.dx) + rr[0]*g.dx/2 + rr[-1]*g.dx/2
        expected_out *= 1e23
        pa.apply_unit_conversions(reverse=True)
        self.assertAlmostEqual(test_PL[-1] / np.amax(test_PL[-1]), expected_out / np.amax(test_PL[-1]),
            places=6)

        # No change should be seen by Parameters()
        for n in param_info["names"]:
            self.assertEqual(getattr(pa, n), vals[n])

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
        pa = Parameters(param_info)

        test_TRTS, out_dN = solve(
            init_dN, g, pa, meas="TRTS", solver=("solveivp",))
        pa.apply_unit_conversions()
        trts = q_C * (pa.mu_n * out_dN + pa.mu_p * out_dN)
        expected_out = trapz(trts, dx=g.dx) + trts[0]*g.dx/2 + trts[-1]*g.dx/2
        expected_out *= 1e9
        pa.apply_unit_conversions(reverse=True)
        self.assertAlmostEqual(test_TRTS[-1] / np.amax(test_TRTS[-1]), expected_out / np.amax(test_TRTS[-1]))

        for n in param_info["names"]:
            self.assertEqual(getattr(pa, n), vals[n])

        # try an undefined measurement
        with self.assertRaises(NotImplementedError):
            solve(init_dN, g, pa, meas="something else")

        # try an undefined solver
        with self.assertRaises(NotImplementedError):
            solve(init_dN, g, pa, meas="TRPL", solver=("somethign else",))

        return

    def test_solve_depletion(self):
        """ A high-inj, rad-only sample problem. Parameters are chosen such that
        the simulated PL will decay by about 2.5 orders of magnitude. """
        g = Grid()
        g.nx = 100
        g.dx = 10
        g.start_time = 0
        g.time = 100
        g.nt = 1000
        g.hmax = 4
        g.tSteps = np.linspace(g.start_time, g.time, g.nt+1)

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
        pa = Parameters(param_info)
        init_dN = 1e18 * np.ones(g.nx) # [cm^-3]

        PL0 = calculate_PL(g.dx * 1e-7, init_dN, init_dN, vals["ks"], vals["n0"], vals["p0"]) # in cm/s units
        # No or large range - PL runs to conclusion
        test_PL, out_dN = solve(init_dN, g, pa, meas="TRPL", solver=("solveivp",),
                                RTOL=1e-10, ATOL=1e-14)
        self.assertEqual(len(g.tSteps), len(test_PL))

        # Smaller range - PL decay stops early at PL_final, and signal over remaining time is set to PL_final
        g.min_y = PL0 * 1e-2
        test_PL, out_dN = solve(init_dN, g, pa, meas="TRPL", solver=("solveivp",),
                                RTOL=1e-10, ATOL=1e-14)

        self.assertTrue(min(test_PL) >= g.min_y)
        np.testing.assert_equal(test_PL[-10:], g.min_y)

        # Try a TRTS
        TRTS0 = calculate_TRTS(g.dx * 1e-7, init_dN, init_dN, vals["mu_n"], vals["mu_p"], vals["n0"], vals["p0"])
        # No or large range - PL runs to conclusion
        g.min_y = TRTS0 * 1e-10
        test_TRTS, out_dN = solve(init_dN, g, pa, meas="TRTS", solver=("solveivp",),
                                RTOL=1e-10, ATOL=1e-14)
        self.assertEqual(len(g.tSteps), len(test_TRTS))


        # Smaller range - TRTS is truncated
        g.min_y = TRTS0 * 1e-1
        test_TRTS, out_dN = solve(init_dN, g, pa, meas="TRTS", solver=("solveivp",),
                                RTOL=1e-10, ATOL=1e-14)
        self.assertTrue(min(test_TRTS) >= g.min_y)
        np.testing.assert_equal(test_TRTS[-10:], g.min_y)

        return

    def test_solve_traps(self):
        # A high-inj, rad-only sample problem using null parameters for the trap model
        # which should be equivalent to the std model
        g = Grid()
        g.nx = 100
        g.dx = 10
        g.start_time = 0
        g.time = 100
        g.nt = 1000
        g.hmax = 4
        g.tSteps = np.linspace(g.start_time, g.time, g.nt+1)

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
        pa = Parameters(param_info)
        init_dN = 1e20 * np.ones(g.nx) # [cm^-3]

        # with solveivp
        test_PL, out_dN = solve(init_dN, g, pa, meas="TRPL", solver=("solveivp",), model="traps",
                                RTOL=1e-10, ATOL=1e-14)
        # Calculate expected output in simulation units
        pa.apply_unit_conversions()
        rr = pa.ks * (out_dN * out_dN - pa.n0 * pa.p0)
        expected_out = trapz(rr, dx=g.dx) + rr[0]*g.dx/2 + rr[-1]*g.dx/2
        expected_out *= 1e23
        pa.apply_unit_conversions(reverse=True)
        self.assertAlmostEqual(test_PL[-1] / np.amax(test_PL[-1]), expected_out / np.amax(test_PL[-1]))

        return

    def test_solve_iniPar(self):
        # A high-inj, rad-only sample problem
        g = Grid()
        g.nx = 100
        g.dx = 10
        g.thickness = 1000 # in nm
        g.start_time = 0
        g.time = 100
        g.nt = 1000
        g.hmax = 4
        g.tSteps = np.linspace(g.start_time, g.time, g.nt+1)

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
        pa = Parameters(param_info)

        fluence = 1e15 # Fluence, alpha in cm units
        alpha = 6e4
        g.xSteps = np.linspace(g.dx / 2, g.thickness - g.dx/2, g.nx)

        init_dN = fluence * alpha * np.exp(-alpha * g.xSteps * 1e-7)  # In cm units

        PL_by_initvals, out_dN = solve(init_dN, g, pa, meas="TRPL", solver=("solveivp",),
                                       RTOL=1e-10, ATOL=1e-14)

        PL_by_initparams, out_dN = solve([fluence, alpha], g, pa, meas="TRPL", solver=("solveivp",),
                                         RTOL=1e-10, ATOL=1e-14)

        np.testing.assert_almost_equal(PL_by_initvals / np.amax(PL_by_initvals), PL_by_initparams / np.amax(PL_by_initvals))


    def test_approve_param(self):
        info = {'names': ['tauP', 'tauN', 'somethingelse'],
                'prior_dist': {'tauP': (0.1, np.inf), 'tauN': (0.1, np.inf),
                               'somethingelse': (-np.inf, np.inf)},
                'do_log': {'tauP': 1, 'tauN': 1, 'somethingelse': 1},
                "active": {'tauP': 1, 'tauN': 1, 'somethingelse': 1}}
        # taun, taup must be within 2 OM
        # Accepts new_p as log10
        # [n0, p0, mu_n, mu_p, ks, sf, sb, taun, taup, eps, m]
        new_p = np.log10([511, 511e2, 1])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)
        new_p = np.log10([511, 511e2+1,  1])
        self.assertTrue("tn_tp_close" in check_approved_param(new_p, info))

        # tn, tp size limit
        new_p = np.log10([0.11, 0.11, 1])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)
        new_p = np.log10([0.1, 0.11, 1])
        self.assertTrue("tauP_size" in check_approved_param(new_p, info))
        new_p = np.log10([0.11, 0.1, 1])
        self.assertTrue("tauN_size" in check_approved_param(new_p, info))

        # If params are inactive, they should not be checked
        info = {'names': ['tauP', 'tauN', 'somethingelse'],
                'prior_dist': {'tauP': (0.1, np.inf), 'tauN': (0.1, np.inf),
                               'somethingelse': (-np.inf, np.inf)},
                'do_log': {'tauP': 1, 'tauN': 1, 'somethingelse': 1},
                "active": {'tauP': 0, 'tauN': 0, 'somethingelse': 1}}
        new_p = np.log10([0.11, 0.1, 1])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)

        # These should still work if p is not logscaled
        info = {'names': ['tauP', 'tauN', 'somethingelse'],
                'prior_dist': {'tauP': (0.1, np.inf), 'tauN': (0.1, np.inf),
                               'somethingelse': (-np.inf, np.inf)},
                'do_log': {'tauP': 0, 'tauN': 0, 'somethingelse': 1},
                "active": {'tauP': 1, 'tauN': 1, 'somethingelse': 1}}
        new_p = np.array([511, 511e2, 1])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)
        new_p = np.array([511, 511e2+1,  1])
        self.assertTrue("tn_tp_close" in check_approved_param(new_p, info))

        new_p = np.array([0.11, 0.11, 1])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)
        new_p = np.array([0.1, 0.11, 1])
        self.assertTrue("tauP_size" in check_approved_param(new_p, info))
        new_p = np.array([0.11, 0.1, 1])
        self.assertTrue("tauN_size" in check_approved_param(new_p, info))

        # Check mu_n, mu_p, Sf, and Sb size limits
        info = {"names": ["mu_n", "mu_p", "Sf", "Sb"],
                'prior_dist': {'mu_n': (0.1, 1e6), 'mu_p': (0.1, 1e6),
                               'Sf': (0, 1e7), 'Sb': (0, 1e7)},
                'do_log': {"mu_n": 1, "mu_p": 1, "Sf": 1, "Sb": 1},
                "active": {"mu_n": 1, "mu_p": 1, "Sf": 1, "Sb": 1}}
        new_p = np.log10([1e6-1, 1e6-1, 1e7-1, 1e7-1])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)

        new_p = np.log10([1e6, 1e6-1, 1e7-1, 1e7-1])
        self.assertTrue("mu_n_size" in check_approved_param(new_p, info))
        new_p = np.log10([1e6-1, 1e6, 1e7-1, 1e7-1])
        self.assertTrue("mu_p_size" in check_approved_param(new_p, info))
        new_p = np.log10([1e6-1, 1e6-1, 1e7, 1e7-1])
        self.assertTrue("Sf_size" in check_approved_param(new_p, info))
        new_p = np.log10([1e6-1, 1e6-1, 1e7-1, 1e7])
        self.assertTrue("Sb_size" in check_approved_param(new_p, info))

        # These should still work if p is not logscaled
        info = {"names": ["mu_n", "mu_p", "Sf", "Sb"],
                'prior_dist': {'mu_n': (0.1, 1e6), 'mu_p': (0.1, 1e6),
                               'Sf': (0, 1e7), 'Sb': (0, 1e7)},
                'do_log': {"mu_n": 0, "mu_p": 0, "Sf": 0, "Sb": 0},
                "active": {"mu_n": 1, "mu_p": 1, "Sf": 1, "Sb": 1}}
        new_p = np.array([1e6-1, 1e6-1, 1e7-1, 1e7-1])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)

        new_p = np.array([1e6, 1e6-1, 1e7-1, 1e7-1])
        self.assertTrue("mu_n_size" in check_approved_param(new_p, info))
        new_p = np.array([1e6-1, 1e6, 1e7-1, 1e7-1])
        self.assertTrue("mu_p_size" in check_approved_param(new_p, info))
        new_p = np.array([1e6-1, 1e6-1, 1e7, 1e7-1])
        self.assertTrue("Sf_size" in check_approved_param(new_p, info))
        new_p = np.array([1e6-1, 1e6-1, 1e7-1, 1e7])
        self.assertTrue("Sb_size" in check_approved_param(new_p, info))

        # Check ks, Cn, Cp size limits
        info = {"names": ["ks", "Cn", "Cp"],
                'prior_dist': {'ks': (0, 1e-7), 'Cn': (0, 1e-21),
                               'Cp': (0, 1e-21)},
                "do_log": {"ks": 1, "Cn": 1, "Cp": 1},
                "active": {"ks": 1, "Cn": 1, "Cp": 1}}
        new_p = np.log10([1e-7*0.9, 1e-21*0.9, 1e-21*0.9])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)

        new_p = np.log10([1e-7, 1e-21*0.9, 1e-21*0.9])
        self.assertTrue("ks_size" in check_approved_param(new_p, info))
        new_p = np.log10([1e-7*0.9, 1e-21, 1e-21*0.9])
        self.assertTrue("Cn_size" in check_approved_param(new_p, info))
        new_p = np.log10([1e-7*0.9, 1e-21*0.9, 1e-21])
        self.assertTrue("Cp_size" in check_approved_param(new_p, info))

        # Should work without log
        info = {"names": ["ks", "Cn", "Cp"],
                'prior_dist': {'ks': (0, 1e-7), 'Cn': (0, 1e-21),
                               'Cp': (0, 1e-21)},
                "do_log": {"ks": 0, "Cn": 0, "Cp": 0},
                "active": {"ks": 1, "Cn": 1, "Cp": 1}}
        new_p = np.array([1e-7*0.9, 1e-21*0.9, 1e-21*0.9])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)

        new_p = np.array([1e-7, 1e-21*0.9, 1e-21*0.9])
        self.assertTrue("ks_size" in check_approved_param(new_p, info))
        new_p = np.array([1e-7*0.9, 1e-21, 1e-21*0.9])
        self.assertTrue("Cn_size" in check_approved_param(new_p, info))
        new_p = np.array([1e-7*0.9, 1e-21*0.9, 1e-21])
        self.assertTrue("Cp_size" in check_approved_param(new_p, info))

        # Check p0, which has a size limit and must also be larger than n0
        info = {"names": ["n0", "p0"],
                'prior_dist': {'n0': (0, 1e19), 'p0': (0, 1e19)},
                "do_log": {"n0": 1, "p0": 1},
                "active": {"n0": 1, "p0": 1}}
        new_p = np.log10([1e19 * 0.8, 1e19 * 0.9])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)

        new_p = np.log10([1e19 * 0.8, 1e19])
        self.assertTrue("p0_size" in check_approved_param(new_p, info))
        new_p = np.log10([1e19, 1e19 * 0.9])
        self.assertTrue("p0_greater" in check_approved_param(new_p, info))

        # Should work without log
        info = {"names": ["n0", "p0"],
                'prior_dist': {'n0': (0, 1e19), 'p0': (0, 1e19)},
                "do_log": {"n0": 0, "p0": 0},
                "active": {"n0": 1, "p0": 1}}
        new_p = np.array([1e19 * 0.8, 1e19 * 0.9])
        self.assertTrue(len(check_approved_param(new_p, info)) == 0)

        new_p = np.array([1e19 * 0.8, 1e19])  # p0 too large
        self.assertTrue("p0_size" in check_approved_param(new_p, info))
        new_p = np.array([1e19, 1e19 * 0.9])  # p0 smaller than n0
        self.assertTrue("p0_greater" in check_approved_param(new_p, info))

        info_without_taus = {'names': ['tauQ', 'somethingelse'],
                             "do_log": {'tauQ': 1, 'somethingelse': 1},
                             "active": {'tauQ': 1, 'somethingelse': 1},
                             'prior_dist': {'tauQ': (-np.inf, np.inf),
                                            'somethingelse': (-np.inf, np.inf)}}
        # No failures if criteria do not cover params
        new_p = np.log10([1, 1e10])
        self.assertTrue(
            len(check_approved_param(new_p, info_without_taus)) == 0)

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

        param_info = {"active": active_params,
                      "do_log": do_log,
                      "names": param_names,
                      "prior_dist": prior_dist,
                      "init_guess": initial_guesses,
                      "trial_move": trial_move}

        pa = Parameters(param_info)
        means = Parameters(param_info)

        # Try box selection
        select_next_params(pa, means, param_info, logger=self.logger)

        # Inactive and shouldn't change
        self.assertEqual(pa.a, initial_guesses['a'])
        self.assertEqual(pa.c, initial_guesses['c'])
        num_tests = 100
        for t in range(num_tests):
            select_next_params(pa, means, param_info, logger=self.logger)
            self.assertTrue(np.abs(np.log10(pa.b) - np.log10(initial_guesses['b'])) <= 0.1,
                            msg="Uniform step #{} failed: {} from mean {} and width 0.1".format(t, pa.b, initial_guesses['b']))
            self.assertTrue(np.abs(pa.d-initial_guesses['d']) <= 1,
                            msg="Uniform step #{} failed: {} from mean {} and width 1".format(t, pa.d, initial_guesses['d']))

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

        param_info = {"active": active_params,
                      "do_log": do_log,
                      "names": param_names,
                      "do_mu_constraint": (20, 3),
                      "prior_dist": prior_dist,
                      "init_guess": initial_guesses,
                      "trial_move": trial_move}

        pa = Parameters(param_info)
        means = Parameters(param_info)

        for i in range(10):
            select_next_params(pa, means, param_info, logger=self.logger)

            self.assertTrue(2 / (pa.mu_n**-1 + pa.mu_p**-1) <= 23)
            self.assertTrue(2 / (pa.mu_n**-1 + pa.mu_p**-1) >= 17)

        return

    def test_do_simulation(self):
        # Make sure the simulation starts at 0, even if the experimental data doesn't
        unit_conversions = {"n0": ((1e-7) ** 3), "p0": ((1e-7) ** 3),
                            "mu_n": ((1e7) ** 2) / (1e9), "mu_p": ((1e7) ** 2) / (1e9),
                            "ks": ((1e7) ** 3) / (1e9), "Sf": 1e-2, "Sb": 1e-2}

        param_info = {"names": ["n0", "p0", "mu_n", "mu_p", "ks", "Cn", "Cp", "tauN", "tauP",
                                "Sf", "Sb", "eps", "Tm"],
                      "active": {"n0": 0, "p0": 1,
                                 "mu_n": 0, "mu_p": 0,
                                 "ks": 1, "Sf": 1, "Sb": 1,
                                 "Cn": 0, "Cp": 0,
                                 "tauN": 1, "tauP": 1, "eps": 0, "Tm": 0},
                      "unit_conversions": unit_conversions}
        vals = {'n0': 1e8,
                'p0': 3e15,
                'mu_n': 20,
                'mu_p': 100,
                "ks": 1e-11,
                "Cn": 0, "Cp": 0,
                'tauN': 120,
                'tauP': 200,
                'Sf': 5,
                'Sb': 20,
                'eps': 1,
                "Tm": 300}

        param_info["init_guess"] = vals
        pa = Parameters(param_info)

        thickness = 1000
        nx = 100
        times = np.linspace(10, 100, 901)

        iniPar = np.logspace(19, 14, nx)
        tSteps, sol = do_simulation(pa, thickness, nx, iniPar, times, hmax=4)
        np.testing.assert_equal(tSteps[0], 0)

        times = np.linspace(0, 100, 1001)
        tSteps2, sol2 = do_simulation(pa, thickness, nx, iniPar, times, hmax=4)
        np.testing.assert_equal(sol[0], sol2[0])
        np.testing.assert_equal(sol[-1], sol2[-1])

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

    def test_run_iter(self):
        # Will basically need to set up a full simulation for this
        np.random.seed(42)
        Length = [2000, 2000]                            # Length (nm)
        L = [2 ** 7, 2 ** 7]                                # Spatial point
        mtype = ["TRPL", "TRPL"]
        simPar = {"lengths": Length,
                  "nx": L,
                  "meas_types": mtype,
                  "num_meas": 2}

        iniPar = [1e15 * np.ones(L[0]), 1e16 * np.ones(L[1])]

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
                     "hmax": 4, "rtol": 1e-5, "atol": 1e-8,
                     "solver": ("solveivp",),
                     "model": "std"}

        p = Parameters(param_info)

        nt = 1000
        times = [np.linspace(0, 100, nt+1), np.linspace(0, 100, nt+1)]
        vals = [np.ones(nt+1) * 23, np.ones(nt+1) * 23]
        uncs = [np.ones(nt+1) * 1e-99, np.ones(nt+1) * 1e-99]
        logll = run_iteration(p, simPar, iniPar, times, vals, uncs, None,
                              sim_flags, logger=self.logger)

        np.testing.assert_almost_equal(
            logll, np.sum([-59340.105083, -32560.139058]), decimal=0)  # rtol=1e-5

    def test_run_iter_depletion(self):
        """Prove that the truncation allows likelihood of two carrier-depleting simulations to be reliably determined."""
        np.random.seed(42)
        Length = [2000]                            # Length (nm)
        L = [2 ** 7]                                # Spatial point
        mtype = ["TRPL"]
        simPar = {"lengths": Length, "nx": L, "meas_types": mtype, "num_meas": 1}

        iniPar = [1e15 * np.ones(L[0])] # PL(t=0) ~ 2e15

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
                     "hmax": 4, "rtol": 1e-5, "atol": 1e-8,
                     "solver": ("solveivp",),
                     "force_min_y": True,
                     "model": "std"}

        p = Parameters(param_info)

        nt = 1000
        times = [np.linspace(0, 100, nt+1)]
        vals = [np.log10(2e14 * np.exp(-times[0] / 8))]
        uncs = [np.ones(nt+1) * 1e-99]
        logll1 = run_iteration(p, simPar, iniPar, times, vals, uncs, None,
                               sim_flags, logger=self.logger)

        # A small move toward the true lifetime of 10 makes the likelihood better
        # Without min_y truncation, the likelihoods were so small they weren't even comparable
        param_info["init_guess"]["tauN"] = 4.01
        param_info["init_guess"]["tauP"] = 4.01

        p2 = Parameters(param_info)
        logll2 = run_iteration(p2, simPar, iniPar, times, vals, uncs, None,
                               sim_flags, logger=self.logger)
        self.assertTrue(logll2 > logll1)

    def test_set_min_y(self):
        t = np.linspace(0, 100, 100)
        vals = np.log10(np.exp(-t / 2)) # A slow decay
        sol = np.exp(-t) # A faster decay
        scale_shift = np.log10(0.1)
        sol, min_y, n_set = set_min_y(sol, vals, scale_shift)

        # Show that the min_y accounts for scale_shift
        np.testing.assert_equal(sol[len(sol)-n_set:], min_y)
        self.assertEqual(np.log10(min_y), min(vals) - scale_shift)

    def test_run_iter_cutoff(self):
        # Same as test_run_iter, only "experimental" data is
        # truncated at [50,100] instead of [0,100].
        # Half as many points means the likelihood should be reduced to about half.
        np.random.seed(42)
        Length = [2000, 2000]                            # Length (nm)
        L = [2 ** 7, 2 ** 7]                                # Spatial point
        mtype = ["TRPL", "TRPL"]
        simPar = {"lengths": Length,
                  "nx": L,
                  "meas_types": mtype,
                  "num_meas": 2}

        iniPar = [1e15 * np.ones(L[0]), 1e16 * np.ones(L[1])]

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
                     "hmax": 4, "rtol": 1e-5, "atol": 1e-8,
                     "solver": ("solveivp",),
                     "model": "std"}

        p = Parameters(param_info)

        nt = 500
        times = [np.linspace(50, 100, nt+1), np.linspace(50, 100, nt+1)]
        vals = [np.ones(nt+1) * 23, np.ones(nt+1) * 23]
        uncs = [np.ones(nt+1) * 1e-99, np.ones(nt+1) * 1e-99]
        logll = run_iteration(p, simPar, iniPar, times, vals, uncs, None,
                              sim_flags, logger=self.logger)

        # First iter; auto-accept
        np.testing.assert_almost_equal(
            logll, np.sum([-29701, -16309]), decimal=0)  # rtol=1e-5


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

        iniPar = [1e15 * np.ones(L[0]), 1e16 * np.ones(L[1])]

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
                     "hmax": 4, "rtol": 1e-5, "atol": 1e-8,
                     "scale_factor": (0.02, [0, 1, 2, 3, 4, 5], [(0, 2, 4), (1, 3, 5)]),
                     "solver": ("solveivp",),
                     "model": "std"}

        p = Parameters(param_info)
        # By setting individual scale factors in this simple case the likelihood can be made perfect
        setattr(p, "_s0", 2e-17 ** -1)
        setattr(p, "_s1", 2e-15 ** -1)
        nt = 1000
        times = [np.linspace(0, 100, nt+1), np.linspace(0, 100, nt+1)]
        vals = [np.ones(nt+1) * 23, np.ones(nt+1) * 23]
        uncs = [np.ones(nt+1) * 1e-99, np.ones(nt+1) * 1e-99]

        logll = run_iteration(p, simPar, iniPar, times, vals, uncs, None,
                              sim_flags, logger=self.logger)

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

        iniPar = [1e15 * np.ones(L[0]), 1e15 * np.ones(L[1])]

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

        sim_flags = {"current_sigma": {"TRPL": 1, "TRTS": 10},
                     "hmax": 4, "rtol": 1e-5, "atol": 1e-8,
                     "solver": ("solveivp",),
                     "model": "std"}

        p = Parameters(param_info)

        nt = 1000
        times = [np.linspace(0, 100, nt+1), np.linspace(0, 100, nt+1)]
        vals = [np.ones(nt+1) * 23, np.ones(nt+1) * -2]
        uncs = [np.ones(nt+1) * 1e-99, np.ones(nt+1) * 1e-99]
        logll = run_iteration(p, simPar, iniPar, times, vals, uncs, None,
                              sim_flags, logger=self.logger)

        # First iter; auto-accept
        np.testing.assert_almost_equal(
            logll, np.sum([-59340.105083, -517.98]), decimal=0)  # rtol=1e-5

    def test_one_sim_ll_errata(self):
        # TODO: The next time odeint fails to do a simulation, upload it into this
        # test case
        return
        # np.random.seed(42)
        # Length  = 2000                            # Length (nm)
        # L   = 2 ** 7                                # Spatial points
        # plT = 1                                  # Set PL interval (dt)
        # pT  = (0,1,3,10,30,100)                   # Set plot intervals (%)
        # tol = 7                                   # Convergence tolerance
        # MAX = 10000                                  # Max iterations

        # simPar = [Length, -1, L, -1, plT, pT, tol, MAX]

        # iniPar = [np.logspace(20,1,L), 1e16 * np.ones(L)]

        # param_names = ["n0", "p0", "mu_n", "mu_p", "ks", "Cn", "Cp", "Tm",
        #                "Sf", "Sb", "tauN", "tauP", "eps", "m"]
        # unit_conversions = {"n0":((1e-7) ** 3), "p0":((1e-7) ** 3),
        #                     "mu_n":((1e7) ** 2) / (1e9), "mu_p":((1e7) ** 2) / (1e9),
        #                     "ks":((1e7) ** 3) / (1e9), "Sf":1e-2, "Sb":1e-2}

        # # Iterations should proceed independent of which params are actively iterated,
        # # as all params are presumably needed to complete the simulation
        # param_info = {"names":param_names,
        #               "unit_conversions":unit_conversions,
        #               "active":{name:0 for name in param_names}}
        # initial_guess = {"n0":1e8,
        #                  "p0":1e12,
        #                  "mu_n":2,
        #                  "mu_p":2,
        #                  "ks":1e-11,
        #                  "Sf":1000,
        #                  "Sb":1e4,
        #                  "Cn":0,
        #                  "Cp":0,
        #                  "Tm":300,
        #                  "tauN":10,
        #                  "tauP":10,
        #                  "eps":10,
        #                  "m":0}

        # sim_flags = {"hmax":MIN_HMAX * 4, "rtol":1e-10, "atol":1e-10,
        #              "measurement":"TRPL",
        #              "solver":("odeint",)}

        # nt = 100
        # i = 0
        # times = [np.linspace(0, 10, nt+1), np.linspace(0, 10, nt+1)]
        # vals = [np.zeros(nt+1), np.zeros(nt+1)]

        # running_hmax = [sim_flags["hmax"]] * len(iniPar)
        # p = Parameters(param_info, initial_guess)
        # p.apply_unit_conversions(param_info)

        # with self.assertLogs() as captured:
        #     one_sim_likelihood(p, simPar, running_hmax, sim_flags, self.logger,
        #                        (i, iniPar[i], times[i], vals[i]))

        # # We can force a simulation failure by using odeint + rapid recombination
        # self.assertEqual(len(captured.records), 8)
        # self.assertEqual(captured.records[1].getMessage(),
        #                  f"{i}: Carriers depleted!")
        # self.assertEqual(captured.records[2].getMessage(),
        #                  f"{i}: Retrying hmax={MIN_HMAX * 2}")

if __name__ == "__main__":
    t = TestUtils()
    t.setUp()
    t.test_run_iter()
