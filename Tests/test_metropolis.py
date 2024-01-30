import unittest
import logging
import sys
sys.path.append("..")
import numpy as np
from scipy.integrate import trapz
from metropolis import all_signal_handler
from forward_solver import E_field, solve, calculate_PL, calculate_TRTS
from metropolis import roll_acceptance
from utils import unpack_simpar, set_min_y, search_c_grps

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

    def test_roll_acceptance(self):
        rng = np.random.default_rng(1)

        logratio = np.ones(100)
        accept = roll_acceptance(rng, logratio)

        self.assertTrue(all(accept))

        logratio = np.ones(10000) * -1
        accept = roll_acceptance(rng, logratio)

        # Should accept around np.exp(-1) ~ 3679
        self.assertEqual(accept.sum(), 3635)
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
