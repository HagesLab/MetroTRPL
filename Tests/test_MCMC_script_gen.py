import unittest
import numpy as np
import os

from bayes_io import generate_config_script_file, read_config_script_file
eps0 = 8.854 * 1e-12 * 1e-9  # [C / V m] to {C / V nm}
q = 1.0  # [e]
q_C = 1.602e-19  # [C]
kB = 8.61773e-5  # [eV / K]


class TestUtils(unittest.TestCase):

    def setUp(self):
        num_measurements = 3
        Length = [2000, 2000, 2000]                           # Length (nm)
        L = [128, 128, 128]                                # Spatial points
        measurement_types = ["TRPL", "TRPL", "TRPL"]
        self.simPar = {"lengths": Length,
                       "nx": L,
                       "meas_types": measurement_types,
                       "num_meas": num_measurements}

        param_names = ["n0", "p0", "mu_n", "mu_p", "ks", "Cn", "Cp",
                       "Sf", "Sb", "tauN", "tauP", "eps", "Tm", "m"]
        unit_conversions = {"n0": ((1e-7) ** 3), "p0": ((1e-7) ** 3),
                            "mu_n": ((1e7) ** 2) / (1e9),
                            "mu_p": ((1e7) ** 2) / (1e9),
                            "ks": ((1e7) ** 3) / (1e9),
                            "Cn": ((1e7) ** 6) / (1e9), "Cp": ((1e7) ** 6) / (1e9),
                            "Sf": 1e-2, "Sb": 1e-2, "Tm": 1}
        do_log = {"n0": 1, "p0": 1, "mu_n": 1, "mu_p": 1, "ks": 1, "Cn": 1, "Cp": 1,
                  "Sf": 1, "Sb": 1, "tauN": 1, "tauP": 1, "eps": 1, "Tm": 1,
                  "m": 1}

        prior_dist = {"n0": (0, np.inf),
                      "p0": (1e14, 1e16),
                      "mu_n": (1e0, 1e2),
                      "mu_p": (1e0, 1e2),
                      "ks": (1e-11, 1e-9),
                      "Cn": (1e-29, 1e-27),
                      "Cp": (1e-29, 1e-27),
                      "Sf": (1e-4, 1e4),
                      "Sb": (1e-4, 1e4),
                      "tauN": (1, 1500),
                      "tauP": (1, 3000),
                      "eps": (0, np.inf),
                      "Tm": (0, np.inf),
                      "m": (-np.inf, np.inf)}

        initial_guesses = {"n0": 1e8,
                           "p0": 1e14,
                           "mu_n": 20,
                           "mu_p": 20,
                           "ks": 5.958e-11,
                           "Cn": 1e-29,
                           "Cp": 1e-29,
                           "Sf": 2.1e2,
                           "Sb": 2.665e2,
                           "tauN": 4.708e2,
                           "tauP": 1.961e2,
                           "eps": 10,
                           "Tm": 300,
                           "m": 1}

        active_params = {"n0": 0,
                         "p0": 1,
                         "mu_n": 0,
                         "mu_p": 0,
                         "ks": 1,
                         "Cn": 1,
                         "Cp": 1,
                         "Sf": 1,
                         "Sb": 1,
                         "tauN": 1,
                         "tauP": 1,
                         "eps": 0,
                         "Tm": 0,
                         "m": 0}

        # Other options
        initial_variance = {"n0": 1e-2,
                            "p0": 1e-2,
                            "mu_n": 1e-2,
                            "mu_p": 1e-2,
                            "ks": 1e-2,
                            "Cn": 1e-2,
                            "Cp": 1e-2,
                            "Sf": 1e-2,
                            "Sb": 1e-2,
                            "tauN": 1e-2,
                            "tauP": 1e-2,
                            "eps": 1e-2,
                            "Tm": 1e-2,
                            "m": 1e-2}

        self.param_info = {"names": param_names,
                           "active": active_params,
                           "unit_conversions": unit_conversions,
                           "do_log": do_log,
                           "prior_dist": prior_dist,
                           "init_guess": initial_guesses,
                           "init_variance": initial_variance}

        self.measurement_fields = {"time_cutoff": [0, np.inf],
                                   "select_obs_sets": [0, 1, 2],
                                   "resample": 2,
                                   "noise_level": None}

        output_path = os.path.join("MCMC", "Example_output_path")
        self.MCMC_fields = {"init_cond_path": os.path.join("MCMC", "Example_IC_path"),
                            "measurement_path": os.path.join("MCMC", "Example_meas_path"),
                            "output_path": output_path,
                            "num_iters": 25000,
                            "solver": ("solveivp",),
                            "model": "std",
                            "rtol": 1e-7,
                            "atol": 1e-10,
                            "hmax": 4,
                            "likel2variance_ratio": {"TRPL": 500},
                            "override_equal_mu": 0,
                            "override_equal_s": 0,
                            "log_pl": 1,
                            "self_normalize": None,
                            "proposal_function": "box",
                            "checkpoint_dirname": os.path.join(output_path, "Checkpoints"),
                            "checkpoint_header": "CPU0",
                            "checkpoint_freq": 12000,
                            "load_checkpoint": None,
                            }
        return

    def test_create_script(self):
        # Verify that we read out what we originally write in
        where_script = os.path.join("Tests", "testfiles", "testmcmc.txt")
        generate_config_script_file(where_script, self.simPar, self.param_info,
                                    self.measurement_fields, self.MCMC_fields,
                                    verbose=True)

        simPar, param_info, measurement_fields, MCMC_fields = read_config_script_file(
            where_script)

        for k in self.simPar.keys():
            np.testing.assert_equal(simPar[k], self.simPar[k])

        for k in self.param_info.keys():
            if isinstance(self.param_info[k], dict):
                for kk in self.param_info[k].keys():
                    self.assertEqual(param_info[k][kk], self.param_info[k][kk])

        for k in self.measurement_fields.keys():
            if isinstance(self.measurement_fields[k], (list, tuple, np.ndarray)):
                np.testing.assert_equal(
                    measurement_fields[k], self.measurement_fields[k])
            else:
                self.assertEqual(
                    measurement_fields[k], self.measurement_fields[k])

        for k in self.MCMC_fields.keys():
            if isinstance(self.MCMC_fields[k], (list, tuple, np.ndarray)):
                np.testing.assert_equal(MCMC_fields[k], self.MCMC_fields[k])
            else:
                self.assertEqual(
                    MCMC_fields[k], self.MCMC_fields[k],
                    msg=f"Read script failed: {k} mismatch")
        return


if __name__ == "__main__":
    t = TestUtils()
    t.setUp()
    t.test_create_script()
