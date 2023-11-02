import unittest
import numpy as np
import logging
import os
import pickle

from metropolis import main_metro_loop
from sim_utils import MetroState
from bayes_io import make_dir, clear_checkpoint_dir


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger()
        num_measurements = 1
        Length = [2000]
        # Length = [2000] * 3             # Length (nm)
        L = [128]                         # Spatial points
        measurement_types = ["TRPL"]
        sim_info = {"lengths": Length,
                    "nx": L,
                    "meas_types": measurement_types,
                    "num_meas": num_measurements}

        param_names = ["n0", "p0", "mu_n", "mu_p", "ks", "Cn", "Cp",
                       "Sf", "Sb", "tauN", "tauP", "eps", "Tm", "m"]

        unit_conversions = {"n0": ((1e-7) ** 3), "p0": ((1e-7) ** 3),
                            "mu_n": ((1e7) ** 2) / (1e9),
                            "mu_p": ((1e7) ** 2) / (1e9),
                            "ks": ((1e7) ** 3) / (1e9),
                            "Cn": ((1e7) ** 6) / (1e9),
                            "Cp": ((1e7) ** 6) / (1e9),
                            "Sf": 1e-2, "Sb": 1e-2, "Tm": 1}

        do_log = {"n0": 1, "p0": 1, "mu_n": 1, "mu_p": 1, "ks": 1, "Cn": 1,
                  "Cp": 1,
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
                           "p0": 3e15,
                           "mu_n": 20,
                           "mu_p": 20,
                           "ks": 4.8e-11,
                           "Cn": 4.4e-29,
                           "Cp": 4.4e-29,
                           "Sf": 10,
                           "Sb": 10,
                           "tauN": 511,
                           "tauP": 871,
                           "eps": 10,
                           "Tm": 300,
                           "m": 1}
        # initial_guesses["tauP"] = initial_guesses["tauN"]

        active_params = {"n0": 0,
                         "p0": 1,
                         "mu_n": 1,
                         "mu_p": 1,
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
        # Proposal function search widths
        initial_variance = {param: 0.1 for param in param_names}

        param_info = {"names": param_names,
                      "active": active_params,
                      "unit_conversions": unit_conversions,
                      "do_log": do_log,
                      "prior_dist": prior_dist,
                      "init_guess": initial_guesses,
                      "init_variance": initial_variance}

        starting_iter = 1
        num_iters = 10

        MCMC_fields = {"init_cond_path": None,
                       "measurement_path": None,
                       "output_path": "test-out",
                       "num_iters": num_iters,
                       "solver": ("solveivp",),
                       "model": "std",
                       "likel2variance_ratio": {"TRPL": 500},
                       "log_pl": 1,
                       "self_normalize": None,
                       "proposal_function": "box",
                       "one_param_at_a_time": 0,
                       "hard_bounds": 1,
                       "checkpoint_dirname": os.path.join(".",
                                                          "test-Checkpoints"),
                       "checkpoint_header": "checkpoint",
                       "checkpoint_freq": 5,
                       # f"checkpointCPU{jobid}_30000.pik",
                       "load_checkpoint": None,
                       }

        annealing_step = 2000
        min_sigma = 0.01
        MCMC_fields["annealing"] = ({m:max(initial_variance.values()) * MCMC_fields["likel2variance_ratio"][m]
                                    for m in sim_info["meas_types"]},
                                    annealing_step,
                                    {m:min_sigma for m in sim_info["meas_types"]})

        self.MS = MetroState(param_info, MCMC_fields, num_iters)

        self.MS.sim_info = sim_info
        # Dummy initial condition and measurement data
        self.MS.iniPar = [np.ones(self.MS.sim_info["nx"][0]) * 1e16]
        self.MS.times = [np.linspace(0, 100, 100)]
        self.MS.vals = [np.ones(len(self.MS.times[0])) * -20]
        self.MS.uncs = [np.ones(len(self.MS.times[0])) * 0.04]
        self.MS.IRF_tables = None

        make_dir(self.MS.MCMC_fields["checkpoint_dirname"])
        clear_checkpoint_dir(MCMC_fields)

        main_metro_loop(self.MS, starting_iter, num_iters,
                        need_initial_state=True, logger=self.logger,
                        verbose=True)
        return

    def test_checkpoint(self):
        with open(os.path.join(os.path.join(".", "test-Checkpoints"),
                               "checkpoint.pik"), 'rb') as ifstream:
            self.MS_from_chpt = pickle.load(ifstream)
            np.random.set_state(self.MS_from_chpt.random_state)
            starting_iter = 5 + 1

        main_metro_loop(self.MS_from_chpt, starting_iter, 10,
                        need_initial_state=False, logger=self.logger,
                        verbose=True)

        # Successful completion - checkpoints not needed anymore
        for chpt in os.listdir(os.path.join(".", "test-Checkpoints")):
            os.remove(os.path.join(os.path.join(".", "test-Checkpoints"), chpt))
        os.rmdir(os.path.join(".", "test-Checkpoints"))

        # self.MS ran continuously from start to k=10 iterations;
        # self.MS_from_chpt ran from checkpoint at k=5 to k=10.
        # Both should yield identical MC walks

        np.testing.assert_equal(self.MS.H.accept, self.MS_from_chpt.H.accept)
        np.testing.assert_equal(self.MS.H.loglikelihood,
                                self.MS_from_chpt.H.loglikelihood)

        for param in self.MS.param_info['names']:
            h1_mean = getattr(self.MS.H, f"mean_{param}")
            h2_mean = getattr(self.MS_from_chpt.H, f"mean_{param}")
            np.testing.assert_equal(h1_mean, h2_mean)
        return


if __name__ == "__main__":
    t = TestUtils()
    t.setUp()
    t.test_checkpoint()
