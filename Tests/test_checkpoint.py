import unittest
import numpy as np
import logging
import os
import pickle

from metropolis import main_metro_loop
from sim_utils import Ensemble
from bayes_io import make_dir

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
        trial_move = {param: 0.1 for param in param_names}

        param_info = {"names": param_names,
                      "active": active_params,
                      "unit_conversions": unit_conversions,
                      "do_log": do_log,
                      "prior_dist": prior_dist,
                      "init_guess": initial_guesses,
                      "trial_move": trial_move}

        starting_iter = 1
        num_iters = 10

        MCMC_fields = {"init_cond_path": None,
                       "measurement_path": None,
                       "output_path": "test-out",
                       "num_iters": num_iters,
                       "solver": ("solveivp",),
                       "model": "std",
                       "likel2move_ratio": {"TRPL": 500},
                       "log_pl": 1,
                       "hard_bounds": 1,
                       "checkpoint_dirname": os.path.join(".",
                                                          "test-Checkpoints"),
                       "checkpoint_header": "checkpoint",
                       "checkpoint_freq": 5,
                       # f"checkpointCPU{jobid}_30000.pik",
                       "load_checkpoint": None,
                       }

        self.MS_list = Ensemble(param_info, sim_info, MCMC_fields, num_iters, logger_name="Test0")
        self.ensemble_from_chpt = None

        # Dummy initial condition and measurement data
        self.MS_list.iniPar = np.array([np.ones(self.MS_list.sim_info["nx"][0]) * 1e16])
        self.MS_list.times = [np.linspace(0, 100, 100)]
        self.MS_list.vals = [np.ones(len(self.MS_list.times[0])) * -20]
        self.MS_list.uncs = [np.ones(len(self.MS_list.times[0])) * 0.04]
        self.MS_list.IRF_tables = {}

        make_dir(self.MS_list.ensemble_fields["checkpoint_dirname"])

        main_metro_loop(self.MS_list, starting_iter, num_iters,
                        need_initial_state=True,
                        verbose=False)
        return

    def test_checkpoint(self):
        with open(os.path.join(os.path.join(".", "test-Checkpoints"),
                               "checkpoint.pik"), 'rb') as ifstream:
            self.ensemble_from_chpt = pickle.load(ifstream)
            self.ensemble_from_chpt.ll_funcs = [None for _ in range(len(self.ensemble_from_chpt.MS))]
            np.random.set_state(self.ensemble_from_chpt.random_state)
            starting_iter = 5 + 1

        main_metro_loop(self.ensemble_from_chpt, starting_iter, 10,
                        need_initial_state=False,
                        verbose=False)
        self.ensemble_from_chpt.stop_logging(0)
        # Successful completion - checkpoints not needed anymore
        for chpt in os.listdir(os.path.join(".", "test-Checkpoints")):
            os.remove(os.path.join(os.path.join(".", "test-Checkpoints"), chpt))
        os.rmdir(os.path.join(".", "test-Checkpoints"))
        for log in os.listdir(os.path.join(".", "test-out")):
            os.remove(os.path.join(os.path.join(".", "test-out"), log))
        os.rmdir(os.path.join(".", "test-out"))

        # self.MS ran continuously from start to k=10 iterations;
        # self.MS_from_chpt ran from checkpoint at k=5 to k=10.
        # Both should yield identical MC walks

        np.testing.assert_equal(self.MS_list.H.accept, self.ensemble_from_chpt.H.accept)
        np.testing.assert_equal(self.MS_list.H.loglikelihood,
                                self.ensemble_from_chpt.H.loglikelihood)

        np.testing.assert_equal(self.MS_list.H.states, self.ensemble_from_chpt.H.states)

        # Saving a checkpoint should also unravel H.states into attributes
        for param in self.MS_list.ensemble_fields['names']:
            h1_mean = getattr(self.ensemble_from_chpt.H, f"mean_{param}")
            np.testing.assert_equal(h1_mean.shape[0], self.MS_list.H.states.shape[0])
            np.testing.assert_equal(h1_mean.shape[1], self.MS_list.H.states.shape[2])
        return


if __name__ == "__main__":
    t = TestUtils()
    t.setUp()
    t.test_checkpoint()
