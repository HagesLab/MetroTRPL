import unittest
import numpy as np
import logging
import os
import pickle

from metropolis import main_metro_loop_serial
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

        self.num_iters = 10

        MCMC_fields = {"init_cond_path": None,
                       "measurement_path": None,
                       "output_path": "test-out",
                       "num_iters": self.num_iters,
                       "solver": ("solveivp",),
                       "model": "std",
                       "likel2move_ratio": {"TRPL": 500},
                       "ini_mode": "density",
                       "log_y": 1,
                       "hard_bounds": 1,
                       "checkpoint_freq": 5,
                       # f"checkpointCPU{jobid}_30000.pik",
                       "load_checkpoint": None,
                       }

        self.export_path = "CPU0-test.pik"
        self.MS_list = Ensemble(param_info, sim_info, MCMC_fields, self.num_iters)
        self.ensemble_from_chpt = None

        # Dummy initial condition and measurement data
        self.MS_list.ensemble_fields["_init_params"] = np.array([np.ones(self.MS_list.ensemble_fields["_sim_info"]["nx"][0]) * 1e16])
        self.MS_list.ensemble_fields["_times"] = [np.linspace(0, 100, 100)]
        self.MS_list.ensemble_fields["_vals"] = [np.ones(len(self.MS_list.ensemble_fields["_times"][0])) * -20]
        self.MS_list.ensemble_fields["_uncs"] = [np.ones(len(self.MS_list.ensemble_fields["_times"][0])) * 0.04]
        self.MS_list.ensemble_fields["_IRF_tables"] = {}

        make_dir(self.MS_list.ensemble_fields["output_path"])

        global_states = self.MS_list.H.states
        global_logll = self.MS_list.H.loglikelihood
        global_accept = self.MS_list.H.accept
        shared_fields = self.MS_list.ensemble_fields
        unique_fields = self.MS_list.unique_fields
        self.RNG = np.random.default_rng(763940682935)
        RNG_state = self.RNG.bit_generator.state
        starting_iter = 0

        self.RNG.bit_generator.state = RNG_state
        ending_iter = min(starting_iter + self.MS_list.ensemble_fields["checkpoint_freq"], self.num_iters)
        need_initial_state = True
        while ending_iter <= self.num_iters:
            global_states, global_logll, global_accept, _, _ = main_metro_loop_serial(
                global_states, global_logll, global_accept,
                starting_iter, ending_iter, shared_fields, unique_fields,
                self.RNG, self.logger, need_initial_state=need_initial_state
            )
            if ending_iter == self.num_iters:
                break

            self.MS_list.latest_iter = ending_iter
            self.MS_list.H.states = global_states
            self.MS_list.H.loglikelihood = global_logll
            self.MS_list.H.accept = global_accept
            self.MS_list.random_state = self.RNG.bit_generator.state
            self.MS_list.checkpoint(os.path.join(self.MS_list.ensemble_fields["output_path"], self.export_path))

            need_initial_state = False
            starting_iter = ending_iter
            ending_iter = min(ending_iter + self.MS_list.ensemble_fields["checkpoint_freq"], self.num_iters)
        return

    def test_checkpoint(self):
        with open(os.path.join(self.MS_list.ensemble_fields["output_path"],
                               self.export_path), 'rb') as ifstream:
            self.ensemble_from_chpt = pickle.load(ifstream)
            self.RNG.bit_generator.state = self.ensemble_from_chpt.random_state
            starting_iter = self.ensemble_from_chpt.latest_iter

        global_states = self.ensemble_from_chpt.H.states
        global_logll = self.ensemble_from_chpt.H.loglikelihood
        global_accept = self.ensemble_from_chpt.H.accept
        shared_fields = self.ensemble_from_chpt.ensemble_fields
        unique_fields = self.ensemble_from_chpt.unique_fields
        need_initial_state = False
        ending_iter = min(starting_iter + self.ensemble_from_chpt.ensemble_fields["checkpoint_freq"], self.num_iters)

        while ending_iter <= self.num_iters:
            global_states, global_logll, global_accept, _, _ = main_metro_loop_serial(
                global_states, global_logll, global_accept,
                starting_iter, ending_iter, shared_fields, unique_fields,
                self.RNG, self.logger, need_initial_state=need_initial_state
            )
            if ending_iter == self.num_iters:
                break

            self.ensemble_from_chpt.latest_iter = ending_iter
            self.ensemble_from_chpt.H.states = global_states
            self.ensemble_from_chpt.H.loglikelihood = global_logll
            self.ensemble_from_chpt.H.accept = global_accept
            self.ensemble_from_chpt.random_state = self.RNG.bit_generator.state
            self.ensemble_from_chpt.checkpoint(os.path.join(self.MS_list.ensemble_fields["output_path"], self.export_path))

            need_initial_state = False
            starting_iter = ending_iter
            ending_iter = min(ending_iter + self.ensemble_from_chpt.ensemble_fields["checkpoint_freq"], self.num_iters)
        # Successful completion - checkpoints not needed anymore
        for chpt in os.listdir(self.MS_list.ensemble_fields["output_path"]):
            os.remove(os.path.join(self.MS_list.ensemble_fields["output_path"], chpt))
        os.rmdir(os.path.join(self.MS_list.ensemble_fields["output_path"]))

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
