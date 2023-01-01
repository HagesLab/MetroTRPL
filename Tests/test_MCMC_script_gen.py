import unittest
import numpy as np
import os

from bayes_io import generate_config_script_file, read_config_script_file
eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
q = 1.0 # [e]
q_C = 1.602e-19 # [C]
kB = 8.61773e-5  # [eV / K]

class TestUtils(unittest.TestCase):
    
    def setUp(self):
        Length  = [2000, 2000, 2000]                           # Length (nm)
        L   = 128                                # Spatial points
        measurement_types = ["TRPL", "TRPL", "TRPL"]
        self.simPar = [Length, L, measurement_types]

        param_names = ["n0", "p0", "mu_n", "mu_p", "ks", "Cn", "Cp",
                       "Sf", "Sb", "tauN", "tauP", "eps", "Tm", "m"]
        unit_conversions = {"n0":((1e-7) ** 3), "p0":((1e-7) ** 3), 
                            "mu_n":((1e7) ** 2) / (1e9), "mu_p":((1e7) ** 2) / (1e9), 
                            "ks":((1e7) ** 3) / (1e9), 
                            "Cn":((1e7) ** 6) / (1e9), "Cp":((1e7) ** 6) / (1e9),
                            "Sf":1e-2, "Sb":1e-2, "Tm":1}
        do_log = {"n0":1, "p0":1,"mu_n":1,"mu_p":1,"ks":1, "Cn":1, "Cp":1,
                  "Sf":1,"Sb":1,"tauN":1,"tauP":1,"eps":1,"Tm":1,
                  "m":1}

        initial_guesses = {"n0":1e8,
                            "p0": 1e14,
                            "mu_n": 20,
                            "mu_p": 20,
                            "ks": 5.958e-11,
                            "Cn": 1e-29,
                            "Cp": 1e-29,
                            "Sf":2.1e2,
                            "Sb": 2.665e2,
                            "tauN": 4.708e2,
                            "tauP": 1.961e2,
                            "eps":10,
                            "Tm":300,
                            "m":1}

        active_params = {"n0":0,
                         "p0":1,
                         "mu_n":0,
                         "mu_p":0,
                         "ks":1,
                         "Cn":1,
                         "Cp":1,
                         "Sf":1,
                         "Sb":1,
                         "tauN":1,
                         "tauP":1,
                         "eps":0,
                         "Tm":0,
                         "m":0}

        # Other options
        initial_variance = {"n0":1e-2,
                         "p0":1e-2,
                         "mu_n":1e-2,
                         "mu_p":1e-2,
                         "ks":1e-2,
                         "Cn":1e-2,
                         "Cp":1e-2,
                         "Sf":1e-2,
                         "Sb":1e-2,
                         "tauN":1e-2,
                         "tauP":1e-2,
                         "eps":1e-2,
                         "Tm":1e-2,
                         "m":1e-2}

        self.param_info = {"names":param_names,
                          "active":active_params,
                          "unit_conversions":unit_conversions,
                          "do_log":do_log,
                          "init_guess":initial_guesses,
                          "init_variance":initial_variance}

        self.measurement_fields = {"time_cutoff":[0,np.inf],
                                    "select_obs_sets": [0,1,2],
                                    "noise_level":None}

        output_path = os.path.join("MCMC", "Example_output_path")
        self.MCMC_fields = {"init_cond_path": os.path.join("MCMC", "Example_IC_path"),
                             "measurement_path": os.path.join("MCMC", "Example_meas_path"),
                             "output_path": output_path,
                             "num_iters": 25000,
                             "solver": "solveivp",
                             "rtol":1e-7,
                             "atol":1e-10,
                             "hmax":4,
                             "verify_hmax":0,
                             "anneal_params": [1/2500*100, 1e3, 1/2500*0.1], # [Unused, unused, initial_T]
                             "override_equal_mu":0,
                             "override_equal_s":0,
                             "log_pl":1,
                             "self_normalize":0,
                             "proposal_function":"box", # box or gauss; anything else disables new proposals
                             "one_param_at_a_time":0,
                             "checkpoint_dirname": os.path.join(output_path, "Checkpoints"),
                             "checkpoint_header": "CPU0",
                             "checkpoint_freq":12000, # Save a checkpoint every #this many iterations#
                             "load_checkpoint": None,
                             }
        return
    
    def test_create_script(self):
        where_script = os.path.join("Tests", "testfiles", "testmcmc.txt")
        generate_config_script_file(where_script, self.simPar, self.param_info, 
                                    self.measurement_fields, self.MCMC_fields, verbose=True)
        
        simPar, param_info, measurement_fields, MCMC_fields = read_config_script_file(where_script)
        
        for i in range(len(self.simPar)):
            np.testing.assert_equal(simPar[i], self.simPar[i])
            
        for k in self.param_info.keys():
            if isinstance(self.param_info[k], dict):
                for kk in self.param_info[k].keys():
                    self.assertEqual(param_info[k][kk], self.param_info[k][kk])
            
        for k in self.measurement_fields.keys():
            if isinstance(self.measurement_fields[k], (list, tuple, np.ndarray)):
                np.testing.assert_equal(measurement_fields[k], self.measurement_fields[k])
            else:
                self.assertEqual(measurement_fields[k], self.measurement_fields[k])
            
        for k in self.MCMC_fields.keys():
            if isinstance(self.MCMC_fields[k], (list, tuple, np.ndarray)):
                np.testing.assert_equal(MCMC_fields[k], self.MCMC_fields[k])
            else:
                self.assertEqual(MCMC_fields[k], self.MCMC_fields[k], msg=f"Read script failed: {k} mismatch")
        return
    
if __name__ == "__main__":
    t = TestUtils()
    t.setUp()
    t.test_create_script()