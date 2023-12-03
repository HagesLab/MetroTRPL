import numpy as np
from forward_solver import MODELS # To verify that the chosen model exists

MODELS["pa"] = lambda x: x

def check_valid_filename(file_name):
    """Screens file_name for prohibited characters
        This one allows slashes
    """
    prohibited_characters = ["<", ">", "*", "?", ":", "\"", "|"]
    # return !any(char in file_name for char in prohibited_characters)
    if any(char in file_name for char in prohibited_characters):
        return False

    return True

def check_fittable_fluence(ff : None | tuple | list) -> bool:
    """Validates fittable_fluence entry in meas_flags"""
    if ff is None:
        pass
    elif isinstance(ff, (list, tuple)):
        if len(ff) < 3 or len(ff) > 4:
            return False
        if not isinstance(ff[0], (float, int)):
            return False
        if not isinstance(ff[1], (list, tuple, np.ndarray)):
            return False
        if ff[2] is not None and not isinstance(ff[2], (list, tuple)):
            return False

        if len(ff[1]) == 0:
            return False

        for i_fittable in ff[1]:
            if not isinstance(i_fittable, (int, np.integer)) or i_fittable < 0:
                return False

        if ff[2] is not None:
            for constraint_grp in ff[2]:
                if not isinstance(constraint_grp, (list, tuple)):
                    return False
                for c in constraint_grp:
                    if not isinstance(c, (int, np.integer)) or c < 0:
                        return False

        if len(ff) == 4:
            if not isinstance(ff[3], (list, tuple, np.ndarray)):
                return False
            if len(ff[3]) == 0:
                return False
            
            for i_guess in ff[3]:
                if not isinstance(i_guess, (int, np.integer, float)) or i_guess < 0:
                    return False
    else:
        return False

    return True

def validate_grid(grid: dict, supported_meas_types=("TRPL", "TRTS", "pa")):
    if not isinstance(grid, dict):
        raise TypeError("MCMC simPar must be type 'dict'")

    required_keys = ("lengths", "nx", "meas_types", "num_meas")
    for k in required_keys:
        if k not in grid:
            raise ValueError(f"MCMC simPar missing entry '{k}'")

    declared_num_measurements = grid["num_meas"]
    if (isinstance(declared_num_measurements, int) and
            declared_num_measurements > 0):
        pass
    else:
        raise ValueError("Invalid number of measurements")

    if (isinstance(grid["lengths"], (list, np.ndarray)) and
        len(grid["lengths"]) == declared_num_measurements and
            all(map(lambda x: x > 0, grid["lengths"]))):
        pass
    else:
        raise ValueError("MCMC simPar entry 'Length' must be a list with "
                         "one positive length value per measurement")

    if (isinstance(grid["nx"], (list, np.ndarray)) and
        len(grid["nx"]) == declared_num_measurements and
            all(map(lambda x: x > 0, grid["nx"]))):
        pass
    else:
        raise ValueError(
            "MCMC simPar entry 'nx' must be a list with one positive integer "
            "number of nodes per measurement")

    if (isinstance(grid["meas_types"], (list, np.ndarray)) and
        len(grid["meas_types"]) == declared_num_measurements and
            all(map(lambda x: x in supported_meas_types, grid["meas_types"]))):
        pass
    else:
        raise ValueError("MCMC simPar entry 'meas_types' must be a list with "
                         "one supported type per measurement.\n"
                         f"Supported types are {supported_meas_types}")

def validate_param_info(param_info: dict):
    if not isinstance(param_info, dict):
        raise TypeError("MCMC param_info must be type 'dict'")

    required_keys = ("names", "active", "unit_conversions", "do_log",
                     "init_guess", "init_variance", "prior_dist")
    for k in required_keys:
        if k not in param_info:
            raise ValueError(f"MCMC param_info missing entry '{k}'")

    names = param_info["names"]
    if (isinstance(names, list) and len(names) > 0):
        pass
    else:
        raise ValueError("Invalid number of param names in param_info")

    # No duplicate names allowed
    if len(names) == len(set(names)):
        pass
    else:
        raise ValueError("Duplicate param names not allowed")

    # Alphanumeric + underscore only
    for k in names:
        if k.replace("_", "").isalnum():
            pass
        else:
            raise ValueError(f"Param name {k} is invalid \n"
                             " Names must be alphanumeric")
        
    # Disallow names starting with _, which is reserved for scale_factors, fittable fluences, etc
    # Alphanumeric + underscore only
    for k in names:
        if k.startswith("_"):
            raise ValueError(f"Param name {k} is invalid \n"
                             " Names must not start with _")

    # Unit conversions CAN be missing entries - these are defaulted to 1
    for k, v in param_info["unit_conversions"].items():
        if isinstance(v, (int, float)):
            pass
        else:
            raise ValueError(f"Invalid unit conversion {v} for param {k}")

    # Mu constraint
    if "do_mu_constraint" in param_info:
        mu = param_info["do_mu_constraint"]
        if isinstance(mu, (list, tuple, np.ndarray)) and len(mu) == 2:
            pass
        else:
            raise ValueError("mu_constraint must be list with center and width values \n"
                             "E.g. [100, 10] to restrict ambipolar mu between 90 and 110.")

    # Others must have ALL entries
    for k in names:
        if k in param_info["do_log"]:
            pass
        else:
            raise KeyError(f"do_log missing param {k}")

        if (isinstance(param_info["do_log"][k], (int, np.integer)) and
                (param_info["do_log"][k] == 0 or param_info["do_log"][k] == 1)):
            pass
        else:
            raise ValueError(f"do_log param {k} invalid - must be 0 or 1")

        if k in param_info["active"]:
            pass
        else:
            raise KeyError(f"param_info's 'active' missing param {k}")

        if (isinstance(param_info["active"][k], (int, np.integer)) and
                (param_info["active"][k] == 0 or param_info["active"][k] == 1)):
            pass
        else:
            raise ValueError(
                f"param_info's 'active' param {k} invalid - must be 0 or 1")

        if k in param_info["init_guess"]:
            pass
        else:
            raise KeyError(f"init_guess missing param {k}")

        if (isinstance(param_info["init_guess"][k], (int, np.integer, float))):
            pass
        else:
            raise ValueError(f"init_guess param {k} invalid")

        if k in param_info["prior_dist"]:
            pass
        else:
            raise KeyError(f"prior_dist missing param {k}")

        if (isinstance(param_info["prior_dist"][k], (tuple, list))):
            pass
        else:
            raise ValueError(f"prior_dist param {k} must be tuple or list")

        if (len(param_info["prior_dist"][k]) == 2):
            pass
        else:
            raise ValueError(f"prior_dist param {k} must be length 2")

        if (isinstance(param_info["prior_dist"][k][0], (int, np.integer, float)) and
                isinstance(param_info["prior_dist"][k][1], (int, np.integer, float))):
            pass
        else:
            raise ValueError(
                f"prior_dist param {k} must contain two numeric bounds")

        if (param_info["prior_dist"][k][0] < param_info["prior_dist"][k][1]):
            pass
        else:
            raise ValueError(f"prior_dist param {k} lower bound must be smaller"
                             " than upper bound")

        if k in param_info["init_variance"]:
            pass
        else:
            raise KeyError(f"init_variance missing param {k}")

        if (isinstance(param_info["init_variance"][k], (int, np.integer, float))
                and param_info["init_variance"][k] >= 0):
            pass
        else:
            raise ValueError(
                f"init_variance param {k} invalid - must be non-negative")

    return

def validate_meas_flags(meas_flags: dict, num_measurements):
    if not isinstance(meas_flags, dict):
        raise TypeError("MCMC meas_flags must be type 'dict'")

    required_keys = ("time_cutoff", "select_obs_sets")
    for k in required_keys:
        if k not in meas_flags:
            raise ValueError(f"MCMC meas_flags missing entry '{k}'")

    time_cutoff = meas_flags["time_cutoff"]
    if isinstance(time_cutoff, (list, np.ndarray)) and len(time_cutoff) == 2:
        pass
    else:
        raise ValueError("meas_flags time_cutoff must be list with 2 cutoff values \n"
                         "E.g. [0, np.inf] to allow all non-negative times.")

    if isinstance(time_cutoff[0], (int, np.integer, float)):
        pass
    else:
        raise ValueError("Invalid time_cutoff lower bound")

    if isinstance(time_cutoff[1], (int, np.integer, float)):
        pass
    else:
        raise ValueError("Invalid time_cutoff upper bound")

    if time_cutoff[1] >= time_cutoff[0]:
        pass
    else:
        raise ValueError("time_cutoff upper bound smaller than lower bound")

    select = meas_flags["select_obs_sets"]
    if select is None or isinstance(select, (list, np.ndarray)):
        pass
    else:
        raise TypeError("select_obs_sets must be None or a list type")

    if isinstance(select, (list, np.ndarray)):
        if all(map(lambda x: x >= 0 and x < num_measurements, select)):
            pass
        else:
            raise ValueError("Invalid select value - must be ints between 0 and"
                             " num_measurements - 1")
    if "noise_level" in meas_flags:
        noise = meas_flags["noise_level"]
        if noise is None or (isinstance(noise, (int, np.integer, float)) and noise >= 0):
            pass
        else:
            raise TypeError("Noise must be numeric and postiive")

    if "resample" in meas_flags:
        resample = meas_flags["resample"]
        if isinstance(resample, int):
            pass
        else:
            raise TypeError("Resample must be an integer")
        if resample >= 1:
            pass
        else:
            raise ValueError("Invalid resample - must be positive")
    return

def validate_MCMC_fields(MCMC_fields: dict, num_measurements: int,
                         supported_solvers=("odeint", "solveivp", "NN", "diagnostic"),
                         supported_prop_funcs=("box", "gauss", "None")):
    if not isinstance(MCMC_fields, dict):
        raise TypeError("MCMC control flags must be type 'dict'")

    required_keys = ("init_cond_path", "measurement_path", "output_path",
                     "num_iters", "solver", "model",
                     "likel2variance_ratio",
                     "log_pl", "self_normalize",
                     "proposal_function",
                     "checkpoint_freq",
                     "load_checkpoint",
                     )
    for k in required_keys:
        if k not in MCMC_fields:
            raise ValueError(f"MCMC control flags missing entry '{k}'")

    if isinstance(MCMC_fields["init_cond_path"], str):
        pass
    else:
        raise ValueError("init_cond_path must be a valid path")

    if isinstance(MCMC_fields["measurement_path"], str):
        pass
    else:
        raise ValueError("measurement_path must be a valid path")

    if isinstance(MCMC_fields["output_path"], str):
        pass
    else:
        raise ValueError("output_path must be a valid path")

    if check_valid_filename(MCMC_fields["output_path"]):
        pass
    else:
        raise ValueError("Invalid char in output_path")

    num_iters = MCMC_fields["num_iters"]
    if isinstance(num_iters, (int, np.integer)) and num_iters > 0:
        pass
    else:
        raise ValueError("Invalid number of iterations")
    
    if "starting_iter" in MCMC_fields:
        starting_iter = MCMC_fields["starting_iter"]
        if isinstance(starting_iter, (int, np.integer)) and starting_iter >= 0:
            pass
        else:
            raise ValueError("Invalid starting iteration")
    
    if isinstance(MCMC_fields["model"], str) and MCMC_fields["model"] in MODELS:
        pass
    else:
        raise ValueError(f"MCMC control 'model' must be one of the following solvers: {list(MODELS.keys())}")
    
    if isinstance(MCMC_fields["solver"], tuple):
        pass
    else:
        raise ValueError("MCMC control 'solver' must be a tuple with at least"
                         f" one element - one solver name from {supported_solvers}")

    if MCMC_fields["solver"][0] in supported_solvers:
        pass
    else:
        raise ValueError("MCMC control 'solver' must be a supported solver.\n"
                         f"Supported solvers are {supported_solvers}")

    if "rtol" in MCMC_fields:
        rtol = MCMC_fields["rtol"]
        if isinstance(rtol, (int, np.integer, float)) and rtol > 0:
            pass
        else:
            raise ValueError("rtol must be a non-negative value")

    if "atol" in MCMC_fields:
        atol = MCMC_fields["atol"]
        if isinstance(atol, (int, np.integer, float)) and atol > 0:
            pass
        else:
            raise ValueError("atol must be a non-negative value")

    if "hmax" in MCMC_fields:
        hmax = MCMC_fields["hmax"]
        if isinstance(hmax, (int, np.integer, float)) and hmax > 0:
            pass
        else:
            raise ValueError("hmax must be a non-negative value")

    l2v = MCMC_fields["likel2variance_ratio"]

    if isinstance(l2v, (int, np.integer, float)):
        if l2v < 0:
            raise ValueError("Likelihood-to-variance must be non-negative value")
    elif isinstance(l2v, dict):
        for meas_type, val in l2v.items():
            if isinstance(meas_type, str) and isinstance(val, (int, np.integer, float)) and val >= 0:
                pass
            else:
                raise ValueError(f"{meas_type}: Likelihood-to-variance must have one non-negative value"
                                 " per measurement type")
    else:
        raise ValueError("Invalid likelihood-to-variance")
        

    if "override_equal_mu" in MCMC_fields:
        mu = MCMC_fields["override_equal_mu"]
        if (isinstance(mu, (int, np.integer)) and
                (mu == 0 or mu == 1)):
            pass
        else:
            raise ValueError("override equal_mu invalid - must be 0 or 1")

    if "override_equal_s" in MCMC_fields:
        s = MCMC_fields["override_equal_s"]
        if (isinstance(s, (int, np.integer)) and
                (s == 0 or s == 1)):
            pass
        else:
            raise ValueError("override equal_s invalid - must be 0 or 1")

    logpl = MCMC_fields["log_pl"]
    if (isinstance(logpl, (int, np.integer)) and
            (logpl == 0 or logpl == 1)):
        pass
    else:
        raise ValueError("logpl invalid - must be 0 or 1")
    
    if "scale_factor" in MCMC_fields:
        scale_f = MCMC_fields["scale_factor"]
        success = check_fittable_fluence(scale_f)
        if not success:
            raise ValueError("Invalid scale_factor - must be None, or tuple"
                             "(see printed description when verbose=True)")

    if "fittable_fluences" in MCMC_fields:
        ff = MCMC_fields["fittable_fluences"]
        success = check_fittable_fluence(ff)
        if not success:
            raise ValueError("Invalid fittable_fluences - must be None, or tuple"
                             "(see printed description when verbose=True)")
        
    if "fittable_absps" in MCMC_fields:
        ff = MCMC_fields["fittable_absps"]
        success = check_fittable_fluence(ff)
        if not success:
            raise ValueError("Invalid fittable_absps - must be None, or tuple"
                             "(see printed description when verbose=True)")

    norm = MCMC_fields["self_normalize"]
    if norm is None:
        pass
    elif (isinstance(norm, list)) and all(map(lambda x: isinstance(x, str), norm)):
        pass
    else:
        raise ValueError("self_normalize invalid - must be None, or a list of measurement types "
                         "that should be normalized.")

    if MCMC_fields["proposal_function"] in supported_prop_funcs:
        pass
    else:
        raise ValueError("MCMC control 'proposal_function' must be a supported proposal function.\n"
                         f"Supported funcs are {supported_prop_funcs}")

    if "hard_bounds" in MCMC_fields:
        bound = MCMC_fields["hard_bounds"]
        if (isinstance(bound, (int, np.integer)) and
                (bound == 0 or bound == 1)):
            pass
        else:
            raise ValueError("hard_bounds invalid - must be 0 or 1")
        
    if "force_min_y" in MCMC_fields:
        fy = MCMC_fields["force_min_y"]
        if (isinstance(fy, (int, np.integer)) and
                (fy == 0 or fy == 1)):
            pass
        else:
            raise ValueError("force_min_y invalid - must be 0 or 1")

    if "irf_convolution" in MCMC_fields:
        irf = MCMC_fields["irf_convolution"]
        if irf is None:
            pass
        elif (isinstance(irf, (list, np.ndarray)) and
              len(irf) == num_measurements and
                all(map(lambda x: x >= 0, irf))):
            pass
        else:
            raise ValueError("MCMC control 'irf_convolution' must be None, or a list with "
                             "one positive wavelength value per measurement")
        
    if "parallel_tempering" in MCMC_fields:
        pa = MCMC_fields["parallel_tempering"]
        if pa is None:
            pass
        elif (isinstance(pa, (list, np.ndarray)) and
              len(pa) > 0 and
                all(map(lambda x: x > 0, pa))):
            pass
        else:
            raise ValueError("MCMC control 'parallel_tempering' must be None, or a list with "
                             "at least one positive temperature value")
        
    if "temper_freq" in MCMC_fields:
        tf = MCMC_fields["temper_freq"]
        if isinstance(tf, (int, np.integer)) and tf > 0:
            pass
        else:
            raise ValueError("temper_freq must be positive integer")

    if "checkpoint_dirname" in MCMC_fields:
        chpt_d = MCMC_fields["checkpoint_dirname"]
        if check_valid_filename(chpt_d):
            pass
        else:
            raise ValueError("Invalid char in checkpoint dirname")

    if "checkpoint_header" in MCMC_fields:
        chpt_h = MCMC_fields["checkpoint_header"]
        if check_valid_filename(chpt_h):
            pass
        else:
            raise ValueError("Invalid char in checkpoint header")

    chpt_f = MCMC_fields["checkpoint_freq"]
    if isinstance(chpt_f, (int, np.integer)) and chpt_f > 0:
        pass
    else:
        raise ValueError("checkpoint_freq must be positive integer")

    load = MCMC_fields["load_checkpoint"]
    if load is None or isinstance(load, str):
        pass
    else:
        raise ValueError("Invalid name of checkpoint to load")
    return
