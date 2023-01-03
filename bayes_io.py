#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 18:10:38 2022

@author: cfai2304
"""
import sys
import csv
import os
import numpy as np
import datetime

# Eventually use io_utils for this
def get_split_and_clean_line(line: str):
    """Split line by colon symbol ':' and
    remove preceding and trailing spaces."""
    split_line = line.split(':')
    split_line = [i.strip() for i in split_line]
    return split_line

def extract_values(string, delimiter, dtype=float):
    """Converts a string with deliimiters into a list of [dtype] values"""
	# E.g. "100,200,300" with "," delimiter becomes [100,200,300] with dtype=float,
    # becomes ["100", "200", "300"] with dtype=str
    values = string.split(delimiter)
    values = np.array(values, dtype=dtype)
    return values

def get_data(exp_file, ic_flags, sim_flags, scale_f=1e-23, verbose=False):
    TIME_RANGE = ic_flags['time_cutoff']
    SELECT = ic_flags['select_obs_sets']
    NOISE_LEVEL = ic_flags['noise_level']

    LOG_PL = sim_flags['log_pl']
    NORMALIZE = sim_flags["self_normalize"]
        
    bval_cutoff = sys.float_info.min
    
    data = np.loadtxt(exp_file, delimiter=",")
    
    times = data[:,0]
    y = data[:,1]
    uncertainty = data[:,2]
    
    if NOISE_LEVEL is not None:
        y = (np.array(y) + NOISE_LEVEL*np.random.normal(0, 1, len(y)))
    else:
        y = np.array(y)

    uncertainty = np.array(uncertainty)
    
    t_list = []
    y_list = []
    u_list = []
    
    cutoff_indices = list(np.where(times == 0)[0]) + [None]
    for i in range(len(cutoff_indices) - 1):
        t_list.append(times[cutoff_indices[i]:cutoff_indices[i+1]])
        y_list.append(y[cutoff_indices[i]:cutoff_indices[i+1]])
        u_list.append(uncertainty[cutoff_indices[i]:cutoff_indices[i+1]])
        
    if TIME_RANGE is not None:
        t_low, t_high = TIME_RANGE[0], TIME_RANGE[1]
        for i in range(len(t_list)):
            keepL = np.searchsorted(t_list[i], t_low, side='left')
            keepR = np.searchsorted(t_list[i], t_high, side='right')
            t_list[i] = t_list[i][keepL:keepR]
            y_list[i] = y_list[i][keepL:keepR]
            u_list[i] = u_list[i][keepL:keepR]
            
    if NORMALIZE:
        for i in range(len(t_list)):
            y_list[i] /= np.nanmax(y_list[i])
            
    if isinstance(scale_f, (float, int)):
        scale_f = [scale_f] * len(t_list)
        
    for i in range(len(t_list)):
        y_list[i] *= scale_f[i]
        u_list[i] *= scale_f[i]
            
    if LOG_PL:
        # Deal with noisy negative values before taking log
        for i in range(len(t_list)):
            y_list[i] = np.abs(y_list[i])
            y_list[i][y_list[i] < bval_cutoff] = bval_cutoff
    
            u_list[i] /= y_list[i]
            u_list[i] /= np.log(10) # Since we use log10 instead of ln
            y_list[i] = np.log10(y_list[i])
    
    if SELECT is not None:
        t_s = []
        y_s = []
        u_s = []
        for i in range(len(t_list)):
            if i in SELECT:
                t_s.append(t_list[i])
                y_s.append(y_list[i])
                u_s.append(u_list[i])
        return (t_s, y_s, u_s)
    else:
        return (t_list, y_list, u_list)

def get_initpoints(init_file, ic_flags, scale_f=1e-21):
    SELECT = ic_flags['select_obs_sets']

    with open(init_file, newline='') as file:
        ifstream = csv.reader(file)
        initpoints = []
        for row in ifstream:
            if len(row) == 0: continue
            initpoints.append(row)
        
    if SELECT is not None:
        initpoints = np.array(initpoints)[SELECT]
    return np.array(initpoints, dtype=float) * scale_f

def put_into_param_info(param_info, vals, new_key):
    if "names" not in param_info:
        raise KeyError("Entry \"Param names\" not found in MCMC config file.\n"
                       "Check whether this entry is present and FIRST in\n"
                       "the Param Info subsection.")
    param_info[new_key] = {param_info["names"][i]:vals[i] 
                           for i in range(len(param_info["names"]))}
    return

def read_config_script_file(path):
    with open(path, 'r') as ifstream:
        grid = {}
        param_info = {}
        meas_flags = {}
        sim_flags = {}

        init_flag = 0
        
        if not ("$$ MCMC CONFIG CREATED") in next(ifstream):
            raise OSError("Error: this file is not a valid MCMC config file")
        
        # system_class = next(ifstream).strip('\n')
        # system_class = system_class[system_class.find(' ') + 1:]
        # if not system_class == self.module.system_ID:
        #     raise ValueError("Error: selected file is not a {}".format(self.module.system_ID))
                        
        # Extract parameters, ICs
        for line in ifstream:
            line = line.strip('\n')
            line_split = get_split_and_clean_line(line)

            if ("#" in line) or not line:
                continue

            # There are three "$" markers in an IC file: "Space Grid", "System Parameters" and "System Flags"
            # each corresponding to a different section of the file

            if "p$ Space Grid" in line:
                init_flag = 'g'
                continue
            
            if "p$ Param Info" in line:
                init_flag = 'p'
                
            if "p$ Measurement handling flags" in line:
                init_flag = 'm'
                
            if "p$ MCMC Control flags" in line:
                init_flag = 's'

            if len(line_split) > 1:

                if (init_flag == 'g'):
                    if line.startswith("Length(s)"):
                        grid["lengths"] = extract_values(line_split[1], delimiter='\t')
                        
                    elif line.startswith("nx"):
                        grid["nx"] = int(line_split[1])
                        
                    elif line.startswith("Measurement type(s)"):
                        grid["meas_types"] = line_split[1].split('\t')
                        
                    elif line.startswith("Number of measurements"):
                        grid["num_meas"] = int(line_split[1])
                        
                if (init_flag == 'p'):
                    if line.startswith("Param Names"):
                        param_info["names"] = line_split[1].split('\t')
                        
                    elif line.startswith("Unit conversions"):
                        vals = extract_values(line_split[1], delimiter='\t', dtype=float)
                        put_into_param_info(param_info, vals, "unit_conversions")
                        
                    elif line.startswith("Do logscale"):
                        vals = extract_values(line_split[1], delimiter='\t', dtype=int)
                        put_into_param_info(param_info, vals, "do_log")
                         
                    elif line.startswith("Active"):
                        vals = extract_values(line_split[1], delimiter='\t', dtype=int)
                        put_into_param_info(param_info, vals, "active")
                        
                    elif line.startswith("Initial guess"):
                        vals = extract_values(line_split[1], delimiter='\t', dtype=float)
                        put_into_param_info(param_info, vals, "init_guess")
                        
                    elif line.startswith("Initial variance"):
                        vals = extract_values(line_split[1], delimiter='\t', dtype=float)
                        put_into_param_info(param_info, vals, "init_variance")
                        
                if (init_flag == 'm'):
                    if line.startswith("Time cutoffs"):
                        vals = extract_values(line_split[1], delimiter='\t', dtype=float)
                        meas_flags["time_cutoff"] = vals
                        
                    elif line.startswith("Select measurement"):
                        if line_split[1] == "None":
                            meas_flags["select_obs_sets"] = None
                        else:
                            meas_flags["select_obs_sets"] = extract_values(line_split[1],
                                                                           delimiter='\t',
                                                                           dtype=int)
                        
                    elif line.startswith("Added noise level"):
                        if line_split[1] == "None":
                            meas_flags["noise_level"] = None
                        else:
                            meas_flags["noise_level"] = float(line_split[1])
                            
                if (init_flag == 's'):
                    if line.startswith("Num iters"):
                        sim_flags["num_iters"] = int(line_split[1])
                    elif line.startswith("Solver name"):
                        sim_flags["solver"] = line_split[1]
                    elif line.startswith("Solver rtol"):
                        sim_flags["rtol"] = float(line_split[1])
                    elif line.startswith("Solver atol"):
                        sim_flags["atol"] = float(line_split[1])
                    elif line.startswith("Solver hmax"):
                        sim_flags["hmax"] = float(line_split[1])
                    elif line.startswith("Repeat hmax"):
                        sim_flags["verify_hmax"] = int(line_split[1])
                    elif line.startswith("Anneal coefs"):
                        sim_flags["anneal_params"] = extract_values(line_split[1], '\t')
                    elif line.startswith("Force equal mu"):
                        sim_flags["override_equal_mu"] = int(line_split[1])
                    elif line.startswith("Force equal S"):
                        sim_flags["override_equal_s"] = int(line_split[1])
                    elif line.startswith("Use log of measurements"):
                        sim_flags["log_pl"] = int(line_split[1])
                    elif line.startswith("Normalize all meas and sims"):
                        sim_flags["self_normalize"] = int(line_split[1])
                    elif line.startswith("Proposal function"):
                        sim_flags["proposal_function"] = line_split[1]
                    elif line.startswith("Propose params one-at-a-time"):
                        sim_flags["one_param_at_a_time"] = int(line_split[1])
                    elif line.startswith("Checkpoint dir"):
                        sim_flags["checkpoint_dirname"] = os.path.join(line_split[1])
                    elif line.startswith("Checkpoint fileheader"):
                        sim_flags["checkpoint_header"] = line_split[1]
                    elif line.startswith("Checkpoint freq"):
                        sim_flags["checkpoint_freq"] = int(line_split[1])
                    elif line.startswith("Load checkpoint"):
                        if line_split[1] == "None":
                            sim_flags["load_checkpoint"] = None
                        else:
                            sim_flags["checkpoint_dirname"] = line_split[1]
                    elif line.startswith("Initial condition path"):
                        sim_flags["init_cond_path"] = os.path.join(line_split[1])
                    elif line.startswith("Measurement path"):
                        sim_flags["measurement_path"] = os.path.join(line_split[1])
                    elif line.startswith("Output path"):
                        sim_flags["output_path"] = os.path.join(line_split[1])
                        
    validate_grid(grid)
    validate_param_info(param_info)
    validate_meas_flags(meas_flags, grid["num_meas"])
              
    return grid, param_info, meas_flags, sim_flags

def generate_config_script_file(path, simPar, param_info, measurement_flags, sim_flags, verbose=False):
    validate_grid(simPar)
    validate_param_info(param_info)
    validate_meas_flags(measurement_flags, simPar["num_meas"])
    
    with open(path, "w+") as ofstream:
        ofstream.write("$$ MCMC CONFIG CREATED {} AT {}\n".format(datetime.datetime.now().date(),
                                                                  datetime.datetime.now().time()))
        ofstream.write("##\n")
        ofstream.write("p$ Space Grid:\n")
        if verbose: ofstream.write("# List of material/system thicknesses - one per measurement\n")
        lengths = simPar["lengths"]
        ofstream.write(f"Length(s): {lengths[0]}")
        for value in lengths[1:]:
            ofstream.write(f"\t{value}")
        ofstream.write('\n')
        if verbose: ofstream.write("# Number of space nodes used by solver discretization\n")
        nx = simPar["nx"]
        ofstream.write(f"nx: {nx}\n")
        if verbose: ofstream.write("# Model to use to simulate each measurement\n")
        meas_types = simPar["meas_types"]
        ofstream.write(f"Measurement type(s): {meas_types[0]}")
        for value in meas_types[1:]:
            ofstream.write(f"\t{value}")
        ofstream.write('\n')
        num_meas = simPar["num_meas"]
        ofstream.write(f"Number of measurements: {num_meas}\n")
        #######################################################################
        ofstream.write("##\n")
        ofstream.write("p$ Param Info:\n")
        if verbose: ofstream.write("# List of names of parameters used in the model\n")
        param_names = param_info["names"]
        ofstream.write(f"Param Names: {param_names[0]}")
        for value in param_names[1:]:
            ofstream.write(f"\t{value}")
        ofstream.write('\n')
        
        if verbose: ofstream.write("# Conversion from units params are entered in to units used by model\n")
        ucs = param_info["unit_conversions"]
        ofstream.write(f"Unit conversions: {ucs.get(param_names[0], 1)}")
        for name in param_names[1:]:
            ofstream.write(f"\t{ucs.get(name, 1)}")
        ofstream.write('\n')
        
        if verbose: ofstream.write("# Whether the MCMC should work with the log of each param. "
                                   "The answer should be YES for most models. \n")
        do_log = param_info["do_log"]
        ofstream.write(f"Do logscale: {do_log.get(param_names[0], 0)}")
        for name in param_names[1:]:
            ofstream.write(f"\t{do_log.get(name, 0)}")
        ofstream.write('\n')
        
        if verbose: ofstream.write("# Whether the MCMC should propose new moves for this parameter. "
                                   "Setting this to 0 or False fixes the parameter at its initial value.\n")
        active_params = param_info["active"]
        ofstream.write(f"Active: {active_params.get(param_names[0], 0)}")
        for name in param_names[1:]:
            ofstream.write(f"\t{active_params.get(name, 0)}")
        ofstream.write('\n')
        
        if verbose: ofstream.write("# Initial values for each parameter.\n")
        init_guess = param_info["init_guess"]
        ofstream.write(f"Initial guess: {init_guess.get(param_names[0], 0)}")
        for name in param_names[1:]:
            ofstream.write(f"\t{init_guess.get(name, 0)}")
        ofstream.write('\n')
        
        if verbose: ofstream.write("# Initial proposal variance for each parameter. "
                                   "I.e. how far from the current parameters new proposals will go.\n")
        init_variance = param_info["init_variance"]
        ofstream.write(f"Initial variance: {init_variance.get(param_names[0], 0)}")
        for name in param_names[1:]:
            ofstream.write(f"\t{init_variance.get(name, 0)}")
        ofstream.write('\n')
        #######################################################################
        ofstream.write("##\n")
        ofstream.write("p$ Measurement handling flags:\n")
        
        if verbose: ofstream.write("# Truncate measurements to only those within this time range. "
                                   "Inf values indicate an unbounded range. \n")
        tc = measurement_flags["time_cutoff"]
        ofstream.write(f"Time cutoffs: {tc[0]}\t{tc[1]}\n")
        
        if verbose: ofstream.write("# Which measurements in a sequence to keep for MCMC. "
                                   "A list such as [0,2] means to keep only the first and third "
                                   "measurements,\n# while omitting the second and others. None means "
                                   "ALL measurements are kept. \n")
        select = measurement_flags["select_obs_sets"]
        if select is None:
            ofstream.write(f"Select measurement: {select}\n")
        else:
            ofstream.write(f"Select measurement: {select[0]}")
            for s in select[1:]:
                ofstream.write(f"\t{s}")
            ofstream.write("\n")
            
        if verbose: ofstream.write("# Whether to add Gaussian noise of the indicated magnitude to "
                                   "the measurement.\n# This should be None (zero noise) unless testing "
                                   "with simulated data.\n")
        noise_level = measurement_flags["noise_level"]
        ofstream.write(f"Added noise level: {noise_level}\n")
        #######################################################################
        ofstream.write("##\n")
        ofstream.write("p$ MCMC Control flags:\n")
        if verbose: ofstream.write("# How many samples to propose.\n")
        num_iters = sim_flags["num_iters"]
        ofstream.write(f"Num iters: {num_iters}\n")
        if verbose: ofstream.write("# Which solver engine to use - solveivp (more robust) or odeint (sometimes faster).\n")
        solver = sim_flags["solver"]
        ofstream.write(f"Solver name: {solver}\n")
        if verbose: ofstream.write("# Solver engine relative tolerance.\n")
        rtol = sim_flags["rtol"]
        ofstream.write(f"Solver rtol: {rtol}\n")
        if verbose: ofstream.write("# Solver engine absolute tolerance.\n")
        atol = sim_flags["atol"]
        ofstream.write(f"Solver atol: {atol}\n")
        if verbose: ofstream.write("# Solver engine maximum adaptive time stepsize.\n")
        hmax = sim_flags["hmax"]
        ofstream.write(f"Solver hmax: {hmax}\n")
        if verbose: ofstream.write("# (Experimental) If 1 or True, MCMC will repeat simulations "
                                   "with progressively smaller hmax until the results converge.\n")
        verify_hmax = sim_flags["verify_hmax"]
        ofstream.write(f"Repeat hmax: {verify_hmax}\n")
        if verbose: ofstream.write("# Control coefficients for the likelihood sigma / annealing temperature.\n")
        anneal = sim_flags["anneal_params"]
        ofstream.write(f"Anneal coefs: {anneal[0]}\t{anneal[1]}\t{anneal[2]}\n")
        if verbose: ofstream.write("# Force parameters mu_n and mu_p to be equal.\n")
        emu = sim_flags["override_equal_mu"]
        ofstream.write(f"Force equal mu: {emu}\n")
        if verbose: ofstream.write("# Force parameters Sf and Sb to be equal.\n")
        es = sim_flags["override_equal_s"]
        ofstream.write(f"Force equal S: {es}\n")
        if verbose: ofstream.write("# Compare log of measurements and simulations for "
                                   "purpose of likelihood evaluation. Recommended to be 1 or True. \n")
        logpl = sim_flags["log_pl"]
        ofstream.write(f"Use log of measurements: {logpl}\n")
        if verbose: ofstream.write("# Normalize all individual measurements and simulations "
                                   "to maximum of 1 before likelihood evaluation. "
                                   "\n# A global scaling coefficient named 'm' may optionally be defined in param_info. "
                                   "\n# If the absolute units or efficiency of the measurement is unknown, "
                                   "\n# it is recommended to try fitting 'm' instead of relying on normalization. \n")
        norm = sim_flags["self_normalize"]
        ofstream.write(f"Normalize all meas and sims: {norm}\n")
        
        if verbose: ofstream.write("# Proposal function used to generate new states. "
                                   "Box for joint uniform box and Gauss for multivariate Gaussian. \n")
        prop_f = sim_flags["proposal_function"]
        ofstream.write(f"Proposal function: {prop_f}\n")
        
        if verbose: ofstream.write("# Whether a proposed move should change in one param or all params at once.\n")
        gibbs = sim_flags["one_param_at_a_time"]
        ofstream.write(f"Propose params one-at-a-time: {gibbs}\n")
        
        if verbose: ofstream.write("# Directory checkpoint files stored in.\n")
        chpt_d = sim_flags["checkpoint_dirname"]
        ofstream.write(f"Checkpoint dir: {chpt_d}\n")
        
        if verbose: ofstream.write("# An optional tag to append to the filename of each checkpoint.\n")
        chpt_h = sim_flags["checkpoint_header"]
        ofstream.write(f"Checkpoint fileheader: {chpt_h}\n")
        
        if verbose: ofstream.write("# Checkpoint saved every 'this many' samples.\n")
        chpt_f = sim_flags["checkpoint_freq"]
        ofstream.write(f"Checkpoint freq: {chpt_f}\n")
        
        if verbose: ofstream.write("# Name of a checkpoint file to resume an MCMC from.\n")
        load_chpt = sim_flags["load_checkpoint"]
        ofstream.write(f"Load checkpoint: {load_chpt}\n")
        
        if verbose: ofstream.write("# Path from which to read initial condition arrays. \n")
        ic = sim_flags["init_cond_path"]
        ofstream.write(f"Initial condition path: {ic}\n")
        
        if verbose: ofstream.write("# Path from which to read measurement data arrays. \n")
        mc = sim_flags["measurement_path"]
        ofstream.write(f"Measurement path: {mc}\n")
        
        if verbose: ofstream.write("# Path from which to save output MCMC objects. \n")
        oc = sim_flags["output_path"]
        ofstream.write(f"Output path: {oc}\n")
        
    return
        
def validate_grid(grid : dict, supported_meas_types=("TRPL", "TRTS")):
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
        all(map(lambda x:x > 0, grid["lengths"]))):
        pass
    else:
        raise ValueError("MCMC simPar entry 'Length' must be a list with "
                         "one positive length value per measurement")
        
    if not isinstance(grid['nx'], (int, np.integer)) or grid["nx"] <= 0:
        raise ValueError("MCMC simPar entry 'num_nodes' must be positive integer")
        
    if (isinstance(grid["meas_types"], (list, np.ndarray)) and 
        len(grid["meas_types"]) == declared_num_measurements and
        all(map(lambda x:x in supported_meas_types, grid["meas_types"]))):
        pass
    else:
        raise ValueError("MCMC simPar entry 'meas_types' must be a list with "
                         "one supported type per measurement.\n"
                         f"Supported types are {supported_meas_types}")

def validate_param_info(param_info : dict):
    if not isinstance(param_info, dict):
        raise TypeError("MCMC param_info must be type 'dict'")
        
    required_keys = ("names","active","unit_conversions","do_log",
                     "init_guess","init_variance")
    for k in required_keys:
        if k not in param_info:
            raise ValueError(f"MCMC param_info missing entry '{k}'")
        
    names = param_info["names"]
    if (isinstance(names, list) and len(names) > 0):
        pass
    else:
        raise ValueError("Invalid number of param names in param_info")
        
    # No duplicate names allowed
    if len(names) != len(set(names)):
        raise ValueError("Duplicate param names not allowed")
        
    # Alphanumeric + underscore only
    for k in names:
        if not k.replace("_", "").isalnum():
            raise ValueError(f"Param name {k} is invalid \n"
                             " Names must be alphanumeric")
    
    # Unit conversions CAN be missing entries - these are defaulted to 1
    for k, v in param_info["unit_conversions"].items():
        if not isinstance(v, (int,float)):
            raise ValueError(f"Invalid unit conversion {v} for param {k}")
        
    # Others must have ALL entries
    for k in names:
        if k not in param_info["do_log"]:
            raise KeyError(f"do_log missing param {k}")
        
        if not (isinstance(param_info["do_log"][k], (int, np.integer)) and 
                (param_info["do_log"][k] == 0 or param_info["do_log"][k] == 1)):
            raise ValueError(f"do_log param {k} invalid - must be 0 or 1")

        if k not in param_info["active"]:
            raise KeyError(f"param_info's 'active' missing param {k}")
        
        if not (isinstance(param_info["active"][k], (int, np.integer)) and 
                (param_info["active"][k] == 0 or param_info["active"][k] == 1)):
            raise ValueError(f"param_info's 'active' param {k} invalid - must be 0 or 1")
            
        if k not in param_info["init_guess"]:
            raise KeyError(f"init_guess missing param {k}")
            
        if not (isinstance(param_info["init_guess"][k], (int, np.integer, float))):
            raise ValueError(f"init_variance param {k} invalid")

        if k not in param_info["init_variance"]:
            raise KeyError(f"init_variance missing param {k}")
        
        if not (isinstance(param_info["init_variance"][k], (int, np.int32, float)) and 
                param_info["init_variance"][k] >= 0):
            raise ValueError(f"init_variance param {k} invalid - must be non-negative")
            
    return

def validate_meas_flags(meas_flags : dict, num_measurements):
    if not isinstance(meas_flags, dict):
        raise TypeError("MCMC meas_flags must be type 'dict'")
        
    required_keys = ("time_cutoff","select_obs_sets","noise_level")
    for k in required_keys:
        if k not in meas_flags:
            raise ValueError(f"MCMC meas_flags missing entry '{k}'")
            
    time_cutoff = meas_flags["time_cutoff"]
    if isinstance(time_cutoff, (list, np.ndarray)) and len(time_cutoff) == 2:
        pass
    else:
        raise ValueError("meas_flags time_cutoff must be list with 2 cutoff values \n"
                         "E.g. [0, np.inf] to allow all non-negative times.")
        
    if not isinstance(time_cutoff[0], (int, np.integer, float)):
        raise ValueError("Invalid time_cutoff lower bound")
        
    if not isinstance(time_cutoff[1], (int, np.integer, float)):
        raise ValueError("Invalid time_cutoff upper bound")
        
    if time_cutoff[1] < time_cutoff[0]:
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
            
    noise = meas_flags["noise_level"]
    if noise is None or (isinstance(noise, (int, np.integer, float)) and noise >= 0):
        pass
    else:
        raise TypeError("Noise must be numeric and postiive")
        
    return


if __name__ == "__main__":
    grid, param_info, meas_flags, sim_flags = read_config_script_file("mcmc0.txt")
    print(grid)
    print(param_info)
    print(meas_flags)
    print(sim_flags)