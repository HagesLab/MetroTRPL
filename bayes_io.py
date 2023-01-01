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

def read_config_file(path):
    with open(path, 'r') as ifstream:
        grid = [-1, -1, -1]
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
                        grid[0] = extract_values(line_split[1], delimiter='\t')
                        
                    elif line.startswith("nx"):
                        grid[1] = int(line_split[1])
                        
                    elif line.startswith("Measurement type(s)"):
                        grid[2] = extract_values(line_split[1], delimiter='\t', dtype=str)
                        
                if (init_flag == 'p'):
                    if line.startswith("Param Names"):
                        param_info["names"] = extract_values(line_split[1], delimiter='\t', dtype=str)
                        
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
                        sim_flags["verify_hmax"] = bool(line_split[1])
                    elif line.startswith("Anneal coefs"):
                        sim_flags["anneal_params"] = extract_values(line_split[1], '\t')
                    elif line.startswith("Force equal mu"):
                        sim_flags["override_equal_mu"] = bool(line_split[1])
                    elif line.startswith("Force equal S"):
                        sim_flags["override_equal_s"] = bool(line_split[1])
                    elif line.startswith("Use log of measurements"):
                        sim_flags["log_pl"] = bool(line_split[1])
                    elif line.startswith("Normalize all meas and sims"):
                        sim_flags["self_normalize"] = bool(line_split[1])
                    elif line.startswith("Proposal function"):
                        sim_flags["proposal_function"] = line_split[1]
                    elif line.startswith("Propose params one-at-a-time"):
                        sim_flags["one_param_at_a_time"] = bool(line_split[1])
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
                        
    return grid, param_info, meas_flags, sim_flags


def generate_config_file(path, simPar, param_info, measurement_flags, sim_flags, verbose=False):
    if isinstance(simPar[0], (float, int)):
        simPar[0] = [simPar[0]]
    
    with open(path, "w+") as ofstream:
        ofstream.write("$$ MCMC CONFIG CREATED {} AT {}\n".format(datetime.datetime.now().date(),
                                                                  datetime.datetime.now().time()))
        ofstream.write("##\n")
        ofstream.write("p$ Space Grid:\n")
        if verbose: ofstream.write("# List of material/system thicknesses - one per measurement\n")
        ofstream.write(f"Length(s): {simPar[0][0]}")
        for value in simPar[0][1:]:
            ofstream.write(f"\t{value}")
        ofstream.write('\n')
        if verbose: ofstream.write("# Number of space nodes used by solver discretization\n")
        ofstream.write(f"nx: {simPar[1]}\n")
        if verbose: ofstream.write("# Model to use to simulate each measurement\n")
        ofstream.write(f"Measurement type(s): {simPar[2][0]}")
        for value in simPar[2][1:]:
            ofstream.write(f"\t{value}")
        ofstream.write('\n')
        #######################################################################
        ofstream.write("##\n")
        ofstream.write("p$ Param Info:\n")
        if verbose: ofstream.write("# List of names of parameters used in the model\n")
        param_names = param_info["names"]
        ofstream.write(f"Param Names: {param_names[0]}")
        for value in param_names[1:]:
            ofstream.write(f"\t{value}")
        ofstream.write('\n')
        
        ucs = param_info["unit_conversions"]
        ofstream.write(f"Unit conversions: {ucs.get(param_names[0], 1)}")
        for name in param_names[1:]:
            ofstream.write(f"\t{ucs.get(name, 1)}")
        ofstream.write('\n')
        
        do_log = param_info["do_log"]
        ofstream.write(f"Do logscale: {do_log.get(param_names[0], 0)}")
        for name in param_names[1:]:
            ofstream.write(f"\t{do_log.get(name, 0)}")
        ofstream.write('\n')
        
        active_params = param_info["active"]
        ofstream.write(f"Active: {active_params.get(param_names[0], 0)}")
        for name in param_names[1:]:
            ofstream.write(f"\t{active_params.get(name, 0)}")
        ofstream.write('\n')
        
        init_guess = param_info["init_guess"]
        ofstream.write(f"Initial guess: {init_guess.get(param_names[0], 0)}")
        for name in param_names[1:]:
            ofstream.write(f"\t{init_guess.get(name, 0)}")
        ofstream.write('\n')
        
        init_variance = param_info["init_variance"]
        ofstream.write(f"Initial variance: {init_variance.get(param_names[0], 0)}")
        for name in param_names[1:]:
            ofstream.write(f"\t{init_variance.get(name, 0)}")
        ofstream.write('\n')
        #######################################################################
        ofstream.write("##\n")
        ofstream.write("p$ Measurement handling flags:\n")
        tc = measurement_flags["time_cutoff"]
        ofstream.write(f"Time cutoffs: {tc[0]}\t{tc[1]}\n")
        select = measurement_flags["select_obs_sets"]
        if select is None:
            ofstream.write(f"Select measurement: {select}\n")
        else:
            ofstream.write(f"Select measurement: {select[0]}")
            for s in select[1:]:
                ofstream.write(f"\t{s}")
            ofstream.write("\n")
        noise_level = measurement_flags["noise_level"]
        ofstream.write(f"Added noise level: {noise_level}\n")
        #######################################################################
        ofstream.write("##\n")
        ofstream.write("p$ MCMC Control flags:\n")
        num_iters = sim_flags["num_iters"]
        ofstream.write(f"Num iters: {num_iters}\n")
        solver = sim_flags["solver"]
        ofstream.write(f"Solver name: {solver}\n")
        rtol = sim_flags["rtol"]
        ofstream.write(f"Solver rtol: {rtol}\n")
        atol = sim_flags["atol"]
        ofstream.write(f"Solver atol: {atol}\n")
        hmax = sim_flags["hmax"]
        ofstream.write(f"Solver hmax: {hmax}\n")
        verify_hmax = sim_flags["verify_hmax"]
        ofstream.write(f"Repeat hmax: {verify_hmax}\n")
        anneal = sim_flags["anneal_params"]
        ofstream.write(f"Anneal coefs: {anneal[0]}\t{anneal[1]}\t{anneal[2]}\n")
        emu = sim_flags["override_equal_mu"]
        ofstream.write(f"Force equal mu: {emu}\n")
        es = sim_flags["override_equal_s"]
        ofstream.write(f"Force equal S: {es}\n")
        logpl = sim_flags["log_pl"]
        ofstream.write(f"Use log of measurements: {logpl}\n")
        norm = sim_flags["self_normalize"]
        ofstream.write(f"Normalize all meas and sims: {norm}\n")
        prop_f = sim_flags["proposal_function"]
        ofstream.write(f"Proposal function: {prop_f}\n")
        gibbs = sim_flags["one_param_at_a_time"]
        ofstream.write(f"Propose params one-at-a-time: {gibbs}\n")
        chpt_d = sim_flags["checkpoint_dirname"]
        ofstream.write(f"Checkpoint dir: {chpt_d}\n")
        chpt_h = sim_flags["checkpoint_header"]
        ofstream.write(f"Checkpoint fileheader: {chpt_h}\n")
        chpt_f = sim_flags["checkpoint_freq"]
        ofstream.write(f"Checkpoint freq: {chpt_f}\n")
        load_chpt = sim_flags["load_checkpoint"]
        ofstream.write(f"Load checkpoint: {load_chpt}\n")
        
    return
        
if __name__ == "__main__":
    grid, param_info, meas_flags, sim_flags = read_config_file("mcmc.txt")
    print(grid)
    print(param_info)
    print(meas_flags)
    print(sim_flags)
    print(os.listdir(sim_flags["checkpoint_dirname"]))