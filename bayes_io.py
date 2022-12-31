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
        y = (np.array(y) + NOISE_LEVEL*np.random.normal(0, 1, len(y))) * scale_f

    else:
        y = np.array(y) * scale_f

    uncertainty = np.array(uncertainty) * scale_f
    
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
                        
    return grid, param_info, meas_flags


def generate_config_file(path, simPar, param_info, measurement_flags, verbose=False):
    if isinstance(simPar[0], (float, int)):
        simPar[0] = [simPar[0]]
    
    with open(path, "w+") as ofstream:
        ofstream.write("$$ MCMC CONFIG CREATED {} AT {}\n".format(datetime.datetime.now().date(),
                                                                  datetime.datetime.now().time()))
        ofstream.write("p$ Space Grid:\n")
        ofstream.write(f"Length(s): {simPar[0][0]}")
        for value in simPar[0][1:]:
            ofstream.write(f"\t{value}")
        ofstream.write('\n')
        ofstream.write(f"nx: {simPar[1]}\n")
        ofstream.write(f"Measurement type(s): {simPar[2][0]}")
        for value in simPar[2][1:]:
            ofstream.write(f"\t{value}")
        ofstream.write('\n')
        #######################################################################
        ofstream.write("p$ Param Info:\n")
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
        
    return
        
if __name__ == "__main__":
    grid, param_info, meas_flags = read_config_file("mcmc.txt")
    print(grid)
    print(param_info)
    print(meas_flags)