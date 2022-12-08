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