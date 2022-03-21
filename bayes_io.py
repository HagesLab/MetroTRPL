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
    # 1e-23 [cm^-2 s^-1] to [nm^-2 ns^-1]
    t = []
    PL = []
    uncertainty = []
    bval_cutoff = sys.float_info.min
    #print("cutoff", bval_cutoff)

    EARLY_CUT = ic_flags['time_cutoff']
    SELECT = ic_flags['select_obs_sets']
    NOISE_LEVEL = ic_flags['noise_level']

    LOG_PL = sim_flags['log_pl']
    NORMALIZE = sim_flags["self_normalize"]

    with open(exp_file, newline='') as file:
        eof = False
        next_t = []
        next_PL = []
        next_uncertainty = []
        ifstream = csv.reader(file)
        count = 0
        for row in ifstream:
            if row[0] == "END":
                eof = True
                finished = True
            else:
                finished = (float(row[0]) == 0 and len(next_t))

            if eof or finished:
                # t=0 means we finished reading the current PL curve - preprocess and package it
                #dataset_end_inds.append(dataset_end_inds[-1] + count)
                next_t = np.array(next_t)
                if NOISE_LEVEL is not None:
                    next_PL = (np.array(next_PL) + NOISE_LEVEL*np.random.normal(0, 1, len(next_PL))) * scale_f

                else:
                    next_PL = np.array(next_PL) * scale_f

                next_uncertainty = np.array(next_uncertainty) * scale_f

                if NORMALIZE:
                    next_PL /= max(next_PL)
                if verbose:
                    print("PL curve #{} finished reading".format(len(t)+1))
                    print("Number of points: {}".format(len(next_t)))
                    print("Times: {}".format(next_t))
                    print("PL values: {}".format(next_PL))
                if LOG_PL:
                    #print("Num exp points affected by cutoff", np.sum(next_PL < bval_cutoff))

                    # Deal with noisy negative values before taking log
                    #bval_cutoff = np.mean(uncertainty)
                    next_PL = np.abs(next_PL)
                    next_PL[next_PL < bval_cutoff] = bval_cutoff

                    next_uncertainty /= next_PL
                    next_uncertainty /= np.log(10) # Since we use log10 instead of ln
                    next_PL = np.log10(next_PL)

                t.append(next_t)
                PL.append(next_PL)
                uncertainty.append(next_uncertainty)

                next_t = []
                next_PL = []
                next_uncertainty = []

                count = 0

            if not eof:
                if (EARLY_CUT is not None and float(row[0]) > EARLY_CUT):
                    pass
                else: 
                    next_t.append(float(row[0]))
                    next_PL.append(float(row[1]))
                    next_uncertainty.append(float(row[2]))
            
            count += 1

    if SELECT is not None:
        t_s = []
        v_s = []
        u_s = []
        for i in range(len(t)):
            if i in SELECT:
                t_s.append(t[i])
                v_s.append(PL[i])
                u_s.append(uncertainty[i])
        return (t_s, v_s, u_s)
    else:
        return (t, PL, uncertainty)

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