# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 10:13:16 2022

@author: cfai2

Use this script to reconstruct output trajectories from the log file
"""

import numpy as np
import os
from sim_utils import MetroState

wdir = r"C:\Users\cfai2\Documents\src\Metro\trts_outputs\sim_everything_norm"
fname = None
i = 3
out_fname = f"CPU{i}-final.pik"

for file in os.listdir(wdir):
    if file.startswith(f"CPU{i}") and file.endswith("log"):
        fname = file

if fname is None:
    raise FileNotFoundError()


def interpret_line(line):
    """ Identify the parameter and mean value """
    param = line[line.find("Next ") + len("Next "):line.rfind(":")]
    mean_val = float(line[line.rfind("mean ") + len("mean "):])
    return param, mean_val


def parse_dict(line):
    dict_ = {}
    while len(line) > 0:
        next_key_l = line.find("'")
        next_key_r = line[next_key_l+1:].find("'") + next_key_l
        next_key = line[next_key_l+1:next_key_r+1]

        line = line[next_key_r+1+1:]

        next_val_l = line.find(":")+1
        next_next_val_l = line[next_val_l:].find(":")
        if next_next_val_l == -1:
            next_next_val_l = len(line) + 1

        next_val_r = line[next_val_l:next_next_val_l-1].rfind("'")

        if next_val_r == -1:
            next_val_r = len(line)
        next_val = line[next_val_l+1:next_val_r-1]
        line = line[next_val_r:]

        dict_[next_key] = next_val

    return dict_


def retype(entry):
    if "array" in entry:
        entry = entry[entry.find("[")+1:entry.find("]")]
        entry = entry.strip(" ")
        entry = np.array(entry.split(","), dtype=float)

    elif "[" in entry and "]" in entry:
        entry = entry[entry.find("[")+1:entry.find("]")]
        entry = entry.strip("\'").split("\', \'")

    elif "(" in entry and ")" in entry:
        entry = entry[entry.find("(")+1:entry.find(")")]
        entry = entry.strip("\'").split(", ")

    elif entry == "None":
        entry = None

    else:
        try:
            entry = float(entry)
            if entry.is_integer():
                entry = int(entry)
        except Exception:
            entry = entry.strip("\'")

    return entry


means = {}
count = 0
meancount = 0
iter_ = 0
accept = []
accepted = 1
logl = []
pll = []

with open(os.path.join(wdir, fname), 'r') as ifstream:
    for line in ifstream:
        if "Measurement handling fields" in line:
            m_fields = parse_dict(line.strip(" \n"))
            for m in m_fields:
                m_fields[m] = retype(m_fields[m])

        elif "Sim info" in line:
            sim_info = parse_dict(line.strip(" \n"))
            for s in sim_info:
                sim_info[s] = retype(sim_info[s])

        elif "Param infos" in line:
            param_infos = parse_dict(line.strip(" \n"))
            param_infos['names'] = retype(param_infos['names'])
            param_infos['active'] = {
                param: 1 for param in param_infos['names']}
            param_infos['init_guess'] = {
                param: 1 for param in param_infos['names']}
            param_infos['init_variance'] = {
                param: 1 for param in param_infos['names']}
            param_infos['do_log'] = {
                param: 1 for param in param_infos['names']}
            param_infos['unit_conversions'] = {
                param: 1 for param in param_infos['names']}

        elif "MCMC fields" in line:
            MCMC_fields = parse_dict(line.strip(" \n"))
            for m in MCMC_fields:
                MCMC_fields[m] = retype(MCMC_fields[m])

        if "mean" in line:
            param, mean_val = interpret_line(line)

            if param not in means:
                means[param] = []

            means[param].append(mean_val)
            meancount += 1

        if "Iter " in line:
            accept.append(accepted)
            accepted = 1

        if "Rejected!" in line:
            accepted = 0

        if "loglikelihood" in line:
            ll = line[line.rfind(":")+1:]
            ll = ll.strip(" \n")
            logl.append(float(ll))

        if "Likelihood of proposed move" in line:
            propl = line[line.rfind(":")+1:]
            propl = propl.strip(" \n")
            pll.append(float(propl))
        count += 1

print(f"Read {count} lines")
print(f"Extracted {meancount} values")


MS = MetroState(param_infos, MCMC_fields, MCMC_fields["num_iters"])

for param in means:
    a = getattr(MS.H, f"mean_{param}")
    a[:len(means[param])] = np.array(means[param])
    setattr(MS.H, f"mean_{param}", a)

MS.H.accept[:len(accept)] = np.array(accept)
MS.H.loglikelihood = np.array(logl)
MS.checkpoint(os.path.join(wdir, out_fname))
