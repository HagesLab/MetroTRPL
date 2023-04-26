# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:29:45 2022

@author: cfai2
"""
import os
import pickle
import numpy as np

# Always do this,even if only one CPU


def join_chains(dirname, exclude=None):
    if exclude is None:
        exclude = []
    if not isinstance(exclude, list):
        raise TypeError("exclude must be list of chain indices")

    MC_chains = []
    for d in os.listdir(dirname):
        if d.endswith("final.pik") and d != "Ffinal.pik" and int(d[d.find("CPU")+len("CPU"):d.rfind("-final.pik")]) not in exclude:
            MC_chains.append(os.path.join(dirname, d))

    print(MC_chains)

    with open(os.path.join(MC_chains[0]), "rb") as ifstream:
        base_MS = pickle.load(ifstream)

    param_info = base_MS.param_info

    try:
        sim_flags = base_MS.MCMC_fields
    except AttributeError:
        # Deprecated, MS to include MS.sim_flags soon
        with open(os.path.join(dirname, "sim_flags.pik"), "rb") as ifstream:
            sim_flags = pickle.load(ifstream)

    sim_flags["num_chains"] = len(MC_chains)

    names = param_info["names"]

    # Make more room for all chains
    for param in names:
        setattr(base_MS.H, param, np.add.outer(
            np.zeros(len(MC_chains)), getattr(base_MS.H, param)))
        setattr(base_MS.H, f"mean_{param}", np.add.outer(
            np.zeros(len(MC_chains)), getattr(base_MS.H, f"mean_{param}")))

    accept = np.add.outer(np.zeros(len(MC_chains)), base_MS.H.accept)
    loglikelihood = np.add.outer(np.zeros(len(MC_chains)), base_MS.H.loglikelihood)

    for i, chain in enumerate(MC_chains):
        if i == 0:
            continue  # Already done when making more room

        with open(os.path.join(chain), 'rb') as ifstream:
            next_MS = pickle.load(ifstream)

        for param in names:  # Transfer each chain's values
            val = getattr(base_MS.H, param)
            val[i] = getattr(next_MS.H, param)
            setattr(base_MS.H, param, val)

            val = getattr(base_MS.H, f"mean_{param}")
            val[i] = getattr(next_MS.H, f"mean_{param}")
            setattr(base_MS.H, f"mean_{param}", val)

        accept[i] = next_MS.H.accept
        loglikelihood[i, :len(next_MS.H.loglikelihood)] = next_MS.H.loglikelihood[:len(loglikelihood[i])]

    base_MS.H.accept = accept
    base_MS.H.loglikelihood = loglikelihood
    base_MS.sim_flags = sim_flags

    with open(os.path.join(dirname, "Ffinal.pik"), "wb+") as ofstream:
        pickle.dump(base_MS, ofstream)


def unitconvert(path):
    with open(os.path.join(path), 'rb') as ifstream:
        MS = pickle.load(ifstream)

    param_info = MS.param_info

    names = param_info["names"]
    unit_conversions = param_info["unit_conversions"]

    for param in names:  # Transfer each chain's values
        val = getattr(MS.H, param)
        uc = unit_conversions.get(param, 1)
        setattr(MS.H, param, val / uc)

        val = getattr(MS.H, f"mean_{param}")
        setattr(MS.H, f"mean_{param}", val / uc)

    with open(os.path.join(path), "wb+") as ofstream:
        pickle.dump(MS, ofstream)
