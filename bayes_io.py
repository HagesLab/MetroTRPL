#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 18:10:38 2022

@author: cfai2304
"""
import sys
import csv
import os
import datetime
import numpy as np

from bayes_validate import validate_grid, validate_MCMC_fields, validate_meas_flags, validate_param_info

# Eventually use io_utils for this
def get_split_and_clean_line(line: str):
    """Split line by colon symbol ':' and
    remove preceding and trailing spaces."""
    # FIXME: This causes absolute file paths to not work
    split_line = line.split(':')
    split_line = [i.strip() for i in split_line]
    return split_line


def extract_values(string, delimiter, dtype=float):
    """Converts a string with deliimiters into a list of [dtype] values"""
    # E.g. "100,200,300" with "," delimiter becomes
    # [100,200,300] with dtype=float,
    # or ["100", "200", "300"] with dtype=str
    values = string.split(delimiter)
    values = np.array(values, dtype=dtype)
    return values


def extract_tuples(string, delimiter, dtype=float):
    """
    Converts a string of arbitrary lengthed tuples
    separated by delimiter into a list of tuples
    This one supports inf and -inf as values
    """
    tuples_as_str = string.split(delimiter)
    tuples = []

    for ts in tuples_as_str:
        ts = ts.strip("()")
        vals = ts.split(", ")
        for i in range(len(vals)):
            if vals[i] == "-inf":
                vals[i] = -np.inf
            elif vals[i] == "inf":
                vals[i] = np.inf
            elif dtype == float:
                vals[i] = float(vals[i])
            elif dtype == int:
                vals[i] = int(vals[i])
            else:
                continue
        tuples.append(tuple(vals))
    return tuples


def get_data(exp_file, meas_types, ic_flags, MCMC_fields, verbose=False):
    TIME_RANGE = ic_flags['time_cutoff']
    SELECT = ic_flags['select_obs_sets']
    NOISE_LEVEL = ic_flags.get('noise_level', 0)

    LOG_PL = MCMC_fields['log_pl']
    NORMALIZE = MCMC_fields["self_normalize"]
    resample = ic_flags.get("resample", 1)

    bval_cutoff = sys.float_info.min

    data = np.loadtxt(exp_file, delimiter=",")

    times = data[:, 0]
    y = data[:, 1]
    uncertainty = data[:, 2]

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

    for i in range(len(t_list)):
        t_list[i] = t_list[i][::resample]
        y_list[i] = y_list[i][::resample]
        u_list[i] = u_list[i][::resample]

    if NORMALIZE is not None:
        for i in range(len(t_list)):
            if meas_types[i] in NORMALIZE:
                norm_f = np.nanmax(y_list[i])
                y_list[i] /= norm_f
                u_list[i] /= norm_f

    if LOG_PL:
        # Deal with noisy negative values before taking log
        for i in range(len(t_list)):
            y_list[i] = np.abs(y_list[i])
            y_list[i][y_list[i] < bval_cutoff] = bval_cutoff

            u_list[i] /= y_list[i]
            u_list[i] /= np.log(10)  # Since we use log10 instead of ln
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


def get_initpoints(init_file, ic_flags):
    select = ic_flags['select_obs_sets']

    with open(init_file, newline='', encoding=None) as file:
        ifstream = csv.reader(file)
        initpoints = []
        for row in ifstream:
            if len(row) == 0:
                continue
            initpoints.append(row)

    if select is not None:
        initpoints = np.array(initpoints)[select]
    return np.array(initpoints, dtype=float)


def make_dir(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)


def clear_checkpoint_dir(MCMC_fields):
    if MCMC_fields["load_checkpoint"] is None:
        for chpt in os.listdir(MCMC_fields["checkpoint_dirname"]):
            os.remove(os.path.join(MCMC_fields["checkpoint_dirname"], chpt))


def put_into_param_info(param_info, vals, new_key):
    if "names" not in param_info:
        raise KeyError("Entry \"Param names\" not found in MCMC config file.\n"
                       "Check whether this entry is present and FIRST in\n"
                       "the Param Info subsection.")
    param_info[new_key] = {param_info["names"][i]: vals[i]
                           for i in range(len(param_info["names"]))}
    return

def insert_scale_factors(grid, param_info, meas_fields, MCMC_fields):
    scale_f = MCMC_fields.get("scale_factor", None)
    if scale_f is None:
        return
    
    scale_type = scale_f[0]
    scale_init = scale_f[1]
    scale_var = scale_f[2]
    if scale_type == "global":
        param_info["names"].append("_s")
        param_info["do_log"]["_s"] = 1
        param_info["prior_dist"]["_s"] = (-np.inf, np.inf)
        param_info["init_guess"]["_s"] = scale_init
        param_info["init_variance"]["_s"] = scale_var
        param_info["active"]["_s"] = 1

    elif scale_type == "ind":
        if meas_fields["select_obs_sets"] is not None:
            num_meas = len(meas_fields["select_obs_sets"])
        else:
            num_meas = grid["num_meas"]
        for i in range(num_meas):
            param_info["names"].append(f"_s{i}")
            param_info["do_log"][f"_s{i}"] = 1
            param_info["prior_dist"][f"_s{i}"] = (-np.inf, np.inf)
            param_info["init_guess"][f"_s{i}"] = scale_init
            param_info["init_variance"][f"_s{i}"] = scale_var
            param_info["active"][f"_s{i}"] = 1
    return

def insert_fluences(param_info, MCMC_fields, fluences):
    ff = MCMC_fields.get("fittable_fluences", None)
    if ff is None:
        return

    f_var = ff[0]

    # The FIRST value in each c_grp determines the name
    # of the fluence parameter used by all in the c_grp
    # All subsequent values do not need their own fluence parameters
    c_grp_dependents = []
    if ff[2] is None or len(ff[2]) == 0:
        pass
    else:
        for c_grp in ff[2]:
            c_grp_dependents += list(c_grp)[1:]

    for i in ff[1]:
        if i in c_grp_dependents:
            continue
        param_info["names"].append(f"_f{i}")
        param_info["do_log"][f"_f{i}"] = 1
        param_info["prior_dist"][f"_f{i}"] = (0, np.inf)
        param_info["init_guess"][f"_f{i}"] = fluences[i]
        param_info["init_variance"][f"_f{i}"] = f_var
        param_info["active"][f"_f{i}"] = 1
    return

def remap_fittable_inds(fittables : np.ndarray | list[int], select_obs_sets : list) -> np.ndarray:
    """
    Reassign new fittable indices (e.g. for fittable_fluence's 2nd argument)
    according to subset of measurements requested by select_obs_sets

    This is essentially the intersection of fittables and select_obs_sets,
    but working on ordered lists rather than sets

    E.g.: select_obs_sets = [0, 2, 4]; fittables = [0, 1, 3, 4]
    MMC will refer to the remaining three measurements as "0", "1", and "2",
    i.e. 0 -> 0, 2 -> 1, 4 -> 2
    so a new_fittables of [0, 2] is required to correspond with the originally
    requested 0 and 4.
    Meanwhile, measurements 1 and 3 are omitted by select_obs_sets, so they should
    be absent from new_fittables.
    """
    new_fittables = []
    for i, s in enumerate(select_obs_sets):
        if s in fittables:
            new_fittables.append(i)

    return np.array(new_fittables)

def remap_constraint_grps(c_grps : list[tuple], select_obs_sets : list) -> list[tuple]:
    """
    Reassign new constraint groups (e.g. for fittable_fluence's 3rd argument)
    according to subset of measurements requested by select_obs_sets

    E.g.: select_obs_sets = [0, 2, 4]; c_grps = [(0, 1, 2), (3, 4, 5)]
    MMC will refer to the remaining three measurements as "0", "1", and "2",
    i.e. 0 -> 0, 2 -> 1, 4 -> 2
    so every occurence of 0, 2, and 4 in c_grps must be replaced with 0, 1, and 2 respectively.
    Measurements 1, 3, and 5 are omitted by select_obs_sets, so they should be removed from
    c_grps.
    This leaves [(0, 1), (2, )]
    However, constraint groups of size 1 are meaningless (such a measurement will not share a
    fittable fluence with any other measurement), so they too can be removed.
    This leaves [(0, 1)]
    """
    new_c_grps = []
    for grp in c_grps:
        new_c_grp = []
        for val in grp:
            if val in select_obs_sets:
                new_c_grp.append(select_obs_sets.index(val))

        if len(new_c_grp) > 1:
            new_c_grps.append(tuple(new_c_grp))
        
    return new_c_grps

def read_config_script_file(path):
    with open(path, 'r') as ifstream:
        grid = {}
        param_info = {}
        meas_flags = {}
        MCMC_fields = {}

        init_flag = 0

        if not ("$$ MCMC CONFIG CREATED") in next(ifstream):
            raise OSError("Error: this file is not a valid MCMC config file")

        # Extract parameters, ICs
        for line in ifstream:
            line = line.strip('\n')
            line_split = get_split_and_clean_line(line)

            if ("#" in line) or not line:
                continue

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
                        grid["lengths"] = extract_values(
                            line_split[1], delimiter='\t')

                    elif line.startswith("nx"):
                        grid["nx"] = extract_values(
                            line_split[1], delimiter='\t', dtype=int)

                    elif line.startswith("Measurement type(s)"):
                        grid["meas_types"] = line_split[1].split('\t')

                    elif line.startswith("Number of measurements"):
                        grid["num_meas"] = int(line_split[1])

                if (init_flag == 'p'):
                    if line.startswith("Param Names"):
                        param_info["names"] = line_split[1].split('\t')

                    elif line.startswith("Unit conversions"):
                        vals = extract_values(
                            line_split[1], delimiter='\t', dtype=float)
                        put_into_param_info(
                            param_info, vals, "unit_conversions")

                    elif line.startswith("Do logscale"):
                        vals = extract_values(
                            line_split[1], delimiter='\t', dtype=int)
                        put_into_param_info(param_info, vals, "do_log")

                    elif line.startswith("Active"):
                        vals = extract_values(
                            line_split[1], delimiter='\t', dtype=int)
                        put_into_param_info(param_info, vals, "active")

                    elif line.startswith("Initial guess"):
                        vals = extract_values(
                            line_split[1], delimiter='\t', dtype=float)
                        put_into_param_info(param_info, vals, "init_guess")

                    elif line.startswith("Prior"):
                        vals = extract_tuples(line_split[1], delimiter='\t')
                        put_into_param_info(param_info, vals, "prior_dist")

                    elif line.startswith("Initial variance"):
                        vals = extract_values(
                            line_split[1], delimiter='\t', dtype=float)
                        put_into_param_info(param_info, vals, "init_variance")

                    elif line.startswith("Mu constraint"):
                        vals = extract_values(
                            line_split[1], delimiter='\t', dtype=float)
                        param_info["do_mu_constraint"] = vals

                if (init_flag == 'm'):
                    if line.startswith("Time cutoffs"):
                        vals = extract_values(
                            line_split[1], delimiter='\t', dtype=float)
                        meas_flags["time_cutoff"] = vals

                    elif line.startswith("Select measurement"):
                        if line_split[1] == "None":
                            meas_flags["select_obs_sets"] = None
                        else:
                            meas_flags["select_obs_sets"] = extract_values(line_split[1],
                                                                           delimiter='\t',
                                                                           dtype=int)
                            meas_flags["select_obs_sets"] = list(meas_flags["select_obs_sets"])

                    elif line.startswith("Added noise level"):
                        if line_split[1] == "None":
                            meas_flags["noise_level"] = None
                        else:
                            meas_flags["noise_level"] = float(line_split[1])

                    elif line.startswith("Resample"):
                        meas_flags["resample"] = int(line_split[1])
                            
                if (init_flag == 's'):
                    if line.startswith("Num iters"):
                        MCMC_fields["num_iters"] = int(line_split[1])
                    elif line.startswith("Solver name"):
                        MCMC_fields["solver"] = tuple(line_split[1].split('\t'))
                    elif line.startswith("Solver rtol"):
                        MCMC_fields["rtol"] = float(line_split[1])
                    elif line.startswith("Solver atol"):
                        MCMC_fields["atol"] = float(line_split[1])
                    elif line.startswith("Solver hmax"):
                        MCMC_fields["hmax"] = float(line_split[1])
                    elif line.startswith("Repeat hmax"):
                        MCMC_fields["verify_hmax"] = int(line_split[1])
                    elif line.startswith("Annealing Controls"):
                        MCMC_fields["annealing"] = tuple(
                            extract_values(line_split[1].strip("()"), ", "))
                    elif line.startswith("Likelihood-to-variance"):
                        MCMC_fields["likel2variance_ratio"] = float(
                            line_split[1])
                    elif line.startswith("Force equal mu"):
                        MCMC_fields["override_equal_mu"] = int(line_split[1])
                    elif line.startswith("Force equal S"):
                        MCMC_fields["override_equal_s"] = int(line_split[1])
                    elif line.startswith("Use log of measurements"):
                        MCMC_fields["log_pl"] = int(line_split[1])
                    elif line.startswith("Scale factor"):
                        if line_split[1] == "None":
                            MCMC_fields["scale_factor"] = None
                        else:
                            MCMC_fields["scale_factor"] = line_split[1].split('\t')
                            MCMC_fields["scale_factor"][1] = float(MCMC_fields["scale_factor"][1]) # type: ignore
                            MCMC_fields["scale_factor"][2] = float(MCMC_fields["scale_factor"][2]) # type: ignore
                    elif line.startswith("Fittable fluences"):
                        if line_split[1] == "None":
                            meas_flags["fittable_fluences"] = None
                        else:
                            init_var, inds, c_grps = line_split[1].split("\t")

                            init_var = float(init_var)

                            inds = inds.strip("[]")
                            inds = extract_values(inds, delimiter=", ", dtype=int)

                            if c_grps == "None":
                                c_grps = None
                            else:
                                c_grps = c_grps.strip("[]")
                                c_grps = extract_tuples(c_grps, delimiter="|", dtype=int)

                            meas_flags["fittable_fluences"] = [init_var, inds, c_grps]
                    elif line.startswith("Normalize these meas and sim types"):
                        if line_split[1] == "None":
                            MCMC_fields["self_normalize"] = None
                        else:
                            MCMC_fields["self_normalize"] = line_split[1].split('\t')
                    elif line.startswith("Proposal function"):
                        MCMC_fields["proposal_function"] = line_split[1]
                    elif line.startswith("Use hard boundaries"):
                        MCMC_fields["hard_bounds"] = int(line_split[1])
                    elif line.startswith("IRF"):
                        if line_split[1] == "None":
                            MCMC_fields["irf_convolution"] = None
                        else:
                            MCMC_fields["irf_convolution"] = extract_values(line_split[1],
                                                                            delimiter='\t',
                                                                            dtype=float)
                    elif line.startswith("Propose params one-at-a-time"):
                        MCMC_fields["one_param_at_a_time"] = int(line_split[1])
                    elif line.startswith("Checkpoint dir"):
                        MCMC_fields["checkpoint_dirname"] = os.path.join(
                            line_split[1])
                    elif line.startswith("Checkpoint fileheader"):
                        MCMC_fields["checkpoint_header"] = line_split[1]
                    elif line.startswith("Checkpoint freq"):
                        MCMC_fields["checkpoint_freq"] = int(line_split[1])
                    elif line.startswith("Load checkpoint"):
                        if line_split[1] == "None":
                            MCMC_fields["load_checkpoint"] = None
                        else:
                            MCMC_fields["load_checkpoint"] = line_split[1]
                    elif line.startswith("Initial condition path"):
                        MCMC_fields["init_cond_path"] = os.path.join(
                            line_split[1])
                    elif line.startswith("Measurement path"):
                        MCMC_fields["measurement_path"] = os.path.join(
                            line_split[1])
                    elif line.startswith("Output path"):
                        MCMC_fields["output_path"] = os.path.join(line_split[1])

    validate_grid(grid)
    validate_param_info(param_info)
    validate_meas_flags(meas_flags, grid["num_meas"])
    validate_MCMC_fields(MCMC_fields, grid["num_meas"])

    insert_scale_factors(grid, param_info, meas_flags, MCMC_fields)

    # Keep fittable_fluence indices consistent after subsetting with select_obs_sets
    if MCMC_fields.get("fittable_fluences", None) is not None and meas_flags["select_obs_sets"] is not None:
        MCMC_fields["fittable_fluences"][1] = remap_fittable_inds(MCMC_fields["fittable_fluences"][1],
                                                                  meas_flags["select_obs_sets"])
        MCMC_fields["fittable_fluences"][2] = remap_constraint_grps(MCMC_fields["fittable_fluences"][2],
                                                                   meas_flags["select_obs_sets"])

    return grid, param_info, meas_flags, MCMC_fields

def generate_config_script_file(path, simPar, param_info, measurement_flags,
                                MCMC_fields, verbose=False):
    validate_grid(simPar)
    validate_param_info(param_info)
    validate_meas_flags(measurement_flags, simPar["num_meas"])
    validate_MCMC_fields(MCMC_fields, simPar["num_meas"])
    if not path.endswith(".txt"):
        path += ".txt"

    with open(path, "w+") as ofstream:
        date = datetime.datetime.now().date()
        time = datetime.datetime.now().time()
        ofstream.write("$$ MCMC CONFIG CREATED {} AT {}\n".format(date, time))
        ofstream.write("##\n")
        ofstream.write("p$ Space Grid:\n")
        if verbose:
            ofstream.write(
                "# List of material/system thicknesses - one per measurement\n")
        lengths = simPar["lengths"]
        ofstream.write(f"Length(s): {lengths[0]}")
        for value in lengths[1:]:
            ofstream.write(f"\t{value}")
        ofstream.write('\n')
        if verbose:
            ofstream.write(
                "# Number of space nodes used by solver discretization\n")
        nx = simPar["nx"]
        ofstream.write(f"nx: {nx[0]}")
        for value in nx[1:]:
            ofstream.write(f"\t{value}")
        ofstream.write('\n')
        if verbose:
            ofstream.write("# Model to use to simulate each measurement\n")
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
        if verbose:
            ofstream.write("# List of names of parameters used in the model\n")
        param_names = param_info["names"]
        ofstream.write(f"Param Names: {param_names[0]}")
        for value in param_names[1:]:
            ofstream.write(f"\t{value}")
        ofstream.write('\n')

        if verbose:
            ofstream.write(
                "# Conversion from units params are entered in to units used by model\n")
        ucs = param_info["unit_conversions"]
        ofstream.write(f"Unit conversions: {ucs.get(param_names[0], 1)}")
        for name in param_names[1:]:
            ofstream.write(f"\t{ucs.get(name, 1)}")
        ofstream.write('\n')

        if verbose:
            ofstream.write("# Whether the MCMC should work with the log of each param."
                           " The answer should be YES for most models. \n")
        do_log = param_info["do_log"]
        ofstream.write(f"Do logscale: {do_log.get(param_names[0], 0)}")
        for name in param_names[1:]:
            ofstream.write(f"\t{do_log.get(name, 0)}")
        ofstream.write('\n')

        if verbose:
            ofstream.write("# Whether the MCMC should propose new moves for this parameter. "
                           "Setting this to 0 or False fixes the parameter at its initial value.\n")
        active_params = param_info["active"]
        ofstream.write(f"Active: {active_params.get(param_names[0], 0)}")
        for name in param_names[1:]:
            ofstream.write(f"\t{active_params.get(name, 0)}")
        ofstream.write('\n')

        if verbose:
            ofstream.write(
                "# Bounds of prior distribution for each parameter.\n")
        prior_dist = param_info["prior_dist"]
        ofstream.write(
            f"Prior: {prior_dist.get(param_names[0], (-np.inf, np.inf))}")
        for name in param_names[1:]:
            ofstream.write(f"\t{prior_dist.get(name, (-np.inf, np.inf))}")
        ofstream.write('\n')

        if verbose:
            ofstream.write("# Initial values for each parameter.\n")
        init_guess = param_info["init_guess"]
        ofstream.write(f"Initial guess: {init_guess.get(param_names[0], 0)}")
        for name in param_names[1:]:
            ofstream.write(f"\t{init_guess.get(name, 0)}")
        ofstream.write('\n')

        if verbose:
            ofstream.write("# Initial proposal variance for each parameter. "
                           "I.e. how far from the current parameters new proposals will go.\n")
        init_variance = param_info["init_variance"]
        ofstream.write(
            f"Initial variance: {init_variance.get(param_names[0], 0)}")
        for name in param_names[1:]:
            ofstream.write(f"\t{init_variance.get(name, 0)}")
        ofstream.write('\n')

        if "do_mu_constraint" in param_info:
            if verbose:
                ofstream.write("# Restrict mu_n and mu_p within a small range of ambipolar mobility. "
                               "Ambipolar mobility is limited within A +/- B.\n")
            mu = param_info["do_mu_constraint"]
            ofstream.write(f"Mu constraint: {mu[0]}\t{mu[1]}\n")
        #######################################################################
        ofstream.write("##\n")
        ofstream.write("p$ Measurement handling flags:\n")

        if verbose:
            ofstream.write("# Truncate measurements to only those within this time range. "
                           "Inf values indicate an unbounded range. \n")
        tc = measurement_flags["time_cutoff"]
        ofstream.write(f"Time cutoffs: {tc[0]}\t{tc[1]}\n")

        if verbose:
            ofstream.write("# Which measurements in a sequence to keep for MCMC. "
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

        if "noise_level" in measurement_flags:
            if verbose:
                ofstream.write("# Whether to add Gaussian noise of the indicated magnitude to "
                               "the measurement.\n# This should be None (zero noise) unless testing "
                               "with simulated data.\n")
            noise_level = measurement_flags["noise_level"]
            ofstream.write(f"Added noise level: {noise_level}\n")

        if "resample" in measurement_flags:
            if verbose:
                ofstream.write("# Resample the measurement,"
                               "taking only every n points.\n"
                               "# This can speed up the simulations a little.\n")
            resample_factor = measurement_flags["resample"]
            ofstream.write(f"Resample: {resample_factor}\n")
        #######################################################################
        ofstream.write("##\n")
        ofstream.write("p$ MCMC Control flags:\n")
        if verbose:
            ofstream.write("# How many samples to propose.\n")
        num_iters = MCMC_fields["num_iters"]
        ofstream.write(f"Num iters: {num_iters}\n")
        if verbose:
            ofstream.write(
                "# Which solver engine to use - solveivp (more robust), odeint (sometimes faster),"
                "# or NN (experimental!).\n")
        solver = MCMC_fields["solver"]
        ofstream.write(f"Solver name: {solver[0]}")
        for value in solver[1:]:
            ofstream.write(f"\t{value}")
        ofstream.write("\n")
        if "rtol" in MCMC_fields:
            if verbose:
                ofstream.write("# Solver engine relative tolerance.\n")
            rtol = MCMC_fields["rtol"]
            ofstream.write(f"Solver rtol: {rtol}\n")
        if "atol" in MCMC_fields:
            if verbose:
                ofstream.write("# Solver engine absolute tolerance.\n")
            atol = MCMC_fields["atol"]
            ofstream.write(f"Solver atol: {atol}\n")
        if "hmax" in MCMC_fields:
            if verbose:
                ofstream.write(
                    "# Solver engine maximum adaptive time stepsize.\n")
            hmax = MCMC_fields["hmax"]
            ofstream.write(f"Solver hmax: {hmax}\n")

        if "verify_hmax" in MCMC_fields:
            if verbose:
                ofstream.write("# (Experimental) If 1 or True, MCMC will repeat simulations "
                               "with progressively smaller hmax until the results converge.\n")
            verify_hmax = MCMC_fields["verify_hmax"]
            ofstream.write(f"Repeat hmax: {verify_hmax}\n")

        if verbose:
            ofstream.write("# Annealing schedule parameters.\n"
                           "# (Starting model uncertainty, steprate, final model uncertainty)\n"
                           "# Will drop one order of magnitude per STEPRATE samples until FINAL is reached.\n")
        anneal = MCMC_fields["annealing"]
        ofstream.write(f"Annealing Controls: {anneal}\n")

        if verbose:
            ofstream.write("# Ratio to maintain betwen Model uncertainty and proposal variance.\n"
                           "# Model uncertainty will be taken as this times proposal variance.\n")
        l2v = MCMC_fields["likel2variance_ratio"]
        ofstream.write(f"Likelihood-to-variance: {l2v}\n")

        if "override_equal_mu" in MCMC_fields:
            if verbose:
                ofstream.write(
                    "# Force parameters mu_n and mu_p to be equal.\n")
            emu = MCMC_fields["override_equal_mu"]
            ofstream.write(f"Force equal mu: {emu}\n")
        if "override_equal_s" in MCMC_fields:
            if verbose:
                ofstream.write("# Force parameters Sf and Sb to be equal.\n")
            es = MCMC_fields["override_equal_s"]
            ofstream.write(f"Force equal S: {es}\n")

        if verbose:
            ofstream.write("# Compare log of measurements and simulations for "
                           "purpose of likelihood evaluation. Recommended to be 1 or True. \n")
        logpl = MCMC_fields["log_pl"]
        ofstream.write(f"Use log of measurements: {logpl}\n")

        if verbose:
            ofstream.write("# Normalize all individual measurements and simulations "
                           "to maximum of 1 before likelihood evaluation. "
                           "\n# Global scaling coefficients named '_s#' may optionally be defined in MCMC_fields by "
                           "\n# enabling the scale_factor setting. "
                           "\n# If the absolute units or efficiency of the measurement is unknown, "
                           "\n# it is recommended to try fitting '_s' instead of relying on normalization. "
                           "\n# Enabling this will disable _s for the selected measurements. \n")
        norm = MCMC_fields["self_normalize"]

        if norm is None:
            ofstream.write(f"Normalize these meas and sim types: {norm}")
        else:
            ofstream.write(f"Normalize these meas and sim types: {norm[0]}")
            for value in norm[1:]:
                ofstream.write(f"\t{value}")
        ofstream.write('\n')

        if "scale_factor" in MCMC_fields:
            if verbose:
                ofstream.write("# Add additional scale factors that MMC will attempt to apply on the simulations "
                               "\n# to better fit measurement data curves. "
                               "\n# Must be None, or a list/tuple of three elements: "
                               "\n# First element \"global\", which will add a single scaling factor \"_s\" shared by all curves, "
                               "\n# or \"ind\", which will add independent scaling factors \"_s0\", \"_s1\", \"_s2\", ... for each curve. "
                               "\n# Second element an initial guess. 1 means no scaling is applied."
                               "\n# Third element an initial variance, similar to the initial_variance parameter for other parameters."
                               "\n# All scale factors will get the same initial guess and variance. \n")
                scale_f = MCMC_fields["scale_factor"]
                if scale_f is None:
                    ofstream.write(f"Scale factor: {scale_f}")
                else:
                    ofstream.write(f"Scale factor: {scale_f[0]}\t{scale_f[1]}\t{scale_f[2]}")
            ofstream.write('\n')

        if "fittable_fluences" in MCMC_fields:
            if verbose:
                ofstream.write("# Whether to try inferring the fluences. None means it will keep"
                               " the fluence values as entered;\n# otherwise, a list of three elements:\n"
                               "# 1. An initial variance value, as in initial_variance.\n"
                               "# All fluences are fitted by log scale and will use the same variance.\n"
                               "# 2. A list of indices for measurements for which fluences will be fitted.\n"
                               "# e.g. [0, 1, 2] means vary the fluences for the first, second, and third measurements.\n"
                               "# Additional parameters named _f0, _f1, _f2... will be created for such measurements.\n"
                               "# 3. Either None, in which all fluences will be independently fitted, or\n"
                               "# A list of constraint groups, in which each measurement in a group will share a fluence\n"
                               "# with all other members. E.g. [(0, 2, 4), (1, 3, 5)] means that the third and fifth\n"
                               "# measurments will share a fluence value with the first, \n"
                               "# while the fourth and sixth measurements will share a fluence value with the second.\n")
                ff = MCMC_fields["fittable_fluences"]
                if ff is None:
                    ofstream.write(f"Fittable fluences: {ff}\n")
                else:
                    ofstream.write(f"Fittable fluences: {ff[0]}\t")
                    ofstream.write(f"{ff[1]}\t")
                    if ff[2] is None:
                        ofstream.write(f"{ff[2]}")
                    else:
                        ofstream.write(f"{ff[2][0]}")
                        for c_grp in ff[2][1:]:
                            ofstream.write(f"|{c_grp}")
                    ofstream.write("\n")

        if verbose:
            ofstream.write("# Proposal function used to generate new states. "
                           "Box for joint uniform box and Gauss for multivariate Gaussian. \n")
        prop_f = MCMC_fields["proposal_function"]
        ofstream.write(f"Proposal function: {prop_f}\n")

        if "hard_bounds" in MCMC_fields:
            if verbose:
                ofstream.write("# Whether to coerce params to stay within the bounds "
                               "listed in metropolis.check_approved_param(). \n"
                               "# =1 will coerce while =0 will only warn.\n")
            bound = MCMC_fields["hard_bounds"]
            ofstream.write(f"Use hard boundaries: {bound}\n")

        if verbose:
            ofstream.write(
                "# None for no convolution, or a list of wavelengths whose IRF profiles\n"
                "# will be used to convolute each simulated TRPL curve. One wavelength per"
                " measurement.\n")
        if "irf_convolution" in MCMC_fields:
            irf = MCMC_fields["irf_convolution"]
            if irf is None:
                ofstream.write(f"IRF: {irf}")
            else:
                ofstream.write(f"IRF: {irf[0]}")
                for value in irf[1:]:
                    ofstream.write(f"\t{value}")
            ofstream.write('\n')

        if verbose:
            ofstream.write(
                "# Whether a proposed move should change in one param or all params at once.\n")
        gibbs = MCMC_fields["one_param_at_a_time"]
        ofstream.write(f"Propose params one-at-a-time: {gibbs}\n")

        if verbose:
            ofstream.write("# Directory checkpoint files stored in.\n")
        chpt_d = MCMC_fields["checkpoint_dirname"]
        ofstream.write(f"Checkpoint dir: {chpt_d}\n")

        if verbose:
            ofstream.write(
                "# An optional tag to append to the filename of each checkpoint.\n")
        chpt_h = MCMC_fields["checkpoint_header"]
        ofstream.write(f"Checkpoint fileheader: {chpt_h}\n")

        if verbose:
            ofstream.write("# Checkpoint saved every 'this many' samples.\n")
        chpt_f = MCMC_fields["checkpoint_freq"]
        ofstream.write(f"Checkpoint freq: {chpt_f}\n")

        if verbose:
            ofstream.write(
                "# Name of a checkpoint file to resume an MCMC from.\n")
        load_chpt = MCMC_fields["load_checkpoint"]
        ofstream.write(f"Load checkpoint: {load_chpt}\n")

        if verbose:
            ofstream.write(
                "# Path from which to read initial condition arrays. \n")
        ic = MCMC_fields["init_cond_path"]
        ofstream.write(f"Initial condition path: {ic}\n")

        if verbose:
            ofstream.write(
                "# Path from which to read measurement data arrays. \n")
        mc = MCMC_fields["measurement_path"]
        ofstream.write(f"Measurement path: {mc}\n")

        if verbose:
            ofstream.write("# Path from which to save output MCMC objects. \n")
        oc = MCMC_fields["output_path"]
        ofstream.write(f"Output path: {oc}\n")

    return

if __name__ == "__main__":
    grid, param_info, meas_flags, MCMC_fields = read_config_script_file(
        "mcmc0.txt")
    print(grid)
    print(param_info)
    print(meas_flags)
    print(MCMC_fields)
