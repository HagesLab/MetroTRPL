# MetroTRPL
Efficient parameter fitting of TRPL curves using a Metropolis-Hastings Monte Carlo sampling algorithm, as described in the article "Rapid optoelectronic characterization of semiconductors combining Bayesian inference with Metropolis sampling" (2023) by C. Fai, A. J. C. Ladd, and C. Hages.

[![DOI](https://zenodo.org/badge/460199913.svg)](https://zenodo.org/badge/latestdoi/460199913)
## Required Packages
Compatible version ranges coming soon!

* numpy 1.20
* numba 0.54.1
* scipy 1.7.1
* csv
* logging
* signal
* pickle
* os
* sys
* time
* datetime

The Anaconda Distribution contains all of the above packages.

## File Overview

* MCMC_script_writer.py - Generates a configuration file(s) containing settings and initial guesses for the Metropolis sampler. Usage: `python MCMC_script_writer.py`. 

For scripting with a SLURM scheduler or other resource-managed machine, `python MCMC_script_writer.py [ID] [config file header]`, where `ID` is an identifier (e.g. the CPU number) and configuration files will be generated as `[config file header][ID].txt`.

* main.py - Starts a Metropolis sampling run with the settings specified in a configuration file. Usage: `python main.py`.

For scripting with a SLURM scheduler or other resource-managed machine, `python main.py [config file header]`, which will search for configuration files named `[output file header][ID].txt`.

* bayes_io.py - Utilities for reading and validating configuration files and measurement data files.
* forward_solver.py - Carrier transport model integrated using scipy solve_ivp().
* mcmc_logging.py - Progress of sampling is written occasionally to a log file.
* metropolis.py - Management of Metropolis algorithm, including initial state selection, proposal of trial moves, simulation of carrier physics, likelihood calculation, and trial move acceptance.
* sim_utils.py - Objects storing results and states visited by sampling run.

* Inputs - Sample measurement data and initial excitation files.
* Tests - Unit tests. Requires the `unittest` and `coverage` packages. Usage: `coverage run -m unittest discover`.
* Visualization - Sample codes to plot the trace plots and histograms in the style of the article.

## Usage
1. Prepare a measurement data file. The measurement data file must be .csv with three columns - [timestamp, intensity, uncertainty], in which a list of *intensities* (e.g. a TRPL decay) are measured at specific *timestamps* with known experimental *uncertainties*. Each measurement must start with a timestamp at time = 0. If the inference is to be performed on multiple measurements (e.g. a power scan consisting of multiple TRPL decay curves), these should be vertically stacked. See the Inputs folder for examples.
2. Prepare an initial excitation file. The initial excitation file must be .csv with one row per measurement (TRPL curve). Each row must be a list of the initial carrier densities at the spatial positions to be integrated by the numerical solver. If the material thickness is 1000 nm and the carrier profile is to be solved at L=100 nodes, for example, the carrier densities at z=5, 15, 25, ...985, 995 nm must be provided. See the Inputs folder for examples.
3. Generate a configuration file using MCMC_script_writer.py. Set the flag verbose=True in the function generate_config_script_file() to view a detailed explanation of each setting. An example configuration file is provided in the Inputs folder.
4. Run the Metropolis sampler using main.py. This outputs a log file detailing the random walk's progress and outputs a MetroState object containing the log-likelihoods and locations of all visited states.
5. Use vis.py in the Visualization folder (as an example) to plot data from the MetroState object.
