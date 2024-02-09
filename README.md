# MetroTRPL
Efficient parameter fitting of TRPL curves using a Metropolis-Hastings Monte Carlo sampling algorithm, as described in the article "Rapid optoelectronic characterization of semiconductors combining Bayesian inference with Metropolis sampling" (2023) by C. Fai, A. J. C. Ladd, and C. Hages.

[![DOI](https://zenodo.org/badge/460199913.svg)](https://zenodo.org/badge/latestdoi/460199913)
For a quick start, download the latest release under the **Releases** tab on the right of the page.

## Required Packages
Python 3.10+ is needed to run all features of this program, particularly the GUI visualization. Earlier versions of Python 3 can run the core Monte Carlo algorithm with some compatibility tweaks.

* numpy 1.20+
* numba 0.54.1+
* scipy 1.7.1+
* pillow 9.5.0+ - for the GUI
* pywin32 306+ - for the GUI, if using Windows
* tensorflow 2.12+ - if you want to use neural network models
* mpi4py 3.1.5 - if you want to run parallel tempering jobs in parallel with MPI

We recommend creating a virtual environment (e.g. with [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/), or [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)) and installing the packages there. Some sample environment specs are provided under the Requirements directory to help setup your environment.

## Usage
1. Prepare a measurement data file. The measurement data file must be .csv with three columns - [timestamp, intensity, uncertainty], in which a list of *intensities* (e.g. a TRPL decay) are measured at specific *timestamps* with known experimental *uncertainties*. Each measurement must start with a timestamp at time = 0. If the inference is to be performed on multiple measurements (e.g. a power scan consisting of multiple TRPL decay curves), these should be vertically stacked. See the Inputs folder for examples.
2. Prepare an initial excitation file. The initial excitation file must be .csv with one row per measurement (TRPL curve). Each row must be a list of three values for a Beer-Lambert excitation profile - a fluence, in cm^-2, an absorption coefficient, in cm^-1, and a direction flag, which is either 1 for forward excitation or -1 for reverse excitation. See Inputs
/staub_MAPI_threepower_twothick_fluences.csv for an example. Alternatively, each row may be a list of the initial carrier densities at the spatial positions to be integrated by the numerical solver. If the material thickness is 1000 nm and the carrier profile is to be solved at nx=100 nodes, for example, the carrier densities at z=5, 15, 25, ...985, 995 nm must be provided. See the Inputs folder for examples.
3. Generate a configuration file using MCMC_script_writer.py. Usage: `python MCMC_script_writer.py` while within the MetroTRPL folder. Set the flag verbose=True in the function generate_config_script_file() to view a detailed explanation of each setting. An example configuration file is provided in the Inputs folder.
4. Run the Metropolis sampler using main.py. Usage: `python main.py` while within the MetroTRPL folder. **Be sure to set serial_fallback=False in the metro() call if you aren't running an MPI job.** This outputs a log file detailing the random walks' progress and outputs an Ensemble object containing the log-likelihoods and locations of all visited states.
5. Run the GUI folder's main.py to visualize and export results. Usage: `python main.py` while within the MetroTRPL/GUI folder. The GUI folder also contains a tutorial on using the GUI.

## Explaining each setting in MCMC_script_writer.py
1. **simPar** - simulation settings
   * num_meas - Number of measurements in the measurement data.
   * lengths - List of lengths (i.e. sample thicknesses) for each measurement.
   * nx - List of node counts for numerical solver, for each measurement.
   * meas_types - List of what each type of measurement (e.g. TRPL, TRTS) in the data is.
2. **param_info** - material parameter information
   * names - List of names for each material parameter involved in the transport model.
   * active - Whether each parameter is "active" - if active (=1), the MCMC algorithm will propose trial moves for this parameter. Otherwise (=0), the value of this parameter will stay at its initial guess.
   * unit_conversions - Unit conversion factors for each parameter. Used by carrier transport models to convert from the unit system you want to use to input material parameters (e.g. ns for lifetimes, cm^2/V/s for mobilities) to the nm/V/ns unit system used by the numerical solver.
   * do_log - Whether trial moves should be made in log scale. **This should be active (=1) for any parameter for which reasonable values can span several orders of magnitude.**
   * prior_dist - Ranges of reasonable values within which the optimal parameters are expected. For instance, bulk lifetimes may range from a few to a few hundred nanoseconds.
   * init_guess - Initial guess / starting values for each parameter.
   * trial_move - Maximum size of trial move for each parameter. Trial moves will be proposed from a uniform distribution with widths determined by this setting. Smaller moves are accepted more often but also increase the equilibration time for the chains. **Adjust this to maintain an acceptance rate of 10-50%. Though it will vary from measurement data to measurement data, a trial_move of 0.01 with do_log activated is a good first try.**
3. **meas_fields** - measurement data settings
   * time_cutoff - Truncate measurements to this time range. For instance, [0-10] means that only the first 10 nanoseconds of each measurement will be kept for the MCMC algorithm. This can be used to make inferences on specific time regimes of the measurements.
   * select_obs_sets - Select specific measurements out of the measurement data file for the MCMC algorithm. A list such as [0,2] means to keep only the first and third measurements, while omitting the second and others. Set to None to keep ALL measurements.
4. **MCMC_fields** - settings controlling the MCMC algorithm itself.
   * init_cond_path - Location of the initial excitation file.
   * measurement_path - Location of the measurement data file.
   * output_path - Location to save MCMC results.
   * num_iters - Number of iterations / trial moves the MCMC algorithm will try before stopping. **With a trial move of 0.01, most inferences equilibrate within a few thousand iterations.**
   * solver - A tuple of size one indicating which numerical solver to use - solve_ivp (more robust), or odeint (sometimes faster). Can also be set to NN to access a saved neural network, for which the location of the neural network and its scaling factor files must also be provided as second and third arguments.
   * model - Choice of carrier transport model. std for the standard carrier model, or traps for the shallow trapping carrier model.
   * ini_mode - Set to "fluence" if the initial condition input is a fluence/absorption/direction trio, or to "density" if it is a carrier density list.
   * rtol, atol, hmax - Solver tolerances and adaptive stepsize. See the [solve_ivp docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) for more details.
   * model_uncertainty - The model uncertainty, or how selective the sampling is. Smaller values make the precision of inferences higher, up to the limit of your measurement uncertainty, but are harder to equilibrate. **Try starting with 1, and adjust according to your precision needs.**
   * log_y - Whether to compare measurements and simulations on log scale. **Set to 1 for measurements which span many orders of magnitude.**
   * hard_bounds - Whether to automatically reject all trial moves leading outside of the prior_dist boundaries.
   * force_min_y - Whether to raise abnormally small simulation values to the minimum measured data value. May make the MCMC algorithm more sensitive in regions of low probability.
   * checkpoint_freq - Interval in which the MCMC algorithm will save snapshots of partially completed inferences. Checkpoints files are formatted identically to output files.
   * load_checkpoint - If None, starts a new inference. If the file name of a checkpoint is supplied, the MCMC algorithm will continue an inference from where that checkpoint left off.
   * scale_factor (Optional) - Add additional scale factors that the MCMC algorithm will attempt to apply on the simulations to account for calibration/detector efficiency factors between simulations and measurement data. Must be None, in which no scaling will be done to the simulations, or a list of four elements:
     1. A trial move size, as described in **param_info**. All factors are fitted by log scale and will use the same move size.
     2. A list of indices for measurements for which factors will be fitted. e.g. [0, 1, 2] means to create scale factors for the first, second, and third measurements. Additional parameters named _s0, _s1, _s2... will be created for such measurements.
     3. Either None, in which all factors will be independently fitted, or a list of constraint groups, in which each measurement in a group will share a factor with all other members. E.g. [(0, 2, 4), (1, 3, 5)] means that the third and fifth measurments will share a factor value with the first, while the fourth and sixth measurements will share a factor value with the second. Note that in Python, the first item is indexed as zero.
     4. (Optional) A list of initial guesses for each factor. Defaults to 1 if not specified.
    * irf_convolution (Optional) - A list of wavelengths for each measurement, for each of which there should be a corresponding IRF added to the IRFs folder. The MCMC algorithm will convolve the simulation for each measurement with the corresponding IRF by wavelength. Entering a wavelength of zero omits the convolution for the corresponding measurements.
    * parallel_tempering (Optional) - A list of temperature factors used to create an ensemble of chains. One chain per temperature. **We suggest starting with a base temperature of 1 and adding a geometric sequence of additional temperatures.**
    * temper_freq (Optional) - Interval in which the MCMC algorithm will attempt swap moves between chains of adjacent temperature. **We suggest between 10 and 100.**

## FAQ
**ValueErrors from get_data() or get_initpoints() with unusual symbols**

Software such as MS Excel can add invisible formatting or characters to data files. **Be sure to save your data as ".csv" - not ".csv UTF-8" or any other variants.** After you save your data, reopen your .csv file with a text editor (e.g. notepad) to verify that there are no unusual symbols.

## File Overview

* MCMC_script_writer.py - Generates a configuration file(s) containing settings and initial guesses for the Metropolis sampler. Usage: `python MCMC_script_writer.py`. 

For scripting with a SLURM scheduler or other resource-managed machine, `python MCMC_script_writer.py [ID] [config file header]`, where `ID` is an identifier (e.g. the CPU number) and configuration files will be generated as `[config file header][ID].txt`.

* main.py - Starts a Metropolis sampling run with the settings specified in a configuration file. Usage: `python main.py`.

For scripting with a SLURM scheduler or other resource-managed machine, `python main.py [config file header]`, which will search for configuration files named `[output file header][ID].txt`.

* bayes_io.py - Utilities for generating, reading, and preprocessing configuration files and measurement data files.
* bayes_validate.py - Validation of configuration file settings.
* forward_solver.py - Carrier transport models integrated using scipy solve_ivp().
* laplace.py - Numerical IRF convolution methods.
* mcmc_logging.py - Progress of sampling is written occasionally to a log file.
* metropolis.py - Metropolis algorithm, including trial move proposals, likelihood calculations, and determining when moves are accepted.
* nn_features.py - Interface for neural network surrogate transport models created in tensorflow.
* sim_utils.py - Objects storing results and states visited by Metropolis algorithm.
* utils.py - Standalone utility functions.

* Inputs - Sample measurement data and initial excitation files.
* IRFs - Sample instrument response function files. Each IRF should be a two column [timestamp, intensity] CSV file, similar to measurement data files, named irf_[wavelength].csv
* Tests - Unit tests. Requires the `unittest` and `coverage` packages. Usage: `coverage run -m unittest discover`.
* GUI - Visualization tool for MCMC outputs.

* Dense_Sample - Random sampling algorithm, ported from our previous [Bayesian Inference repository](https://github.com/HagesLab/Bayesian-Inference-TRPL).
* MLE - Maximum likelihood estimator based on scipy minimize(), provided for comparison with MCMC.
* Requirements - Environment specification files.
