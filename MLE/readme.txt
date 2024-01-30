This directory contains a Maximum Likelihood Estimator (MLE) script implemented using scipy.optimize.least_squares, as a contrast to the Metropolis sampling approach.

The usage is the same as the Metropolis sampler - first run MCMC_script_writer.py to generate a script file, and then run run_MLE.py.

However, the following options are treated differently by the MLE sampler:

param_info - "prior_dist" and "trial_move" have no effect on the sampler.
MCMC_fields - 
    "num_iters" has no effect.
    "likel2move_ratio" has no effect.
    "fittable_fluences" and "fittable_absp" have no effect.
    "hard_bounds" - no effect.
    "force_min_y" - no effect.
    Anything involving checkpoints has no effect.
