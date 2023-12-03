import os
import sys
sys.path.append("..")
import time

import numpy as np

from metropolis import do_simulation, search_c_grps
from laplace import make_I_tables, do_irf_convolution, post_conv_trim

class Par():

    def __init__(self):
        """Dummy class to hold material parameters"""
        self.param_names = dict()
        self.ucs = dict()
        return

    def apply_unit_conversions(self, reverse=False):
        """ Multiply the currently stored parameters according to a stored
            unit conversion dictionary.
        """
        for param in self.param_names:
            val = getattr(self, param)
            if reverse:
                setattr(self, param, val / self.ucs.get(param, 1))
            else:
                setattr(self, param, val * self.ucs.get(param, 1))
        return

def random_grid(min_X, max_X, do_log, num_samples):
    """ Draw [num_samples] random points from hyperspace bounded by [min_X], [max_X] """
    num_params = len(min_X)
    grid = np.empty((num_samples, num_params))
    
    for i in range(num_params):
        if min_X[i] == max_X[i]:
            grid[:,i] = min_X[i]
        else:
            if do_log[i]:
                grid[:,i] = 10 ** np.random.uniform(np.log10(min_X[i]), np.log10(max_X[i]), (num_samples,))
            else:
                grid[:,i] = np.random.uniform(min_X[i], max_X[i], (num_samples,))
            
    return grid

def make_grid(N, P, min_X, max_X, do_log, sim_flags):
    """ Set up sampling grid - random sample """

    num_samples = sim_flags["num_iters"]

    N = np.arange(num_samples)
    X = random_grid(min_X, max_X, do_log, num_samples)

    # Likelihoods
    P = np.zeros(len(N))

    return N, P, X

def simulate(model, e_data, P, X, param_info,
             sim_params, init_params, sim_flags, gpu_info, gpu_id, solver_time, err_sq_time, misc_time,
             logger=None):
    """ Delegate blocks of simulation tasks to connected GPUs """
    # has_GPU = gpu_info["has_GPU"]
    # GPU_GROUP_SIZE = gpu_info["sims_per_gpu"]
    # num_gpus = gpu_info["num_gpus"]
    GPU_GROUP_SIZE = 1000
    num_gpus = 1
    
    # if has_GPU:
    #     TPB = gpu_info["threads_per_block"]
    #     max_sims_per_block = gpu_info["max_sims_per_block"]
        
    #     try:
    #         cuda.select_device(gpu_id)
    #     except IndexError:
    #         if logger is not None:
    #             logger.error("Error: threads failed to launch")
    #         return
    #     device = cuda.get_current_device()
    #     num_SMs = getattr(device, "MULTIPROCESSOR_COUNT")
    
    LOG_PL = sim_flags["log_pl"]
    scale_f_info = sim_flags.get("scale_factor", None)
    where_sfs = {s_name:i for i, s_name in enumerate(param_info["names"]) if s_name.startswith("_s")}
    irf_convolution = sim_flags.get("irf_convolution", None)
    IRF_tables = sim_flags.get("IRF_tables", None)
    thicknesses = sim_params["lengths"]
    nxes = sim_params["nx"]
    p = Par()
    p.param_names = param_info["names"]
    p.ucs = param_info["unit_conversions"]

    for ic_num in range(sim_params["num_meas"]):
        meas_type = sim_params["meas_types"][ic_num]
        times = e_data[0][ic_num]
        values = e_data[1][ic_num]
        std = e_data[2][ic_num]

        thickness = thicknesses[ic_num]
        nx = nxes[ic_num]

        for i, mp in enumerate(X):
            if logger is not None and i % 1000 == 0:
                logger.info("Curve #{}: Calculating {} of {}".format(ic_num, i, len(X)))

            for j, n in enumerate(param_info["names"]):
                setattr(p, n, mp[j])
            clock0 = time.perf_counter()
            tSteps, sol = model(p, thickness, nx, init_params[ic_num], times, 1,
                                meas=meas_type, solver=("solveivp",), model="std")
            solver_time[gpu_id] += time.perf_counter() - clock0
            
            clock0 = time.perf_counter()
            if irf_convolution is not None and irf_convolution[ic_num] != 0:
                wave = int(irf_convolution[ic_num])
                tSteps, sol, success = do_irf_convolution(
                    tSteps, sol, IRF_tables[wave], time_max_shift=True)
                if not success:
                    raise ValueError("Error: Interpolation for conv failed. Check measurement data"
                                    " times for floating-point inaccuracies.")
                sol, times_c, vals_c, uncs_c = post_conv_trim(tSteps, sol, times, values, std)

            else:
                # Still need to trim, in case experimental data doesn't start at t=0
                times_c = times
                vals_c = values
                uncs_c = std
                sol = sol[-len(times_c):]
            misc_time[gpu_id] += time.perf_counter() - clock0

            where_failed = sol < 0
            n_fails = np.sum(where_failed)

            if n_fails > 0 and logger is not None:
                logger.warning(f"{i}: {n_fails} / {len(sol)} non-positive vals")

            sol[where_failed] *= -1

            # Convolution has to be done per-sample. Otherwise the simulations could be postprocessed
            # in blocks.
            if LOG_PL:
                # if has_GPU:
                #     misc_time[gpu_id] += fastlog(plI[gpu_id], sys.float_info.min, TPB[0], num_SMs)
                # else:
                clock0 = time.perf_counter()
                sol = np.log10(sol)
                misc_time[gpu_id] += time.perf_counter() - clock0

            if (scale_f_info is not None and ic_num in scale_f_info[1]):
                if scale_f_info[2] is not None and len(scale_f_info[2]) > 0:
                    s_name = f"_s{search_c_grps(scale_f_info[2], ic_num)}"
                else:
                    s_name = f"_s{ic_num}"
                scale_shift = np.log10(X[i, where_sfs[s_name]])
            else:
                scale_shift = 0

            # Calculate errors
            # if has_GPU:
            #     err_sq_time[gpu_id] += prob(P[e, blk:blk+size], plI_int[gpu_id], values, std, np.ascontiguousarray(X[blk:blk+size, -1]), 
            #                                 TPB[0], num_SMs)

            # else:
            clock0 = time.perf_counter()
            P[i] -= np.sum((sol + scale_shift - vals_c)**2 / (sim_flags["current_sigma"][meas_type]**2 + 2*uncs_c**2))
            err_sq_time[gpu_id] += time.perf_counter() - clock0
        # END LOOP OVER BLOCKS
    # END LOOP OVER ICs

def modify_scale_factors(param_info, sim_flags):
    """Replace the (0, inf) default bounds for scale factors with their init_guess * or / their trial move size"""
    spread = sim_flags["scale_factor"][0]
    for name in param_info["names"]:
        if name.startswith("_s"):
            param_info["prior_dist"][name] = (param_info["init_guess"][name] / spread, param_info["init_guess"][name] * spread)

def bayes(N, P, init_params, sim_params, e_data, sim_flags, param_info, logger=None):
    """
    Driver function from Bayesian-Inference-TRPL, made compatible with MetroTRPL
    iniPar acts as init_params, sim_info as sim_params, and MCMC_fields as sim_flags
    """
    num_gpus = 1 # gpu_info["num_gpus"]
    solver_time = np.zeros(num_gpus)
    err_sq_time = np.zeros(num_gpus)
    misc_time = np.zeros(num_gpus)

    if "scale_factor" in sim_flags:
    	modify_scale_factors(param_info, sim_flags)

    min_X = np.array([param_info["prior_dist"][name][0] if param_info['active'][name] else param_info["init_guess"][name]
                      for name in param_info["names"]])
    max_X = np.array([param_info["prior_dist"][name][1] if param_info['active'][name] else param_info["init_guess"][name]
                      for name in param_info["names"]])
    do_log = np.array([param_info["do_log"][name] for name in param_info["names"]])

    N, P, X = make_grid(N, P, min_X, max_X, do_log, sim_flags)
    
    if logger is not None:
        logger.info("Initializing {} random samples".format(len(X)))
        logger.info(f"First three samples: {X[0:3]}")

    sim_flags["current_sigma"] = dict(sim_flags["annealing"][0])
    sim_params = [dict(sim_params) for i in range(num_gpus)]

    if sim_flags.get("irf_convolution", None) is not None:
        irfs = {}
        for i in sim_flags["irf_convolution"]:
            if i > 0 and i not in irfs:
                irfs[int(i)] = np.loadtxt(os.path.join("IRFs", f"irf_{int(i)}nm.csv"),
                                            delimiter=",")

        sim_flags["IRF_tables"] = make_I_tables(irfs)
        if logger is not None:
            logger.info(f"Found IRFs for WLs {list(sim_flags['IRF_tables'].keys())}")
    else:
        sim_flags["IRF_tables"] = None
    
    model = do_simulation
    gpu_id = 0
    # Single core control
    simulate(model, e_data, P, X,
             param_info, sim_params[gpu_id], init_params, sim_flags, {}, gpu_id,
             solver_time, err_sq_time, misc_time, logger=logger)
    # Multi-GPU thread control
    # threads = []
    #for gpu_id in range(num_gpus):
    #    logger.info("Starting thread {}".format(gpu_id))
    #    thread = threading.Thread(target=simulate, args=(model, e_data, P, X, plI, plI_int,
    #                              num_curves,sim_params[gpu_id], init_params, sim_flags, gpu_info, gpu_id,
    #                              solver_time, err_sq_time, misc_time, logger))
    #    threads.append(thread)
    #    thread.start()

    #for gpu_id, thread in enumerate(threads):
    #    logger.info("Ending thread {}".format(gpu_id))
    #    thread.join()
    #    logger.info("Thread {} closed".format(gpu_id))

    if logger is not None:
        logger.info("Total tEvol time: {}, avg {}".format(solver_time, np.mean(solver_time)))
        logger.info("Total err_sq time (temperatures and mag_offsets): {}, avg {}".format(err_sq_time, np.mean(err_sq_time)))
        logger.info("Total misc time: {}, avg {}".format(misc_time, np.mean(misc_time)))
    return N, P, X

def export(out_filename, P, X, logger=None):
    """ Export list of likelihoods (*_BAYRAN_P.npy) and sample parameter points (*_BAYRAN_X.npy) """
    head = os.path.dirname(out_filename)
    base = os.path.basename(out_filename)

    if logger is not None:
        logger.info(f"Creating dir {head}")
    os.makedirs(head, exist_ok=True)

    if logger is not None: 
        logger.info(f"Writing to {out_filename}:")
    np.save(os.path.join(head, f"{base}_P.npy"), P)
    np.save(os.path.join(head, f"{base}_X.npy"), X)
