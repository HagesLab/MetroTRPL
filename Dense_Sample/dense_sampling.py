import os
import sys
sys.path.append("..")
import time

import numpy as np

from metropolis import do_simulation

class Par():

    def __init__(self):
        """Dummy class to hold material parameters"""
        self.param_names = None
        self.ucs = None
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

def almost_equal(x, x0, threshold=1e-10):
    if x.shape != x0.shape: return False
    
    return np.abs(np.nanmax((x - x0) / x0)) < threshold

def simulate(model, e_data, P, X, plI, param_info,
             sim_params, init_params, sim_flags, gpu_info, gpu_id, solver_time, err_sq_time, misc_time,
             logger=None):
    """ Delegate blocks of simulation tasks to connected GPUs """
    # has_GPU = gpu_info["has_GPU"]
    # GPU_GROUP_SIZE = gpu_info["sims_per_gpu"]
    # num_gpus = gpu_info["num_gpus"]
    GPU_GROUP_SIZE = sim_flags["num_iters"]
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
    
    thicknesses = sim_params["lengths"]
    nxes = sim_params["nx"]
    p = Par()
    p.param_names = param_info["names"]
    p.ucs = param_info["unit_conversions"]

    for ic_num in range(sim_params["num_meas"]):
        meas_type = sim_params["meas_types"][ic_num]
        times = e_data[0][ic_num]
        # Update thickness
        thickness = thicknesses[ic_num]
        nx = nxes[ic_num]
        for blk in range(gpu_id*GPU_GROUP_SIZE,len(X),num_gpus*GPU_GROUP_SIZE):
            if logger is not None:
                logger.info("Curve #{}: Calculating {} of {}".format(ic_num, blk, len(X)))
            size = min(GPU_GROUP_SIZE, len(X) - blk)

            plI[gpu_id] = np.empty((size, len(times)))
            #assert len(plI[gpu_id][0]) == len(values), "Error: plI size mismatch"

            # if has_GPU:
            #     plN = np.empty((size, 2, sim_params[2]))
            #     plP = np.empty((size, 2, sim_params[2]))
            #     plE = np.empty((size, 2, sim_params[2]+1))
            #     solver_time[gpu_id] += model(plI[gpu_id], plN, plP, plE, X[blk:blk+size, :-1], 
            #                                  sim_params, init_params[ic_num],
            #                                  TPB,8*num_SMs, max_sims_per_block, init_mode="points")
            # else:

            clock0 = time.perf_counter()
            for i, mp in enumerate(X[blk:blk+size]):
                for j, n in enumerate(param_info["names"]):
                    setattr(p, n, mp[j])

                tSteps, sol = model(p, thickness, nx, init_params[ic_num], times, 1,
                                    meas=meas_type, solver=("solveivp",), model="std")
                plI[gpu_id][i] = sol

            solver_time[gpu_id] += time.perf_counter() - clock0

            if LOG_PL:
                # if has_GPU:
                #     misc_time[gpu_id] += fastlog(plI[gpu_id], sys.float_info.min, TPB[0], num_SMs)
                # else:
                clock0 = time.perf_counter()
                plI[gpu_id] = np.log10(plI[gpu_id])
                misc_time[gpu_id] += time.perf_counter() - clock0

            values = e_data[1][ic_num]
            std = e_data[2][ic_num]

            # Calculate errors
            # if has_GPU:
            #     err_sq_time[gpu_id] += prob(P[e, blk:blk+size], plI_int[gpu_id], values, std, np.ascontiguousarray(X[blk:blk+size, -1]), 
            #                                 TPB[0], num_SMs)

            # else:
            clock0 = time.perf_counter()
            P[blk:blk+size] -= np.sum((plI[gpu_id] - values)**2 / (sim_flags["current_sigma"][meas_type]**2 + 2*std**2), axis=1)
            err_sq_time[gpu_id] += time.perf_counter() - clock0
        # END LOOP OVER BLOCKS
    # END LOOP OVER ICs

    return

def bayes(N, P, init_params, sim_params, e_data, sim_flags, param_info, logger=None):
    """ 
    Driver function from Bayesian-Inference-TRPL, made compatible with MetroTRPL
    iniPar acts as init_params, sim_info as sim_params, and MCMC_fields as sim_flags
    """
    num_gpus = 1 # gpu_info["num_gpus"]
    solver_time = np.zeros(num_gpus)
    err_sq_time = np.zeros(num_gpus)
    misc_time = np.zeros(num_gpus)

    min_X = np.array([param_info["prior_dist"][name][0] if param_info['active'][name] else param_info["init_guess"][name]
                      for name in param_info["names"]])
    max_X = np.array([param_info["prior_dist"][name][1] if param_info['active'][name] else param_info["init_guess"][name]
                      for name in param_info["names"]])
    do_log = np.array([param_info["do_log"][name] for name in param_info["names"]])

    N, P, X = make_grid(N, P, min_X, max_X, do_log, sim_flags)
    
    if logger is not None:
        logger.info("Initializing {} random samples".format(len(X)))

    sim_flags["current_sigma"] = dict(sim_flags["annealing"][0])
    sim_params = [dict(sim_params) for i in range(num_gpus)]
    plI = [None for i in range(num_gpus)]
    
    model = do_simulation
    gpu_id = 0
    # Single core control
    simulate(model, e_data, P, X, plI,
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
