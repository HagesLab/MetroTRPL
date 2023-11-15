import os
import numpy as np
from scipy.optimize import least_squares

from metropolis import do_simulation
from sim_utils import MetroState

DEFAULT_NUM_ITERS = 1000
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

def cost(x, e_data, MS, logger):
    """
    Cost function to minimize. As we are seeking a max likelihood,
    this function should return the negative log likelihood.
    """
    _cost = 0
    j = 0
    for n in MS.param_info["names"]:
        if MS.param_info["active"][n]:
            setattr(MS.means, n, 10 ** x[j])
            j += 1

    logger.info("#####")
    logger.info("Iter #")

    LOG_PL = MS.MCMC_fields["log_pl"]
    thicknesses = MS.sim_info["lengths"]
    nxes = MS.sim_info["nx"]
    for ic_num in range(MS.sim_info["num_meas"]):
        thickness = thicknesses[ic_num]
        nx = nxes[ic_num]
        meas_type = MS.sim_info["meas_types"][ic_num]
        times = e_data[0][ic_num]
        values = e_data[1][ic_num]
        std = e_data[2][ic_num]

        tSteps, sol = do_simulation(MS.means, thickness, nx, MS.iniPar[ic_num], times, 1, meas=meas_type,
                                    solver=("solveivp",), model="std")

        times_c = times
        vals_c = values
        uncs_c = std
        sol = sol[-len(times_c):]

        if LOG_PL:
            sol = np.log10(sol)

        scale_shift = 0
        _cost += np.sum((sol + scale_shift - vals_c)**2 / (MS.MCMC_fields["current_sigma"][meas_type]**2 + 2*uncs_c**2))

    current_num_iters = len(MS.H.accept[0])
    if MS.latest_iter > current_num_iters:
        MS.H.extend(2 * current_num_iters, MS.param_info)
    MS.H.update(MS.latest_iter, MS.means, MS.means, MS.param_info)
    MS.H.loglikelihood[0, MS.latest_iter] = _cost

    for n in MS.param_info['names']:
        if MS.param_info["active"][n]:
            mean = getattr(MS.means, n)
            logger.info("Current {}: {:.6e}".format(n, mean))
    logger.info(f"Iter {MS.latest_iter} Cost: {_cost}")
    logger.info("#####")

    MS.latest_iter += 1
    return _cost

def mle(e_data, sim_params, param_info, init_params, sim_flags, export_path, logger):
    MS = MetroState(param_info, sim_flags, DEFAULT_NUM_ITERS)

    # Not needed for MLE
    del MS.p
    del MS.prev_p
    del MS.variances

    # Prefer having these attached to MS, to match the original MCMC method
    MS.sim_info = sim_params
    MS.iniPar = init_params
    logger.info(f"Sim info: {MS.sim_info}")
    logger.info(f"Param infos: {MS.param_info}")
    logger.info(f"MCMC fields: {MS.MCMC_fields}")

    # Optimize over only active params, while holding all others constant
    x0 = []
    for n in MS.means.param_names:
        if param_info["active"][n]:
            x0.append(np.log10(param_info["init_guess"][n]))
        else:
            setattr(MS.means, n, param_info["init_guess"][n])


    cost_ = lambda x: cost(x, e_data, MS, logger)
    opt = least_squares(cost_, x0, loss="soft_l1")
    x = opt.x
    print(10 ** x)
    final_logll = opt.fun * -1
    print(final_logll)
    print(opt.message)

    current_num_iters = len(MS.H.accept[0])
    if MS.latest_iter < current_num_iters:
        MS.H.truncate(MS.latest_iter, MS.param_info)
    if export_path is not None:
        logger.info(f"Exporting to {MS.MCMC_fields['output_path']}")
        MS.checkpoint(os.path.join(MS.MCMC_fields["output_path"], export_path))

    return MS