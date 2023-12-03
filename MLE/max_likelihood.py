import os
import numpy as np
from scipy.optimize import minimize

from metropolis import do_simulation, search_c_grps
from laplace import make_I_tables, do_irf_convolution, post_conv_trim
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
    scale_f_info = MS.MCMC_fields.get("scale_factor", None)
    irf_convolution = MS.MCMC_fields.get("irf_convolution", None)
    IRF_tables = MS.MCMC_fields.get("IRF_tables", None)
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
                                    solver=MS.MCMC_fields["solver"], model=MS.MCMC_fields["model"])

        if irf_convolution is not None and irf_convolution[ic_num] != 0:
            wave = int(irf_convolution[ic_num])
            tSteps, sol, success = do_irf_convolution(
                tSteps, sol, IRF_tables[wave], time_max_shift=True)
            if not success:
                raise ValueError("Error: Interpolation for conv failed. Check measurement data"
                                " times for floating-point inaccuracies.")
            sol, times_c, vals_c, uncs_c = post_conv_trim(tSteps, sol, times, values, std)
        else:
            times_c = times
            vals_c = values
            uncs_c = std
            sol = sol[-len(times_c):]

        where_failed = sol < 0
        n_fails = np.sum(where_failed)

        if n_fails > 0 and logger is not None:
            logger.warning(f"{MS.latest_iter}: {n_fails} / {len(sol)} non-positive vals")

        sol[where_failed] *= -1


        if LOG_PL:
            sol = np.log10(sol)

        if (scale_f_info is not None and ic_num in scale_f_info[1]):
            if scale_f_info[2] is not None and len(scale_f_info[2]) > 0:
                s_name = f"_s{search_c_grps(scale_f_info[2], ic_num)}"
            else:
                s_name = f"_s{ic_num}"
            scale_shift = np.log10(getattr(MS.means, s_name))
        else:
            scale_shift = 0

        _cost += np.sum((sol + scale_shift - vals_c)**2 / (MS.MCMC_fields["current_sigma"][meas_type]**2 + 2*uncs_c**2))

    current_num_iters = len(MS.H.accept[0])
    if MS.latest_iter >= current_num_iters:
        MS.H.extend(2 * current_num_iters, MS.param_info)
    MS.H.update(MS.latest_iter, MS.means, MS.means, MS.param_info)
    MS.H.loglikelihood[0, MS.latest_iter] = _cost * -1

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

    # Prefer having these attached to MS, to match the original MCMC method
    MS.sim_info = sim_params
    MS.iniPar = init_params
    logger.info(f"Sim info: {MS.sim_info}")
    logger.info(f"Param infos: {MS.param_info}")
    logger.info(f"MCMC fields: {MS.MCMC_fields}")

    if MS.MCMC_fields.get("irf_convolution", None) is not None:
        irfs = {}
        for i in MS.MCMC_fields["irf_convolution"]:
            if i > 0 and i not in irfs:
                irfs[int(i)] = np.loadtxt(os.path.join("IRFs", f"irf_{int(i)}nm.csv"),
                                            delimiter=",")

        MS.MCMC_fields["IRF_tables"] = make_I_tables(irfs)
        if logger is not None:
            logger.info(f"Found IRFs for WLs {list(MS.MCMC_fields['IRF_tables'].keys())}")
    else:
        MS.MCMC_fields["IRF_tables"] = None

    # Optimize over only active params, while holding all others constant
    x0 = []
    for n in MS.means.param_names:
        if param_info["active"][n]:
            x0.append(np.log10(param_info["init_guess"][n]))
        else:
            setattr(MS.means, n, param_info["init_guess"][n])


    cost_ = lambda x: cost(x, e_data, MS, logger)
    opt = minimize(cost_, x0, method="Nelder-Mead")
    x = opt.x
    logger.info(10 ** x)
    final_logll = opt.fun * -1
    logger.info(final_logll)
    logger.info(opt.message)

    current_num_iters = len(MS.H.accept[0])
    if MS.latest_iter < current_num_iters:
        MS.H.truncate(MS.latest_iter, MS.param_info)
    if export_path is not None:
        logger.info(f"Exporting to {MS.MCMC_fields['output_path']}")
        MS.checkpoint(os.path.join(MS.MCMC_fields["output_path"], export_path))

    return MS
