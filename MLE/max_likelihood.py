import numpy as np
from scipy.optimize import least_squares

from metropolis import do_simulation

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

def cost(x, e_data, p, param_info, sim_params, init_params, sim_flags, logger):
    """
    Cost function to minimize. As we are seeking a max likelihood,
    this function should return the negative log likelihood.
    """
    _cost = 0
    j = 0
    for n in param_info["names"]:
        if param_info["active"][n]:
            setattr(p, n, 10 ** x[j])
            j += 1

    logger.info("#####")
    logger.info("Iter #")

    LOG_PL = sim_flags["log_pl"]
    thicknesses = sim_params["lengths"]
    nxes = sim_params["nx"]
    for ic_num in range(sim_params["num_meas"]):
        thickness = thicknesses[ic_num]
        nx = nxes[ic_num]
        meas_type = sim_params["meas_types"][ic_num]
        times = e_data[0][ic_num]
        values = e_data[1][ic_num]
        std = e_data[2][ic_num]

        tSteps, sol = do_simulation(p, thickness, nx, init_params[ic_num], times, 1, meas=meas_type,
                                    solver=("solveivp",), model="std")

        times_c = times
        vals_c = values
        uncs_c = std
        sol = sol[-len(times_c):]

        if LOG_PL:
            sol = np.log10(sol)

        scale_shift = 0
        _cost += np.sum((sol + scale_shift - vals_c)**2 / (sim_flags["current_sigma"][meas_type]**2 + 2*uncs_c**2))

    logger.info(f"Cost: {_cost}")
    logger.info("#####")
    return _cost

def mle(e_data, sim_params, param_info, init_params, sim_flags, logger):
    p = Par()
    p.param_names = param_info["names"]
    p.ucs = param_info["unit_conversions"]
    sim_flags["current_sigma"] = dict(sim_flags["annealing"][0])

    # Optimize over only active params, while holding all others constant
    x0 = []
    for n in p.param_names:
        if param_info["active"][n]:
            x0.append(np.log10(param_info["init_guess"][n]))
        else:
            setattr(p, n, param_info["init_guess"][n])


    cost_ = lambda x: cost(x, e_data, p, param_info, sim_params, init_params, sim_flags, logger)
    opt = least_squares(cost_, x0)
    x = opt.x
    print(10 ** x)
    final_logll = opt.fun * -1
    print(final_logll)
    print(opt.message)