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

def cost(x, p, param_info, sim_params):
    """
    Cost function to minimize. As we are seeking a max likelihood,
    this function should return the negative log likelihood.
    """
    cost = 0
    j = 0
    for n in param_info["names"]:
        if param_info["active"][n]:
            setattr(p, n, 10 ** x[j])
            j += 1

    for ic_num in range(sim_params["num_meas"]):
        tSteps, sol = do_simulation(p, thickness, nx, iniPar, times, 1, meas=meas_type,
                                    solver=("solveivp",), model="std")
        
        if LOG_PL:
            sol = np.log10(sol)

        scale_shift = 0
        cost += np.sum((sol + scale_shift - vals_c)**2 / (sim_flags["current_sigma"][meas_type]**2 + 2*uncs_c**2))
    return cost

def mle(sim_params, param_info):
    p = Par()
    p.param_names = param_info["names"]
    p.ucs = param_info["unit_conversions"]

    # Optimize over only active params, while holding all others constant
    x0 = []
    for n in p.param_names:
        if param_info["active"][n]:
            x0.append(np.log10(param_info["init_guess"][n]))


    cost_ = lambda x: cost(x, p, param_info, sim_params)
    cost_(x0)
    return
    opt = least_squares(cost_, x0)
    x = opt.x
    print(x)
    print(opt.cost)
    final_logll = opt.fun * -1
    print(final_logll)