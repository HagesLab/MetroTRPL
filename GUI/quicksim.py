"""
For handling all calculations done by the quicksim feature, which
recreates the simulation seen by MMC at a specific state.
"""
import multiprocessing
from functools import partial
import numpy as np
from metropolis import do_simulation
import sim_utils

class QuicksimManager():
    proc : multiprocessing.Process
    queue : multiprocessing.Queue

    def __init__(self, window, queue):
        self.window = window # A Window object, from (currently) main.py
        self.queue = queue

    def quicksim(self, sim_tasks):
        """
        Regenerate simulations using a selected state.

        Simulations require info that isn't necessarily part of the state - 
        such as fluence, thickness, and absorption coefficient.

        These should be specified through the dict sim_tasks, in which
        each key is an external parameter and each value is a list
        with one value per desired simulation.

        Multiple simulations (e.g. multiple TRPL curves) may be
        generated from a state based on the length of values in sim_tasks.
        """
        simulate = []
        for fname in self.window.file_names:
            if self.window.file_names[fname].get() == 0: # Don't simulate disabled chains
                continue
            param_info = {}
            param_info["names"] = [x for x in self.window.data[fname] if x not in self.window.sp.func]
            param_info["init_guess"] = {x: self.window.data[fname][x][True][-1] for x in param_info["names"]}
            param_info["active"] = {x: True for x in param_info["names"]}
            param_info["unit_conversions"] = {"n0": ((1e-7) ** 3), "p0": ((1e-7) ** 3),
                            "mu_n": ((1e7) ** 2) / (1e9),
                            "mu_p": ((1e7) ** 2) / (1e9),
                            "ks": ((1e7) ** 3) / (1e9),
                            "Cn": ((1e7) ** 6) / (1e9),
                            "Cp": ((1e7) ** 6) / (1e9),
                            "Sf": 1e-2, "Sb": 1e-2}
            p = sim_utils.Parameters(param_info)

            thickness = sim_tasks["thickness"]
            n_sims = len(thickness)
            nx = sim_tasks["nx"]
            iniPar = list(zip(sim_tasks["fluence"], sim_tasks["absp"], sim_tasks["direction"]))
            t_sim = [np.linspace(0, sim_tasks["final_time"][i], sim_tasks["nt"][i] + 1) for i in range(n_sims)]
            simulate += [partial(do_simulation, p, thickness[i], nx[i], iniPar[i], t_sim[i],
                                 hmax=4, meas="TRPL", solver=("solveivp",)) for i in range(n_sims)]

        self.proc = multiprocessing.Process(target=qs_simulate, args=(self.queue, simulate))
        self.proc.start()

    def join(self):
        """Terminate quicksim process"""
        if self.proc.is_alive():
            self.proc.join()

def qs_simulate(queue, tasks) -> None:
    """
    Do quicksim tasks and put in queue.
    A task is any simulation call that returns two arrays (e.g. delay times and signal)
    This cannot be a GUI method - as the GUI instance is not pickleable.
    """
    for task in tasks:
        t, sol = task()
        queue.put((t, sol))
