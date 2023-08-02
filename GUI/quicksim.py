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

    def __init__(self, tk_gui, queue):
        self.tk_gui = tk_gui # A Window object, from (currently) main.py
        self.queue = queue

    def quicksim(self):
        """Regenerate a simulation using a selected state"""
        # Currently this just uses the last state from the first chain
        fname = next(iter(self.tk_gui.file_names.keys()))
        param_info = {}
        param_info["names"] = [x for x in self.tk_gui.data[fname] if x not in self.tk_gui.sp.func]
        param_info["init_guess"] = {x: self.tk_gui.data[fname][x][True][-1] for x in param_info["names"]}
        param_info["active"] = {x: True for x in self.tk_gui.data[fname] if x not in self.tk_gui.sp.func}
        param_info["unit_conversions"] = {"n0": ((1e-7) ** 3), "p0": ((1e-7) ** 3),
                        "mu_n": ((1e7) ** 2) / (1e9),
                        "mu_p": ((1e7) ** 2) / (1e9),
                        "ks": ((1e7) ** 3) / (1e9),
                        "Cn": ((1e7) ** 6) / (1e9),
                        "Cp": ((1e7) ** 6) / (1e9),
                        "Sf": 1e-2, "Sb": 1e-2}
        self.tk_gui.status("Simulating")
        p = sim_utils.Parameters(param_info)
        thickness = 2000
        nx = 128
        iniPar = (1e13, 6e4)
        t_sim = np.linspace(0, 2000, 8000)
        simulate = partial(do_simulation, p, thickness, nx, iniPar, t_sim,
                           hmax=4, meas="TRPL", solver=("solveivp",))

        self.proc = multiprocessing.Process(target=qs_simulate, args=(self.queue, simulate))
        self.proc.start()

    def join(self):
        """Terminate quicksim process"""
        if self.proc.is_alive():
            self.proc.join()

def qs_simulate(queue, task) -> None:
    """
    Do a quicksim task and put in queue.
    This cannot be a GUI method - as the GUI instance is not pickleable.
    """
    t, sol = task()
    queue.put((t, sol))
