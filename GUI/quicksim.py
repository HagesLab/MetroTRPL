"""
For handling all calculations done by the quicksim feature, which
recreates the simulation seen by MMC at a specific state.
"""
import multiprocessing
import os
from functools import partial
import numpy as np
from scipy.interpolate import griddata
from metropolis import do_simulation
from laplace import make_I_tables, do_irf_convolution
import sim_utils

IRF_PATH = os.path.join("..", "IRFs")

class QuicksimManager():
    proc : multiprocessing.Process
    queue : multiprocessing.Queue

    def __init__(self, window, queue):
        self.window = window # A Window object, from (currently) main.py
        self.queue = queue

    def quicksim(self, sim_tasks, model, meas):
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
        irfs = {}
        missing_irfs = []
        for i in sim_tasks["wavelength"]:
            if i > 0 and i not in irfs:
                try:
                    irfs[int(i)] = np.loadtxt(os.path.join(IRF_PATH, "irf_{}nm.csv".format(int(i))),
                                                delimiter=",")
                except FileNotFoundError:
                    if i not in missing_irfs:
                        missing_irfs.append(i)
                        self.window.status(f"Warning: no IRF for wavelength {i}")
                    continue

        if len(irfs) > 0:
            IRF_tables = make_I_tables(irfs)
        else:
            self.window.status("Warning: no IRFs found")
            IRF_tables = dict()

        simulate = []
        for chain in self.window.chains:
            if chain.visible.get() == 0: # Don't simulate disabled chains
                continue
            param_info = {}
            param_info["names"] = [x for x in self.window.data[chain.fname] if x not in self.window.sp.func]
            param_info["init_guess"] = {x: self.window.data[chain.fname][x][-1] for x in param_info["names"]}
            param_info["active"] = {x: True for x in param_info["names"]}
            param_info["unit_conversions"] = {"n0": ((1e-7) ** 3), "p0": ((1e-7) ** 3),
                            "mu_n": ((1e7) ** 2) / (1e9),
                            "mu_p": ((1e7) ** 2) / (1e9),
                            "ks": ((1e7) ** 3) / (1e9),
                            "Cn": ((1e7) ** 6) / (1e9),
                            "Cp": ((1e7) ** 6) / (1e9),
                            "Sf": 1e-2, "Sb": 1e-2, 
                            "kC": ((1e7) ** 3) / (1e9),
                            "Nt": ((1e-7) ** 3)}
            p = sim_utils.Parameters(param_info)

            thickness = sim_tasks["thickness"]
            wavelength = sim_tasks["wavelength"]
            n_sims = len(thickness)
            nx = sim_tasks["nx"]
            iniPar = list(zip(sim_tasks["fluence"], sim_tasks["absp"], sim_tasks["direction"]))
            t_sim = [np.linspace(0, sim_tasks["final_time"][i], sim_tasks["nt"][i] + 1) for i in range(n_sims)]
            simulate += [partial(task, p, thickness[i], nx[i], iniPar[i], t_sim[i],
                                 hmax=4, meas=meas, solver=("solveivp",), model=model,
                                 wavelength=wavelength[i], IRF_tables=IRF_tables) for i in range(n_sims)]

        self.proc = multiprocessing.Process(target=qs_simulate, args=(self.queue, simulate))
        self.proc.start()

    def join(self):
        """Join quicksim process"""
        if self.proc.is_alive():
            self.proc.join()

    def terminate(self):
        """Abort quicksim process"""
        self.proc.terminate()

def task(p, thickness, nx, iniPar, times, hmax, meas, solver, model, wavelength, IRF_tables):
    """What each task needs to do - simulate then optionally convolve"""
    t, sol = do_simulation(p, thickness, nx, iniPar, times, hmax, meas, solver, model)
    if wavelength != 0 and int(wavelength) in IRF_tables:
        t, sol, success = do_irf_convolution(
            t, sol, IRF_tables[int(wavelength)], time_max_shift=True)
        if not success:
            raise ValueError("Error: Interpolation for conv failed. Check measurement data"
                             " times for floating-point inaccuracies.")
        
        conv_cutoff = np.where(times < np.nanmax(t))[0][-1]
        sol = griddata(t, sol, times[:conv_cutoff+1])
        t = times[:conv_cutoff+1]
    return t, sol

def qs_simulate(queue, tasks) -> None:
    """
    Do quicksim tasks and put in queue.
    A task is any simulation call that returns two arrays (e.g. delay times and signal)
    This cannot be a GUI method - as the GUI instance is not pickleable.
    """
    for i, task_f in enumerate(tasks):
        try:
            t, sol = task_f()
            message = ""
        except AttributeError:
            message = f"Warning: simulation {i} failed - possibly wrong model?"
            t = np.zeros(0)
            sol = np.zeros(0)
        queue.put((t, sol, message))
