# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:04:20 2022

@author: cfai2
"""
import numpy as np
from scipy.integrate import trapz
from matplotlib import pyplot as plt
import os
# Constants
eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
q_C = 1.602e-19 # [C per carrier]

class Parameters():
    Sf : float      # Front surface recombination velocity
    Sb : float      # Back surface recombination velocity
    mu_n : float    # Electron mobility
    mu_p : float    # Hole mobility
    n0 : float      # Electron doping level
    p0 : float      # Hole doping level
    B : float       # Radiative recombination rate
    tauN : float    # Electron bulk nonradiative decay lifetime
    tauP : float    # Hole bulk nonradiative decayl lifetime
    eps : float     # Relative dielectric cofficient
    Tm : float      # Temperature
    def __init__(self, param_info, initial_guesses):
        param_names = param_info["names"]
        
        for param in param_names:
            setattr(self, param, initial_guesses[param])
        self.Tm = 300
        return
    
    def apply_unit_conversions(self, param_info):
        for param in param_info["names"]:
            val = getattr(self, param)
            setattr(self, param, val * param_info["unit_conversions"].get(param, 1))
        
    def make_log(self, param_info):
        for param in param_info["names"]:
            if param_info["do_log"].get(param, 0) and hasattr(self, param):
                val = getattr(self, param)
                setattr(self, param, np.log10(val))
                        
class History():
    
    def __init__(self, num_iters, param_info):
        
        for param in param_info["names"]:
            setattr(self, param, np.zeros(num_iters))
            setattr(self, f"mean_{param}", np.zeros(num_iters))
        
        self.accept = np.zeros(num_iters)
        self.ratio = np.zeros(num_iters)
    
    def apply_unit_conversions(self, param_info):
        for param in param_info["names"]:
            val = getattr(self, param)
            setattr(self, param, val / param_info["unit_conversions"].get(param, 1))
            
            val = getattr(self, f"mean_{param}")
            setattr(self, f"mean_{param}", val / param_info["unit_conversions"].get(param, 1))
            
    def export(self, param_info, out_pathname):
        for param in param_info["names"]:
            np.save(os.path.join(out_pathname, f"{param}"), getattr(self, param))
            np.save(os.path.join(out_pathname, f"mean_{param}"), getattr(self, f"mean_{param}"))
                    
        np.save(os.path.join(out_pathname, "accept"), self.accept)
        
    def truncate(self, k, param_info):
        for param in param_info["names"]:
            val = getattr(self, param)
            setattr(self, param, val[:k])
            
            val = getattr(self, f"mean_{param}")
            setattr(self, f"mean_{param}", val[:k])
            
        self.accept = self.accept[:k]
        self.ratio = self.ratio[:k]
        return
        
class HistoryList():
    
    def __init__(self, list_of_histories, param_info):
        self.join("accept", list_of_histories)
        self.join("ratio", list_of_histories)
        for param in param_info["names"]:
            self.join(param, list_of_histories)
            self.join(f"mean_{param}", list_of_histories)
        
        
    def join(self, attr, list_of_histories):
        attr_from_each_hist = [getattr(H, attr) for H in list_of_histories]
        setattr(self, attr, np.vstack(attr_from_each_hist))
        
    def export(self, param_info, out_pathname):
        for param in param_info["names"]:
            np.save(os.path.join(out_pathname, f"{param}"), getattr(self, param))
            np.save(os.path.join(out_pathname, f"mean_{param}"), getattr(self, f"mean_{param}"))
                    
        np.save(os.path.join(out_pathname, "accept"), self.accept)
        
        
class Grid():
    def __init__(self):
        return
    
class Solution():
    def __init__(self):
        return
    
    def convert_to_cms(self):
        self.N /= ((1e-7) ** 3) # [nm^-3] to [cm^-3]
        self.P /= ((1e-7) ** 3) # [nm^-3] to [cm^-3]
        self.PL /= ((1e-7) ** 3) / (1e9) * 1e7 # [nm^-2 ns^-1] to [cm^-2 s^-1]
        
    def calculate_PL(self, g, p):
        rr = p.B * (self.N * self.P - p.n0*p.p0)
        
        self.PL = trapz(rr, dx=g.dx, axis=1)
        self.PL += rr[:, 0] * g.dx / 2
        self.PL += rr[:, -1] * g.dx / 2
        
    def plot_PL(self, g):
        fig, ax = plt.subplots(1,2,figsize=(8,4), dpi=120)
        
        ax[0].plot((g.tSteps[1:]), (self.PL[1:]))
        
        ax[0].set_yscale('log')
        ax[0].set_ylabel("PL [cm$^{-2}$ s$^{-1}$]")
        ax[0].set_xlabel("delay time [ns]")
        
        ax[1].plot(np.log10(g.tSteps[1:]), np.log10(self.PL[1:]))
        
        #ax.set_yscale('log')
        ax[1].set_ylabel("log PL [cm$^{-2}$ s$^{-1}$]")
        ax[1].set_xlabel("log delay time [ns]")
        
        fig.tight_layout()
        
        return
    
class InitialConditions():
    power_density : float
    absorption : float
    wavelength : float
    pulse_freq : float
    
    def __init__(self):
        h = 6.626e-34   # [J*s]
        c = 2.997e8     # [m/s]
        self.hc = h * c * 1e9 # [J*nm]
        return
    
    def apply_unit_conversions(self):
        self.power_density *= 1e-6 * ((1e-7) ** 2)  # [uW / cm^2] to [W nm^2]
        self.absorption *= 1e-7 # [cm^-1] to [nm^-1]
        self.pulse_freq *= 1e3    # [kHz] to [1/s]
        return
    
    def pulse_laser_powerdensity(self, x_array):
        return (self.power_density / (self.pulse_freq * self.hc / self.wavelength) * 
                self.absorption * np.exp(-self.absorption * x_array))
        
    def calc_E_field(self, g, p):
        """Calculate electric field from N, P"""
        
        dEdx = q_C * (self.dP - self.dN) / (eps0 * p.eps)
        if dEdx.ndim == 1:
            E_field = np.concatenate(([0], np.cumsum(dEdx) * g.dx)) #[V/nm]
       
        return E_field
        
    def generate_initial_conditions(self, g, p, x_array):
        self.dN = self.pulse_laser_powerdensity(x_array)
        self.dP = np.array(self.dN)
        self.E_field = self.calc_E_field(g, p)
        
        self.N = self.dN + p.n0
        self.P = self.dP + p.p0
        
        self.data_splits = [g.nx, 2*g.nx]
        self.init_condition = np.concatenate([self.N, self.P, self.E_field], axis=None)
        
        return