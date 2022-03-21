# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:04:20 2022

@author: cfai2
"""
import numpy as np
from scipy.integrate import trapz
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
            if hasattr(self, param): raise KeyError(f"Param with name {param} already exists")
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

    def calculate_PL(self, g, p):
        rr = p.B * (self.N * self.P - p.n0*p.p0)
        
        self.PL = trapz(rr, dx=g.dx, axis=1)
        self.PL += rr[:, 0] * g.dx / 2
        self.PL += rr[:, -1] * g.dx / 2
