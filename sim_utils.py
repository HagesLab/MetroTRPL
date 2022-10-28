# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:04:20 2022

@author: cfai2
"""
import numpy as np
from scipy.integrate import trapz
import os
import pickle
from numba import njit
# Constants
eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
q_C = 1.602e-19 # [C per carrier]

class MetroState():
    
    def __init__(self, param_info, initial_guess, initial_variance, num_iters):
        self.p = Parameters(param_info, initial_guess)
        self.p.apply_unit_conversions(param_info)
        
        self.H = History(num_iters, param_info)
        
        self.prev_p = Parameters(param_info, initial_guess)
        self.prev_p.apply_unit_conversions(param_info)
        
        self.means = Parameters(param_info, initial_guess)
        self.means.apply_unit_conversions(param_info)
        
        self.variances = Covariance(param_info)
        self.variances.apply_values(initial_variance)
        return
    
    def checkpoint(self, fname):
        with open(fname, "wb+") as ofstream:
            pickle.dump(self, ofstream)
        return

class Parameters():
    Sf : float      # Front surface recombination velocity
    Sb : float      # Back surface recombination velocity
    mu_n : float    # Electron mobility
    mu_p : float    # Hole mobility
    n0 : float      # Electron doping level
    p0 : float      # Hole doping level
    B : float       # Radiative recombination rate
    Cn : float      # Auger coef for two-electron one-hole
    Cp : float      # Auger coef for two-hole one-electron
    tauN : float    # Electron bulk nonradiative decay lifetime
    tauP : float    # Hole bulk nonradiative decayl lifetime
    eps : float     # Relative dielectric cofficient
    Tm : float      # Temperature
    def __init__(self, param_info, initial_guesses):
        self.param_names = param_info["names"]
        self.actives = [(param, index) for index, param in enumerate(self.param_names) if param_info["active"][param]]
        
        for param in self.param_names:
            if hasattr(self, param): raise KeyError(f"Param with name {param} already exists")
            setattr(self, param, initial_guesses[param])
        self.Tm = 300
        return
    
    def apply_unit_conversions(self, param_info=None):
        for param in self.param_names:
            val = getattr(self, param)
            setattr(self, param, val * param_info["unit_conversions"].get(param, 1))
        
    def make_log(self, param_info=None):
        for param in self.param_names:
            if param_info["do_log"].get(param, 0) and hasattr(self, param):
                val = getattr(self, param)
                setattr(self, param, np.log10(val))
                
    def to_array(self, param_info=None):
        arr = np.array([getattr(self, param) for param in self.param_names], dtype=float)
        return arr
    
class Covariance():
    
    def __init__(self, param_info):
        self.names = param_info["names"]
        self.actives = param_info['active']
        d = len(self.names)
        self.cov = np.zeros((d,d))
        return
        
    def set_variance(self, param, var):
        i = self.names.index(param)
        
        if isinstance(var, (int, float)):
            self.cov[i,i] = var
        elif isinstance(var, dict):
            self.cov[i,i] = var[param]
        return
    
    def trace(self):
        return np.diag(self.cov)
    
    def apply_values(self, initial_variance):
        for param in self.names:
            if self.actives[param]:
                self.set_variance(param, initial_variance)
                
        iv_arr = 0
        if isinstance(initial_variance, dict):
            iv_arr = np.ones(len(self.cov))
            for i, param in enumerate(self.names):
                if self.actives[param]:
                    iv_arr[i] = initial_variance[param]
        
        elif isinstance(initial_variance, (float, int)):
            iv_arr = initial_variance
                
        self.little_sigma = np.ones(len(self.cov)) * iv_arr
        self.big_sigma = self.cov * iv_arr**-1
                        
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
        np.save(os.path.join(out_pathname, "final_cov"), self.final_cov)
        
    def truncate(self, k, param_info):
        for param in param_info["names"]:
            val = getattr(self, param)
            setattr(self, param, val[:k])
            
            val = getattr(self, f"mean_{param}")
            setattr(self, f"mean_{param}", val[:k])
            
        self.accept = self.accept[:k]
        self.ratio = self.ratio[:k]
        return
    
             
class Grid():
    def __init__(self):
        return
    
class Solution():
    def __init__(self):
        return

    def calculate_PL(self, g, p):
        rr = calculate_RR(self.N, self.P, p.ks, p.n0, p.p0)
        
        self.PL = trapz(rr, dx=g.dx, axis=1)
        self.PL += rr[:, 0] * g.dx / 2
        self.PL += rr[:, -1] * g.dx / 2

    def calculate_TRTS(self, g, p):
        trts = p.mu_n * (self.N - p.n0) + p.mu_p * (self.P - p.p0)
        self.trts = trapz(trts, dx=g.dx, axis=1)
        self.trts += trts[:, 0] * g.dx / 2
        self.trts += trts[:, -1] * g.dx / 2

@njit(cache=True)
def calculate_RR(N, P, ks, n0, p0):
    return ks * (N * P - n0 * p0)

