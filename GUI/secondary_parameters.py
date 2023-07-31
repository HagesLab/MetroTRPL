# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 12:05:40 2022

@author: cfai2
"""
import numpy as np

# Constants
kb = 0.0257 #[ev]
q = 1

class SecondaryParameters():
    """
    Additional quantities or lifetimes calculable from 
    material parameters that go directly into the 
    carrier simulation.

    All material parameters in [cm, V, s] units.
    Thickness in nm.
    """

    def __init__(self):
        self.func = {"t_rad": (self.t_rad, ("ks", "p0")),
                     "t_auger": (self.t_auger, ("Cp", "p0")),
                     "LI_tau_eff": (self.li_tau_eff, ("ks", "p0", "tauN", "Sf", "Sb", "Cp", "thickness", "mu_n", "mu_p")),
                     "LI_tau_srh": (self.li_tau_srh, ("tauN", "Sf", "Sb", "thickness", "mu_n", "mu_p")),
                     "HI_tau_srh": (self.hi_tau_srh, ("tauN", "tauP", "Sf", "Sb", "thickness", "mu_n", "mu_p")),
                     "Sf+Sb": (self.s_eff, ("Sf", "Sb")),
                     "mu_ambi": (self.mu_eff, ("mu_n", "mu_p")),
                     "epsilon": (self.epsilon, ("lambda",))}

    def t_rad(self, p : dict[str, np.ndarray | float]) -> np.ndarray | float:
        """Radiative recombination lifetime, in ns"""
        return 1 / (p["ks"] * p["p0"]) * 1e9

    def t_auger(self, p : dict[str, np.ndarray | float]) -> np.ndarray | float:
        """Auger recombination lifetime, in ns"""
        return 1 / (p["Cp"] * p["p0"]**2) * 10**9

    def li_tau_eff(self, p : dict[str, np.ndarray | float]) -> np.ndarray | float:
        """Low injection effective lifetime, in ns"""
        diffusivity = self.mu_eff(p) * kb / q * 1e14 / 1e9 # [cm^2 / V s] * [eV] / [eV/V] = [cm^2/s] -> [nm^2/ns]
        tau_surf = (p["thickness"] / ((p["Sf"] + p["Sb"]) * 0.01)) + (p["thickness"]**2 / (np.pi**2 * diffusivity))
        t_r = self.t_rad(p)
        t_aug = self.t_auger(p)
        return (t_r**-1 + t_aug**-1 + tau_surf**-1 + p["tauN"]**-1)**-1

    def li_tau_srh(self, p : dict[str, np.ndarray | float]) -> np.ndarray | float:
        """Low injection Shockley-Reed-Hall lifetime, in ns (this excludes radiative and Auger)"""
        diffusivity = self.mu_eff(p) * kb / q * 1e14 / 1e9
        tau_surf = (p["thickness"] / ((p["Sf"] + p["Sb"]) * 0.01)) + (p["thickness"]**2 / (np.pi ** 2 * diffusivity))
        return (tau_surf**-1 + p["tauN"]**-1)**-1

    def hi_tau_srh(self, p : dict[str, np.ndarray | float]) -> np.ndarray | float:
        """High injection Shockley-Reed-Hall lifetime, in ns"""
        diffusivity = self.mu_eff(p) * kb / q * 1e14 / 1e9
        tau_surf = 2 * (p["thickness"] / ((p["Sf"] + p["Sb"]) * 0.01)) + (p["thickness"]**2 / (np.pi ** 2 * diffusivity))
        return (tau_surf**-1 + (p["tauN"] + p["tauP"])**-1)**-1

    def s_eff(self, p: dict[str, np.ndarray | float]) -> np.ndarray | float:
        """Total surface recombination, in cm s^-1"""
        return p["Sf"] + p["Sb"]

    def mu_eff(self, p : dict[str, np.ndarray | float]) -> np.ndarray | float:
        """Ambipolar mobility, in cm^2 V^-1 s^-1"""
        return 2 / (p["mu_n"]**-1 + p["mu_p"]**-1)

    def epsilon(self, p : dict[str, np.ndarray | float]) -> np.ndarray | float:
        """Relative dielectric permittivity"""
        return p["lambda"]**-1