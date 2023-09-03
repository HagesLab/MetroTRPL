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
        # dict[str, (fxn, needed params)]
        self.func = {"t_rad": (self.t_rad, ("ks", "p0")),
                     "t_auger": (self.t_auger, ("Cp", "p0")),
                     "LI_tau_eff": (self.li_tau_eff, ("ks", "p0", "tauN", "Sf", "Sb", "Cp", "thickness", "mu_n", "mu_p")),
                     "LI_tau_srh": (self.li_tau_srh, ("tauN", "Sf", "Sb", "thickness", "mu_n", "mu_p")),
                     "HI_tau_srh": (self.hi_tau_srh, ("tauN", "tauP", "Sf", "Sb", "thickness", "mu_n", "mu_p")),
                     "tauN+tauP": (self.tauN_tauP, ("tauN", "tauP")),
                     "Sf+Sb": (self.s_eff, ("Sf", "Sb")),
                     "Cn+Cp": (self.c_eff, ("Cn", "Cp")),
                     "mu_ambi": (self.mu_eff, ("mu_n", "mu_p")),
                     "epsilon": (self.epsilon, ("lambda",)),
                     "tauC": (self.tauC, ("kC", "Nt")),
                     "Rc-Re": (self.trap_rate, ("kC", "Nt", "tauE")),
                     "Rc+Rsrh": (self.n_removal_rate, ("tauN", "tauP", "Sf", "Sb", "thickness", "mu_n", "mu_p", "kC", "Nt", "tauE"))}

        # Most recent thickness used to calculate; determines if recalculation needed when thickness updated
        self.last_thickness = {name: -1 for name in self.func if "thickness" in self.func[name][1]}

    def get(self, data, data_keys, thickness : str) -> None:
        """
        Calculate and cache the requested secondary parameter from a Data object provided by GUI

        Parameters
        ----------
        data : Data
            Object containing MCMC results loaded from a .pik file produced by metropolis()
        data_keys : dict[str, str | bool]
            Dict of three necessary indices for data - file_name that the result came from,
            value to calculate, and whether to use the accepted or raw data.
        thickness : str
            Thickness string from GUI, needed for some diffusion lifetimes.
        """
        file_name = data_keys["file_name"]
        value = data_keys["value"]
        accepted = data_keys["accepted"]
        primary_params = {}
        for needed_param in self.func[value][1]:
            if needed_param == "thickness": # Not included in MCMC data
                try:
                    primary_params["thickness"] = float(thickness)
                except ValueError as err: # invalid thickness
                    raise ValueError("Thickness value needed") from err
            else:
                try:
                    primary_params[needed_param] = data[file_name][needed_param][accepted]
                except KeyError as err:
                    raise KeyError(f"Data {file_name} missing parameter {needed_param}") from err

        try:
            y = self.func[value][0](primary_params)
            data[file_name][value][accepted] = np.array(y)
        except KeyError as err:
            raise KeyError(f"Failed to calculate {value}") from err


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
    
    def tauN_tauP(self, p: dict[str, np.ndarray | float]) -> np.ndarray | float:
        """Sum of tau_N and tau_P, in ns"""
        return p["tauN"] + p["tauP"]

    def s_eff(self, p: dict[str, np.ndarray | float]) -> np.ndarray | float:
        """Total surface recombination, in cm s^-1"""
        return p["Sf"] + p["Sb"]
    
    def c_eff(self, p: dict[str, np.ndarray | float]) -> np.ndarray | float:
        """Effective Auger recombination rate, in cm^6 s^-1"""
        return p["Cn"] + p["Cp"]

    def mu_eff(self, p : dict[str, np.ndarray | float]) -> np.ndarray | float:
        """Ambipolar mobility, in cm^2 V^-1 s^-1"""
        return 2 / (p["mu_n"]**-1 + p["mu_p"]**-1)

    def epsilon(self, p : dict[str, np.ndarray | float]) -> np.ndarray | float:
        """Relative dielectric permittivity"""
        return p["lambda"]**-1
    
    def tauC(self, p):
        """Maximum low-occupation capture time, in ns"""
        return 1 / (p["Nt"] * p["kC"]) * 1e9
    
    def trap_rate(self, p):
        """Trap 'rate', in s^-1 """
        return p["kC"] * p['Nt'] - (1 / p["tauE"] * 1e9)
    
    def n_removal_rate(self, p):
        return (1 / self.hi_tau_srh(p) * 1e9) + p["kC"] * p['Nt']
