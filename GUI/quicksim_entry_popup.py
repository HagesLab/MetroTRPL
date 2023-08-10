"""
Simulations require info that isn't necessarily part of the state - 
such as fluence, thickness, and absorption coefficient.
This popup collects values for these "external variables".
"""
import tkinter as tk
from functools import partial

from popup import Popup
from gui_colors import LIGHT_GREY, BLACK, WHITE, DARK_GREY, RED
from gui_styles import LABEL_KWARGS

WIDTH = 970
HEIGHT = 600
DEFAULT_N_SIMS = 3

class QuicksimEntryPopup(Popup):

    def __init__(self, window, master, ext_var) -> None:
        super().__init__(window, master, WIDTH, HEIGHT)
        self.ext_var = {ev: [] for ev in ext_var}
        self.continue_ = False
        self.toplevel.resizable(False, False)
        self.toplevel.title("Quicksim Settings")
        self.toplevel.protocol("WM_DELETE_WINDOW", partial(self.on_close, False))
        self.toplevel.bind(";", self.DEBUG)
        self.toplevel.bind("z", self.DEBUG_CD0)
        self.toplevel.bind("v", self.DEBUG_CD3)

        self.c_frame = self.window.Panel(self.toplevel, width=WIDTH,
                                              height=100, color=DARK_GREY)
        self.ev_frame = self.window.Panel(self.toplevel, width=WIDTH, height=480,
                                         color=LIGHT_GREY)

        self.n_sims = DEFAULT_N_SIMS
        self.c_frame.variables["n_sims"] = tk.StringVar(value=str(self.n_sims))
        self.c_frame.variables["n_sims"].trace("w", self.n_sim_trace)
        self.c_frame.variables["total_sims"] = tk.StringVar()
        self.draw_c_frame()
        self.c_frame.place(0, 0)
        for i in range(self.n_sims):
            self.expand_ev_frame(i)
        self.ev_frame.place(0, 100)

        self.is_open = True

    def n_sim_trace(self, *args):
        """When the number of simulations slider is updated"""
        self.calc_total_sims()
        self.redraw_ev_frame()

    def parse_n_sims(self):
        try:
            n_sims = int(self.c_frame.variables["n_sims"].get())
        except ValueError:
            n_sims = 0
        return n_sims

    def calc_total_sims(self, *args):
        """Calculate total number of sims, n_sims * number of chains"""
        n_sims = self.parse_n_sims()
        n_chains = self.window.get_n_chains()

        self.c_frame.variables["total_sims"].set(f"{n_sims * n_chains} total simulations")

    def on_close(self, continue_ : bool=False) -> None:
        self.continue_ = continue_
        if self.continue_:
            self.continue_ = self.validate_all()
            if not self.continue_: # Validation fail
                return
        self.is_open = False
        self.toplevel.destroy()

    def draw_c_frame(self) -> None:
        """Draw the top frame with the # sims slider, continue button, etc"""
        self.calc_total_sims()
        self.c_frame.widgets["n_sims_label"] = tk.Label(master=self.c_frame.widget,
                                                text="# Simulations", **LABEL_KWARGS)
        self.c_frame.widgets["n_sims_label"].place(x=20, y=20)
        
        self.c_frame.widgets["n_sims_entry"] = tk.Spinbox(master=self.c_frame.widget, from_=1, to=12,
                                                          state="readonly", width=14, border=3,
                                                          textvariable=self.c_frame.variables["n_sims"])
        self.c_frame.widgets["n_sims_entry"].place(x=20, y=48)

        self.c_frame.widgets["copy down"] = tk.Button(master=self.c_frame.widget, width=12, text="Copy #1",
                                                      background=BLACK, foreground=WHITE,
                                                      command=self.duplicate,
                                                      border=4)
        self.c_frame.widgets["copy down"].place(x=180, y=48)

        self.c_frame.widgets["clear all"] = tk.Button(master=self.c_frame.widget, width=12, text="Clear All",
                                                      background=BLACK, foreground=WHITE,
                                                      command=self.wipe,
                                                      border=4)
        self.c_frame.widgets["clear all"].place(x=300, y=48)

        self.c_frame.widgets["total_sims_label"] = tk.Label(master=self.c_frame.widget,
                                                            textvariable=self.c_frame.variables["total_sims"],
                                                            width=16, background=LIGHT_GREY)
        self.c_frame.widgets["total_sims_label"].place(x=420, y=20)

        self.c_frame.widgets["continue"] = tk.Button(master=self.c_frame.widget, width=15, text="Continue",
                                                     background=BLACK, foreground=WHITE,
                                                     command=partial(self.on_close, True),
                                                     border=4)
        self.c_frame.widgets["continue"].place(x=420, y=48)

    def redraw_ev_frame(self) -> None:
        """Adjust the large botton frame to accommodate number of sims"""
        n_sims = self.parse_n_sims()
        if n_sims > self.n_sims:
            self.expand_ev_frame(n_sims-1)
        elif n_sims < self.n_sims:
            self.contract_ev_frame(n_sims)
        self.n_sims = n_sims

    def expand_ev_frame(self, i : int) -> None:
        """
        Add more widgets for the additional ith simulation
        First simulation is i=0
        """
        self.ev_frame.widgets[f"Number-{i}"] = tk.Label(self.ev_frame.widget, text=f"{i+1}.", width=4, border=3,
                                                            background=LIGHT_GREY)
        self.ev_frame.widgets[f"Number-{i}"].place(x=0, y=60+30*i)
        for e, ev in enumerate(self.ext_var):
            tk.Label(self.ev_frame.widget, text=ev, **LABEL_KWARGS).place(x=60+110*e, y=20)
            self.ext_var[ev].append(tk.StringVar())
            self.ev_frame.widgets[f"{e}-{i}"] = tk.Entry(master=self.ev_frame.widget, width=16, border=3,
                textvariable=self.ext_var[ev][-1], highlightthickness=2, highlightcolor=LIGHT_GREY)
            self.ev_frame.widgets[f"{e}-{i}"].place(x=60+110*e, y=60+30*i)

    def contract_ev_frame(self, i : int) -> None:
        """Remove widgets belonging to the deleted ith simulation"""
        self.ev_frame.widgets[f"Number-{i}"].destroy()
        self.ev_frame.widgets.pop(f"Number-{i}")

        for e, ev in enumerate(self.ext_var):
            self.ext_var[ev].pop()
            self.ev_frame.widgets[f"{e}-{i}"].destroy()
            self.ev_frame.widgets.pop(f"{e}-{i}")

    def duplicate(self) -> None:
        """Copy the values written for the first sim into all other sims"""
        for ev in self.ext_var:
            for i in range(1, len(self.ext_var[ev])):
                self.ext_var[ev][i].set(self.ext_var[ev][0].get())

    def wipe(self) -> None:
        """Clear all written values"""
        for ev in self.ext_var:
            for i in range(len(self.ext_var[ev])):
                self.ext_var[ev][i].set("")

    def validate_all(self) -> bool:
        """
        Validates all written values; highlights invalid ones
        The continue button will not function while an invalid value exists
        """
        valid = True
        for e, ev in enumerate(self.ext_var):
            for i in range(len(self.ext_var[ev])):
                try:
                    test = self.ext_var[ev][i].get()
                    if test == "":
                        raise ValueError
                    float(test)
                    self.ev_frame.widgets[f"{e}-{i}"].config(highlightbackground=LIGHT_GREY)
                except ValueError:
                    valid = False
                    self.ev_frame.widgets[f"{e}-{i}"].config(highlightbackground=RED)

        return valid

    def clear_ev_frame(self) -> None:
        for widget in self.ev_frame.widgets:
            self.ev_frame.widgets[widget].destroy()

        self.ev_frame.widgets.clear()

    def DEBUG(self, *args) -> None:
        """Popupate Sim #1 with specific external values"""
        self.ext_var["thickness"][0].set(2000)
        self.ext_var["nx"][0].set(128)
        self.ext_var["final_time"][0].set(2000)
        self.ext_var["nt"][0].set(8000)
        self.ext_var["fluence"][0].set(2.75e13)
        self.ext_var["absp"][0].set(6e4)
        self.ext_var["direction"][0].set(1)
        self.ext_var["wavelength"][0].set(496)

        self.ext_var["thickness"][1].set(2000)
        self.ext_var["nx"][1].set(128)
        self.ext_var["final_time"][1].set(2000)
        self.ext_var["nt"][1].set(8000)
        self.ext_var["fluence"][1].set(1.92e12)
        self.ext_var["absp"][1].set(6e4)
        self.ext_var["direction"][1].set(1)
        self.ext_var["wavelength"][1].set(496)

        self.ext_var["thickness"][2].set(2000)
        self.ext_var["nx"][2].set(128)
        self.ext_var["final_time"][2].set(2000)
        self.ext_var["nt"][2].set(8000)
        self.ext_var["fluence"][2].set(2.12e11)
        self.ext_var["absp"][2].set(6e4)
        self.ext_var["direction"][2].set(1)
        self.ext_var["wavelength"][2].set(496)

    def DEBUG_CD3(self, *args) -> None:
        """Popupate Sim #1 with specific external values"""
        for i in range(8):
            self.ext_var["thickness"][i].set(3000)
            self.ext_var["nx"][i].set(128)
            self.ext_var["nt"][i].set(3000)
        
        
        self.ext_var["final_time"][0].set(35)
        self.ext_var["final_time"][1].set(35)
        self.ext_var["final_time"][2].set(80)
        self.ext_var["final_time"][3].set(170)
        self.ext_var["final_time"][4].set(150)
        self.ext_var["final_time"][5].set(70)
        self.ext_var["final_time"][6].set(160)
        self.ext_var["final_time"][7].set(300)

        self.ext_var["fluence"][0].set(4.27e12)
        self.ext_var["fluence"][1].set(3.34e12)
        self.ext_var["fluence"][2].set(8.15e12)
        self.ext_var["fluence"][3].set(2.25e13)
        self.ext_var["fluence"][4].set(4.71e12)
        self.ext_var["fluence"][5].set(4.00e12)
        self.ext_var["fluence"][6].set(9.04e12)
        self.ext_var["fluence"][7].set(2.51e13)

        self.ext_var["absp"][0].set(117150)
        self.ext_var["absp"][1].set(37215)
        self.ext_var["absp"][2].set(37215)
        self.ext_var["absp"][3].set(37215)
        self.ext_var["absp"][4].set(117150)
        self.ext_var["absp"][5].set(37215)
        self.ext_var["absp"][6].set(37215)
        self.ext_var["absp"][7].set(37215)

        self.ext_var["direction"][0].set(1)
        self.ext_var["direction"][1].set(1)
        self.ext_var["direction"][2].set(1)
        self.ext_var["direction"][3].set(1)
        self.ext_var["direction"][4].set(-1)
        self.ext_var["direction"][5].set(-1)
        self.ext_var["direction"][6].set(-1)
        self.ext_var["direction"][7].set(-1)

        self.ext_var["wavelength"][0].set(520)
        self.ext_var["wavelength"][1].set(745)
        self.ext_var["wavelength"][2].set(745)
        self.ext_var["wavelength"][3].set(745)
        self.ext_var["wavelength"][4].set(520)
        self.ext_var["wavelength"][5].set(745)
        self.ext_var["wavelength"][6].set(745)
        self.ext_var["wavelength"][7].set(745)

    def DEBUG_CD0(self, *args) -> None:
        """Popupate Sim #1 with specific external values"""
        for i in range(8):
            self.ext_var["thickness"][i].set(3000)
            self.ext_var["nx"][i].set(128)
            self.ext_var["nt"][i].set(3000)
        
        
        self.ext_var["final_time"][0].set(20)
        self.ext_var["final_time"][1].set(20)
        self.ext_var["final_time"][2].set(30)
        self.ext_var["final_time"][3].set(90)
        self.ext_var["final_time"][4].set(15)
        self.ext_var["final_time"][5].set(15)
        self.ext_var["final_time"][6].set(30)
        self.ext_var["final_time"][7].set(90)

        self.ext_var["fluence"][0].set(4.27e12)
        self.ext_var["fluence"][1].set(3.37e12)
        self.ext_var["fluence"][2].set(8.15e12)
        self.ext_var["fluence"][3].set(2.24e13)
        self.ext_var["fluence"][4].set(4.74e12)
        self.ext_var["fluence"][5].set(3.89e12)
        self.ext_var["fluence"][6].set(8.89e12)
        self.ext_var["fluence"][7].set(2.47e13)

        self.ext_var["absp"][0].set(80000)
        self.ext_var["absp"][1].set(30500)
        self.ext_var["absp"][2].set(30500)
        self.ext_var["absp"][3].set(30500)
        self.ext_var["absp"][4].set(80000)
        self.ext_var["absp"][5].set(30500)
        self.ext_var["absp"][6].set(30500)
        self.ext_var["absp"][7].set(30500)

        self.ext_var["direction"][0].set(1)
        self.ext_var["direction"][1].set(1)
        self.ext_var["direction"][2].set(1)
        self.ext_var["direction"][3].set(1)
        self.ext_var["direction"][4].set(-1)
        self.ext_var["direction"][5].set(-1)
        self.ext_var["direction"][6].set(-1)
        self.ext_var["direction"][7].set(-1)

        self.ext_var["wavelength"][0].set(520)
        self.ext_var["wavelength"][1].set(745)
        self.ext_var["wavelength"][2].set(745)
        self.ext_var["wavelength"][3].set(745)
        self.ext_var["wavelength"][4].set(520)
        self.ext_var["wavelength"][5].set(745)
        self.ext_var["wavelength"][6].set(745)
        self.ext_var["wavelength"][7].set(745)