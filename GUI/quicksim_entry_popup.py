"""
Simulations require info that isn't necessarily part of the state - 
such as fluence, thickness, and absorption coefficient.
This popup collects values for these "external variables".
"""
import os
import tkinter as tk
import numpy as np
from tkinter import filedialog
from functools import partial
from forward_solver import MODELS

from popup import Popup
from gui_colors import LIGHT_GREY, BLACK, WHITE, DARK_GREY, RED
from gui_styles import LABEL_KWARGS

WIDTH = 970
HEIGHT = 600
DEFAULT_N_SIMS = 3
KEYBIND_DIR = "keybinds"

class QuicksimEntryPopup(Popup):

    def __init__(self, window, master, ext_var) -> None:
        super().__init__(window, master, WIDTH, HEIGHT)
        self.ext_var = {ev: [] for ev in ext_var}
        self.continue_ = False
        self.toplevel.resizable(False, False)
        self.toplevel.title("Quicksim Settings")
        self.toplevel.attributes('-topmost', 'true')
        self.toplevel.attributes('-topmost', 'false')
        self.toplevel.protocol("WM_DELETE_WINDOW", partial(self.on_close, False))
        self.load_keybinds()

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

        self.c_frame.widgets["model_label"] = tk.Label(master=self.c_frame.widget,
                                                       text="Select model", **LABEL_KWARGS)
        
        self.c_frame.widgets["model_label"].place(x=140, y=20)

        available_models = list(MODELS)
        default_model = available_models.pop()
        self.model = tk.StringVar(value=default_model)
        self.c_frame.widgets["model"] = tk.OptionMenu(self.c_frame.widget, self.model, default_model,
                                                      *available_models)
        self.c_frame.widgets["model"].configure(width=10, highlightcolor=DARK_GREY)
        self.c_frame.widgets["model"].place(x=140, y=48)

        self.c_frame.widgets["copy down"] = tk.Button(master=self.c_frame.widget, width=12, text="Copy #1",
                                                      background=BLACK, foreground=WHITE,
                                                      command=self.duplicate,
                                                      border=4)
        self.c_frame.widgets["copy down"].place(x=340, y=48)

        self.c_frame.widgets["clear all"] = tk.Button(master=self.c_frame.widget, width=12, text="Clear All",
                                                      background=BLACK, foreground=WHITE,
                                                      command=self.wipe,
                                                      border=4)
        self.c_frame.widgets["clear all"].place(x=460, y=48)

        self.c_frame.widgets["total_sims_label"] = tk.Label(master=self.c_frame.widget,
                                                            textvariable=self.c_frame.variables["total_sims"],
                                                            width=16, background=LIGHT_GREY)
        self.c_frame.widgets["total_sims_label"].place(x=580, y=20)

        self.c_frame.widgets["continue"] = tk.Button(master=self.c_frame.widget, width=15, text="Continue",
                                                     background=BLACK, foreground=WHITE,
                                                     command=partial(self.on_close, True),
                                                     border=4)
        self.c_frame.widgets["continue"].place(x=580, y=48)

        self.c_frame.widgets["save_keybind"] = tk.Button(master=self.c_frame.widget, width=15, text="Save All as Keybind",
                                                     background=BLACK, foreground=WHITE,
                                                     command=partial(self.save_keybind),
                                                     border=4)
        self.c_frame.widgets["save_keybind"].place(x=720, y=48)

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

    def clear_ev_frame(self) -> None:
        for widget in self.ev_frame.widgets:
            self.ev_frame.widgets[widget].destroy()

        self.ev_frame.widgets.clear()

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

    def save_keybind(self) -> None:
        if not self.validate_all():
            return
        
        if not os.path.exists(KEYBIND_DIR):
            os.makedirs(KEYBIND_DIR, exist_ok=True)

        self.toplevel.attributes('-topmost', 'false')
        fname = filedialog.asksaveasfilename(filetypes=[("Text", "*.txt")],
                                             defaultextension=".txt",
                                             title="Save keybind",
                                             initialdir=KEYBIND_DIR)
        self.toplevel.attributes('-topmost', 'true')
        self.toplevel.attributes('-topmost', 'false')
        if fname == "":
            return

        vals = np.zeros((len(self.ext_var), self.n_sims))
        ext_vars = []
        for e, ev in enumerate(self.ext_var):
            ext_vars.append(ev)
            for i in range(self.n_sims):
                vals[e, i] = float(self.ext_var[ev][i].get())

        np.savetxt(fname, vals.T, delimiter="\t", header="\t".join(ext_vars))

    def load_keybinds(self) -> None:
        keybinds = []
        for f in os.listdir(KEYBIND_DIR):
            kb = f[:f.find(".txt")]
            if len(kb) > 1:
                self.window.status(f"Warning: {kb} invalid keybind not loaded")
            else:
                keybinds.append(kb)

        for kb in keybinds:
            def keybind(kb, *args) -> None:
                vals = np.loadtxt(os.path.join(KEYBIND_DIR, f"{kb}.txt")).T
                for i in range(self.n_sims):
                    for e, ev in enumerate(self.ext_var):
                        try:
                            self.ext_var[ev][i].set(str(vals[e, i]))
                        except IndexError:
                            continue

            self.toplevel.bind(kb, partial(keybind, kb))
