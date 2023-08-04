"""
Simulations require info that isn't necessarily part of the state - 
such as fluence, thickness, and absorption coefficient.
This popup collects values for these "external variables".
"""
import tkinter as tk
from functools import partial

import mc_plot
from popup import Popup
from gui_colors import LIGHT_GREY, BLACK, WHITE, DARK_GREY
from gui_styles import LABEL_KWARGS

WIDTH = 720
HEIGHT = 600

class QuicksimEntryPopup(Popup):

    def __init__(self, window, master, ext_var) -> None:
        super().__init__(window, master, WIDTH, HEIGHT)
        self.ext_var = {ev: [] for ev in ext_var}
        self.continue_ = False
        self.toplevel.resizable(False, False)
        self.toplevel.title("Quicksim Settings")
        self.toplevel.protocol("WM_DELETE_WINDOW", partial(self.on_close, False))

        self.c_frame = self.window.Panel(self.toplevel, width=WIDTH,
                                              height=100, color=DARK_GREY)
        self.ev_frame = self.window.Panel(self.toplevel, width=WIDTH, height=480,
                                         color=LIGHT_GREY)

        self.n_sims = 1
        self.c_frame.variables["n_sims"] = tk.StringVar(value="1")
        self.c_frame.variables["n_sims"].trace("w", self.n_sim_trace)
        self.c_frame.variables["total_sims"] = tk.StringVar()
        self.draw_c_frame()
        self.c_frame.place(0, 0)

        self.expand_ev_frame(0)
        self.ev_frame.place(0, 100)

    def n_sim_trace(self, *args):
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
        n_chains = 0
        for file_name in self.window.file_names:
            if self.window.file_names[file_name].get():
                n_chains += 1

        self.c_frame.variables["total_sims"].set(f"{n_sims * n_chains} total simulations")

    def on_close(self, continue_ : bool=False) -> None:
        self.continue_ = continue_
        self.is_open = False
        self.toplevel.destroy()

    def draw_c_frame(self) -> None:
        self.calc_total_sims()
        self.c_frame.widgets["n_sims_label"] = tk.Label(master=self.c_frame.widget,
                                                text="# Simulations", **LABEL_KWARGS)
        self.c_frame.widgets["n_sims_label"].place(x=20, y=20)
        
        self.c_frame.widgets["n_sims_entry"] = tk.Spinbox(master=self.c_frame.widget, from_=1, to=12,
                                                          state="readonly", width=15, border=3,
                                                          textvariable=self.c_frame.variables["n_sims"])
        self.c_frame.widgets["n_sims_entry"].place(x=20, y=48)

        self.c_frame.widgets["total_sims_label"] = tk.Label(master=self.c_frame.widget,
                                                            textvariable=self.c_frame.variables["total_sims"],
                                                            width=16, background=LIGHT_GREY)
        self.c_frame.widgets["total_sims_label"].place(x=360, y=20)

        self.c_frame.widgets["continue"] = tk.Button(master=self.c_frame.widget, width=15, text="Continue",
                                                     background=BLACK, foreground=WHITE,
                                                     command=partial(self.on_close, True),
                                                     border=4)
        self.c_frame.widgets["continue"].place(x=360, y=40)

    def redraw_ev_frame(self) -> None:
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
            tk.Label(self.ev_frame.widget, text=ev, **LABEL_KWARGS).place(x=60+100*e, y=20)
            self.ext_var[ev].append(tk.StringVar())
            self.ev_frame.widgets[f"{e}-{i}"] = tk.Entry(master=self.ev_frame.widget, width=16, border=3,
                textvariable=self.ext_var[ev][-1])
            self.ev_frame.widgets[f"{e}-{i}"].place(x=60+100*e, y=60+30*i)

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