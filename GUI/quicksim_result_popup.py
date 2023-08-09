import tkinter as tk
from tkinter import filedialog
import os

import mc_plot
from popup import Popup
from bayes_io import get_data
from gui_colors import BLACK, WHITE, PLOT_COLOR_CYCLE, LIGHT_GREY, DARK_GREY
from gui_styles import LABEL_KWARGS

BASE_WIDTH = 480
WIDTH_PER_CHAIN = 80
HEIGHT = 500
PLOT_SIZE = 400

class QuicksimResultPopup(Popup):

    def __init__(self, window, master, n_chains, n_sims, active_chain_names):
        self.width = int(BASE_WIDTH + WIDTH_PER_CHAIN * n_chains)
        super().__init__(window, master, self.width, HEIGHT)
        self.toplevel.resizable(False, False)
        self.toplevel.title("Quicksim result")
        self.toplevel.attributes('-topmost', 'true')
        self.qs_chart = self.window.Chart(self.toplevel, PLOT_SIZE, PLOT_SIZE)
        self.qs_chart.place(0, 0)
        self.qs_chart.figure.clear()
        self.qs_axes = self.qs_chart.figure.add_subplot()

        self.c_frame = self.window.Panel(self.toplevel, width=self.width-PLOT_SIZE,
                                         height=100, color=DARK_GREY)
        self.c_frame.place(x=PLOT_SIZE, y=0)

        self.load_button = tk.Button(master=self.c_frame.widget, width=12, text="Load",
                                     background=BLACK, foreground=WHITE,
                                     command=self.load_exp_data,
                                     border=4)
        self.load_button.place(x=20, y=20)

        self.sim_results = []
        self.exp_data = []
        self.n_chains = n_chains
        self.n_sims = n_sims
        self.active_chain_names = active_chain_names

        self.draw_s_frame()
        
    def draw_s_frame(self):
        """
        Grid of entryboxes for each sim's scale factor
        """
        self.s_frame = self.window.Panel(self.toplevel, width=self.width-PLOT_SIZE, height=HEIGHT-100,
                                         color=LIGHT_GREY)
        self.s_frame.place(x=PLOT_SIZE, y=100)
        self.scale_var = [[] for i in range(self.n_sims)]

        for c, fname in enumerate(self.active_chain_names):
            tk.Label(self.s_frame.widget, text=f"\"{os.path.basename(fname)[:4]}...\"\nScale", **LABEL_KWARGS).place(x=40+80*c, y=20)

        for i in range(self.n_sims):
            self.s_frame.widgets[f"Number-{i}"] = tk.Label(self.s_frame.widget, text=f"{i+1}.", width=4, border=3,
                                                                background=LIGHT_GREY)
            self.s_frame.widgets[f"Number-{i}"].place(x=0, y=60+30*i)
            for c in range(self.n_chains):
                self.scale_var[i].append(tk.StringVar())
                self.s_frame.widgets[f"{c}-{i}"] = tk.Entry(master=self.s_frame.widget, width=8, border=3,
                    textvariable=self.scale_var[i][-1], highlightthickness=2, highlightcolor=LIGHT_GREY)
                self.s_frame.widgets[f"{c}-{i}"].place(x=60+80*c, y=60+30*i)

    def group_results_by_chain(self) -> None:
        """Group results in self.sim_results according to which chain they originated from"""
        new_sim_results = [[] for i in range(self.n_chains)]

        for i, sr in enumerate(self.sim_results):
            new_sim_results[i // self.n_sims].append(sr)

        self.sim_results = new_sim_results

    def load_exp_data(self):
        """
        Loads a measurement data file (same as input for MMC)
        for comparison of quicksims.
        """
        self.exp_data.clear()
        ic_flags = {"time_cutoff": None,
                    "select_obs_sets": None,
                    "noise_level": None}
        MCMC_fields = {"log_pl": False,
                       "self_normalize": None}
        
        self.toplevel.attributes('-topmost', 'false')
        fname = filedialog.askopenfilename(filetypes=[("CSV File", "*.csv")],
                                           title="Select Measurement Data", initialdir=".")
        self.toplevel.attributes('-topmost', 'true')
        if fname == "":
            return

        exp_t, exp_y, exp_u = get_data(fname, None, ic_flags, MCMC_fields)
        for ty in zip(exp_t, exp_y):
            self.exp_data.append(ty)

        self.clear()
        self.replot_sim_results(["black"] * self.n_chains)
        self.replot_exp_results(PLOT_COLOR_CYCLE)
        return

    def replot_sim_results(self, colors):
        """
        Replot all stored quicksim results.
        Requires a "grouped" self.sim_results - see group_results_by_chain()
        """
        for c in range(self.n_chains):
            for sr in self.sim_results[c]:
                self.plot(sr, colors[c])
        self.qs_chart.figure.tight_layout()
        self.qs_chart.canvas.draw()

    def replot_exp_results(self, colors):
        """Replot all loaded measurement data"""
        for ed in self.exp_data:
            self.plot(ed, colors)
        self.qs_chart.figure.tight_layout()
        self.qs_chart.canvas.draw()

    def plot(self, sim_result, color):
        """Add a curve to the quicksim plot"""
        xlabel = "delay time [ns]"
        ylabel = "TRPL"
        scale = "log"
        mc_plot.sim_plot(self.qs_axes, sim_result[0], sim_result[1], xlabel,
                         ylabel, scale, color)

    def clear(self):
        """Reset the quicksim plot"""
        # clf() does not work - must use cla() on axis
        self.qs_chart.figure.gca().cla()

    def on_close(self):
        """Re-enable the simulate button"""
        self.toplevel.destroy()
        self.window.mini_panel.widgets["quicksim button"].configure(state=tk.NORMAL)
        self.window.mini_panel.widgets["load button"].configure(state=tk.NORMAL)
