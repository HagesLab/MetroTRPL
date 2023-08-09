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
HEIGHT = 600
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

        self.qs_finished = False
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
        self.scale_var = [[] for c in range(self.n_chains)] # chain-major ordering, like quicksim

        for i in range(self.n_sims):
            self.s_frame.widgets[f"Number-{i}"] = tk.Label(self.s_frame.widget, text=f"{i+1}.", width=4, border=3,
                                                                background=LIGHT_GREY)
            self.s_frame.widgets[f"Number-{i}"].place(x=0, y=105+30*i)
            
        for c, fname in enumerate(self.active_chain_names):
            tk.Label(self.s_frame.widget, text=f"\"{os.path.basename(fname)[:4]}...\"\nScale", **LABEL_KWARGS).place(x=38+80*c, y=20)
            tk.Label(self.s_frame.widget, width=8, height=1, background=PLOT_COLOR_CYCLE[c % len(PLOT_COLOR_CYCLE)]).place(x=60+80*c, y=60)
            for i in range(self.n_sims):
                self.scale_var[c].append(tk.StringVar())
                self.s_frame.widgets[f"{c}-{i}"] = tk.Entry(master=self.s_frame.widget, width=8, border=3,
                    textvariable=self.scale_var[c][-1], highlightthickness=2, highlightcolor=LIGHT_GREY)
                self.s_frame.widgets[f"{c}-{i}"].place(x=62+80*c, y=100+30*i)
                self.s_frame.widgets[f"{c}-{i}"].bind("<FocusOut>", self.redraw)

        self.populate_scale_factors()

    def populate_scale_factors(self) -> None:
        """Fill in s_frame with scale factors for each chain's final state"""
        for i in range(self.n_sims):
            for c in range(self.n_chains):
                scale_f = self.window.data[self.active_chain_names[c]]
                if f"_s{i}" in scale_f:
                    scale_f = scale_f[f"_s{i}"][True][-1]
                else:
                    scale_f = 1
                self.scale_var[c][i].set("{:.2e}".format(scale_f))

    def group_results_by_chain(self) -> None:
        """
        Group results in self.sim_results according to which chain they originated from
        This is designated "chain-major ordering"
        """
        new_sim_results = [[] for c in range(self.n_chains)]

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
        self.replot_exp_results()
        self.replot_sim_results()
        return

    def replot_sim_results(self, colors=PLOT_COLOR_CYCLE):
        """
        Replot all stored quicksim results.
        Requires a "grouped" self.sim_results - see group_results_by_chain()
        """
        for c in range(self.n_chains):
            for i, sr in enumerate(self.sim_results[c]):
                self.plot(sr[0], sr[1] * float(self.scale_var[c][i].get()), colors[c % len(PLOT_COLOR_CYCLE)])
        self.qs_chart.figure.tight_layout()
        self.qs_chart.canvas.draw()

    def replot_exp_results(self, color="black"):
        """Replot all loaded measurement data"""
        for ed in self.exp_data:
            self.plot(ed[0], ed[1], color)
        self.qs_chart.figure.tight_layout()
        self.qs_chart.canvas.draw()

    def plot(self, x, y, color):
        """Add a curve to the quicksim plot"""
        xlabel = "delay time [ns]"
        ylabel = "TRPL"
        scale = "log"
        mc_plot.sim_plot(self.qs_axes, x, y, xlabel,
                         ylabel, scale, color)

    def redraw(self, *args) -> None:
        """Callback for scale_f change"""
        if not self.qs_finished:
            return
        self.clear()
        self.replot_exp_results()
        self.replot_sim_results()

    def clear(self):
        """Reset the quicksim plot"""
        # clf() does not work - must use cla() on axis
        self.qs_chart.figure.gca().cla()

    def finalize(self):
        """End of quicksim callback"""
        self.group_results_by_chain()
        self.clear()
        self.replot_sim_results()
        self.qs_finished = True

    def on_close(self):
        """Re-enable the simulate button"""
        self.toplevel.destroy()
        self.window.mini_panel.widgets["quicksim button"].configure(state=tk.NORMAL)
        self.window.mini_panel.widgets["load button"].configure(state=tk.NORMAL)
