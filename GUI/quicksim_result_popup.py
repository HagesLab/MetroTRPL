import tkinter as tk
from tkinter import filedialog

import mc_plot
from popup import Popup
from bayes_io import get_data
from gui_colors import BLACK, WHITE, PLOT_COLOR_CYCLE

WIDTH = 600
HEIGHT = 500

class QuicksimResultPopup(Popup):

    def __init__(self, window, master):
        super().__init__(window, master, WIDTH, HEIGHT)
        self.toplevel.resizable(False, False)
        self.toplevel.title("Quicksim result")
        self.toplevel.attributes('-topmost', 'true')
        self.qs_chart = self.window.Chart(self.toplevel, 400, 400)
        self.qs_chart.place(0, 0)
        self.qs_chart.figure.clear()
        self.qs_axes = self.qs_chart.figure.add_subplot()

        self.load_button = tk.Button(master=self.toplevel, width=12, text="Load",
                                     background=BLACK, foreground=WHITE,
                                     command=self.load_exp_data,
                                     border=4)
        self.load_button.place(x=460, y=20)

        self.sim_results = []
        self.exp_data = []

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
        self.replot_sim_results(["black"])
        self.replot_exp_results(PLOT_COLOR_CYCLE)
        return

    def replot_sim_results(self, colors):
        """Replot all stored quicksim results"""
        for sr in self.sim_results:
            self.plot(sr, colors)
        self.qs_chart.figure.tight_layout()
        self.qs_chart.canvas.draw()

    def replot_exp_results(self, colors):
        """Replot all loaded measurement data"""
        for ed in self.exp_data:
            self.plot(ed, colors)
        self.qs_chart.figure.tight_layout()
        self.qs_chart.canvas.draw()

    def plot(self, sim_result, colors):
        """Add a curve to the quicksim plot"""
        xlabel = "delay time [ns]"
        ylabel = "TRPL"
        scale = "log"
        color = colors[0]
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
