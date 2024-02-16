import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Progressbar
import os
import itertools
from functools import partial
import numpy as np
from matplotlib.backends._backend_tk import NavigationToolbar2Tk

import mc_plot
from popup import Popup
from bayes_io import get_data
from rclickmenu import Clickmenu, CLICK_EVENTS
from gui_colors import BLACK, WHITE, PLOT_COLOR_CYCLE, LIGHT_GREY, DARK_GREY
from gui_styles import LABEL_KWARGS

BASE_WIDTH = 480
WIDTH_PER_CHAIN = 80
HEIGHT = 600
PLOT_SIZE = 400
SCALE_Y_OFFSET = 105
SCALE_Y_INTERVAL = 30

class QSRClickmenu(Clickmenu):

    def __init__(self, window, master, chart, scale_factors : dict[str, list], scale_var, callback):
        super().__init__(window, master, chart)
        for s_f in scale_factors:
            self.menu.add_command(label=s_f, command=partial(self.paste, scale_factors[s_f]))
        self.scale_var = scale_var
        self.callback = callback

    def show(self, event):
        """Check that click falls within a row before showing"""
        row = int((event.y - SCALE_Y_OFFSET) / SCALE_Y_INTERVAL)

        if row < 0:
            # Clicked too high
            return

        if row >= self.window.n_sims:
            # Clicked too low
            return

        super().show(event)

    def paste(self, scales):
        """Copy selected row's values"""
        row = int((self.latest_event[1] - SCALE_Y_OFFSET) / SCALE_Y_INTERVAL)
        for c in range(len(scales)):
            self.scale_var[c][row].set(f"{scales[c]:.2e}")
        self.callback()


class QuicksimResultPopup(Popup):

    def __init__(self, window, master, n_chains, n_sims, active_chain_inds):
        self.width = int(BASE_WIDTH + WIDTH_PER_CHAIN * n_chains)
        super().__init__(window, master, self.width, HEIGHT)
        self.toplevel.resizable(False, False)
        self.toplevel.title("Quicksim result")
        self.toplevel.attributes('-topmost', 'true')
        self.toplevel.attributes('-topmost', 'false')
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
        self.load_button.place(x=20, y=10)
        self.load_button.configure(state=tk.DISABLED)

        self.export_button = tk.Button(master=self.c_frame.widget, width=12, text="Export sims",
                                       background=BLACK, foreground=WHITE,
                                       command=self.export_sims,
                                       border=4)
        self.export_button.place(x=20, y=50)
        self.export_button.configure(state=tk.DISABLED)

        self.qs_finished = False
        self.sim_results = []
        self.exp_data = []
        self.n_chains = n_chains
        self.n_sims = n_sims
        self.active_chain_inds = active_chain_inds
        self.available_scale_factors = [s_f for s_f in self.window.chains[self.active_chain_inds[0]].data if s_f.startswith("_s")]

        self.all_scale_factors = {}
        for s_f in self.available_scale_factors:
            self.all_scale_factors[s_f] = []
            for c in range(self.n_chains):
                self.all_scale_factors[s_f].append(self.window.chains[self.active_chain_inds[c]].data[s_f][-1])

        self.draw_s_frame()

        self.clickmenu = QSRClickmenu(self, self.toplevel, self.s_frame.widget, self.all_scale_factors, self.scale_var, self.redraw)
        self.toplevel.bind(CLICK_EVENTS["click"]["right"], self.clickmenu.show)

        self.o_frame = self.window.Panel(self.toplevel, width=PLOT_SIZE, height=HEIGHT-PLOT_SIZE,
                                         color=LIGHT_GREY)
        self.o_frame.place(x=0, y=PLOT_SIZE)

        toolbar = NavigationToolbar2Tk(self.qs_chart.canvas, self.o_frame.widget, pack_toolbar=False)
        toolbar.place(x=0, y=0, width=PLOT_SIZE-10)

        self.progress_text = tk.StringVar(value=f"0 of {self.n_chains*self.n_sims} complete")
        self.progress_bar_text = tk.Label(self.o_frame.widget, textvariable=self.progress_text, **LABEL_KWARGS)
        self.progress_bar_text.place(x=150, y=80)

        self.progress = tk.IntVar(value=0)
        self.progress_bar = Progressbar(self.o_frame.widget, mode="determinate", length=PLOT_SIZE-30,
                                        maximum=(self.n_chains*self.n_sims), variable=self.progress)
        self.progress_bar.place(x=10, y=100)

        self.sim_vis = tk.IntVar(value=1)
        self.sim_vis_cb = tk.Checkbutton(self.o_frame.widget, text="Toggle Sim Vis.",
                                             variable=self.sim_vis,
                                             **{"width": 10, "background": LIGHT_GREY})
        self.sim_vis.trace("w", self.redraw)
        self.sim_vis_cb.place(x=10, y=140)

        self.is_open = True
        
    def draw_s_frame(self):
        """
        Grid of entryboxes for each sim's scale factor
        """
        self.s_frame = self.window.Panel(self.toplevel, width=self.width-PLOT_SIZE, height=HEIGHT-100,
                                         color=LIGHT_GREY)
        self.s_frame.place(x=PLOT_SIZE, y=100)
        self.scale_var = [[] for c in range(self.n_chains)] # chain-major ordering, like quicksim
        self.specific_sim_visibility = []
        for i in range(self.n_sims):
            self.s_frame.widgets[f"Number-{i}"] = tk.Label(self.s_frame.widget, text=f"{i+1}.", width=4, border=3,
                                                                background=LIGHT_GREY)
            self.s_frame.widgets[f"Number-{i}"].place(x=0, y=SCALE_Y_OFFSET+SCALE_Y_INTERVAL*i)
            self.specific_sim_visibility.append(tk.IntVar(value=1))
            self.specific_sim_visibility[i].trace("w", self.redraw)
            self.s_frame.widgets[f"Visible-{i}"] = tk.Checkbutton(self.s_frame.widget, variable=self.specific_sim_visibility[i], background=LIGHT_GREY)
            self.s_frame.widgets[f"Visible-{i}"].place(x=25, y=SCALE_Y_OFFSET+SCALE_Y_INTERVAL*i)
            
        for c, ind in enumerate(self.active_chain_inds):
            chain = self.window.chains[ind]
            tk.Label(self.s_frame.widget, text=f"\"{os.path.basename(chain.fname)[:4]}...\"\nScale", **LABEL_KWARGS).place(x=38+80*c, y=20)
            tk.Label(self.s_frame.widget, width=8, height=1, background=PLOT_COLOR_CYCLE[c % len(PLOT_COLOR_CYCLE)]).place(x=60+80*c, y=60)
            for i in range(self.n_sims):
                self.scale_var[c].append(tk.StringVar())
                self.s_frame.widgets[f"{c}-{i}"] = tk.Entry(master=self.s_frame.widget, width=8, border=3,
                    textvariable=self.scale_var[c][-1], highlightthickness=2,)# highlightcolor=LIGHT_GREY)
                self.s_frame.widgets[f"{c}-{i}"].place(x=62+80*c, y=SCALE_Y_OFFSET+SCALE_Y_INTERVAL*i)
                self.s_frame.widgets[f"{c}-{i}"].bind("<FocusOut>", self.redraw)

        self.populate_scale_factors()

    def populate_scale_factors(self) -> None:
        """Fill in s_frame with scale factors for each chain's final state"""

        for i in range(self.n_sims):
            for c in range(self.n_chains):
                scale_f = self.window.chains[self.active_chain_inds[c]].data

                if f"_s{i}" in scale_f:
                    scale_f = scale_f[f"_s{i}"][-1]
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
                    }
        MCMC_fields = {"log_y": False}
        
        self.toplevel.attributes('-topmost', 'false')
        fname = filedialog.askopenfilename(filetypes=[("CSV File", "*.csv")],
                                           title="Select Measurement Data", initialdir=".")
        self.toplevel.attributes('-topmost', 'true')
        self.toplevel.attributes('-topmost', 'false')
        if fname == "":
            return

        exp = get_data(fname, ic_flags, MCMC_fields)
        for ty in zip(exp[0], exp[1]):
            self.exp_data.append(ty)

        self.clear()
        self.replot_exp_results()
        self.replot_sim_results()
        return

    def export_sims(self):
        """
        Save quicksim results to file
        """
        self.export_button.configure(state=tk.DISABLED)
        self.load_button.configure(state=tk.DISABLED)
        self.toplevel.attributes('-topmost', 'false')
        fname = filedialog.asksaveasfilename(filetypes=[("CSV file", "*.csv")],
                                             defaultextension=".csv",
                                             title="Export sims", initialdir=".")
        self.toplevel.attributes('-topmost', 'true')
        self.toplevel.attributes('-topmost', 'false')
        if fname == "":
            self.load_button.configure(state=tk.NORMAL)
            self.export_button.configure(state=tk.NORMAL)
            return

        result = []
        header = []
        for c in range(self.n_chains):
            for i, sr in enumerate(self.sim_results[c]):
                result.append(sr[0])
                result.append(sr[1] * float(self.scale_var[c][i].get()))
                header.append(f"{os.path.basename(self.window.chains[self.active_chain_inds[c]].fname)} - {i} - time")
                header.append(f"{os.path.basename(self.window.chains[self.active_chain_inds[c]].fname)} - {i} - y")

        result = np.array(list(map(list, itertools.zip_longest(*result, fillvalue=-1))))
        np.savetxt(fname, result, header=",".join(header), delimiter=",")
        self.export_button.configure(state=tk.NORMAL)
        self.load_button.configure(state=tk.NORMAL)
        self.window.status(f"Sims exported to {fname}")

    def replot_sim_results(self, colors=PLOT_COLOR_CYCLE):
        """
        Replot all stored quicksim results.
        Requires a "grouped" self.sim_results - see group_results_by_chain()
        """
        for c in range(self.n_chains):
            for i, sr in enumerate(self.sim_results[c]):
                if self.specific_sim_visibility[i].get() == 0:
                    continue
                self.plot(sr[0], sr[1] * float(self.scale_var[c][i].get()), colors[c % len(PLOT_COLOR_CYCLE)],
                          size=1.5, mode="line")
        self.qs_chart.figure.tight_layout()
        self.qs_chart.canvas.draw()

    def replot_exp_results(self, color="gray"):
        """Replot all loaded measurement data"""
        for ed in self.exp_data:
            self.plot(ed[0], ed[1], color, size=0.5, mode="scatter")
        self.qs_chart.figure.tight_layout()
        self.qs_chart.canvas.draw()

    def plot(self, x, y, color, mode, size):
        """Add a curve to the quicksim plot"""
        xlabel = "delay time [ns]"
        ylabel = "TRPL"
        scale = "log"
        mc_plot.sim_plot(self.qs_axes, x, y, xlabel,
                         ylabel, scale, color, size=size, mode=mode)

    def redraw(self, *args) -> None:
        """Callback for scale_f change"""
        if not self.qs_finished:
            return
        self.clear()
        self.replot_exp_results()
        if self.sim_vis.get():
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
        self.load_button.configure(state=tk.NORMAL)
        self.export_button.configure(state=tk.NORMAL)
        self.qs_finished = True

    def on_close(self):
        """
        Re-enable the simulate button
        Stop query_quicksim if still running
        """
        self.toplevel.destroy()
        self.window.mini_panel.widgets["quicksim button"].configure(state=tk.NORMAL)
        self.window.mini_panel.widgets["load button"].configure(state=tk.NORMAL)
        self.is_open = False
