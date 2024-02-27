"""
Second of a two-part tkinter GUI construction - this adds functionality
to a skeleton provided by tkgui.py
"""
import pickle
import sys
import os
from datetime import datetime
import multiprocessing
import tkinter as tk
from tkinter import filedialog
from types import FunctionType
from queue import Empty
from functools import partial
import numpy as np
sys.path.append("..")

from tkgui import TkGUI
from quicksim_result_popup import QuicksimResultPopup
from quicksim_entry_popup import QuicksimEntryPopup
from activate_chain_popup import ActivateChainPopup
import sim_utils
import mc_plot
from quicksim import QuicksimManager
from secondary_parameters import SecondaryParameters

from gui_colors import PLOT_COLOR_CYCLE

events = {"key": {"escape": "<Escape>", "enter": "<Return>"},
          "click": {"left": "<Button-1>", "right": "<Button-3>"}}

PICKLE_FILE_LOCATION = "../output/TEST_REAL_STAUB"
DEFAULT_HIST_BINS = 96
ACC_BIN_SIZE = 100
DEFAULT_THICKNESS = 2000
MAX_STATUS_MSGS = 20

class TracedIntVar(tk.IntVar):
    """ Extension of tkinter IntVar - records ID of its trace function"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trace_id = -1

class Chain():
    """ Attributes for each loaded MCMC chain """

    def __init__(self, fname):
        self.fname = fname
        self.visible = TracedIntVar(value=1)
        self.active_sampled = {}
        self.param_names = []

        # Stores all MCMC states - self.data[param_name]
        # param_name e.g. p0, mu_n, mu_p
        self.data = dict[str, np.ndarray]()

    def is_visible(self):
        return self.visible.get()

class Window(TkGUI):
    """ The main GUI object"""
    qsr_popup: QuicksimResultPopup
    qse_popup: QuicksimEntryPopup
    ac_popup: ActivateChainPopup

    def __init__(self, width: int, height: int, title: str) -> None:
        super().__init__(width, height, title)
        self.set_mini_panel_commands()
        self.set_side_panel_commands()

        # List of loaded MCMC chains
        self.chains = list[Chain]()
        self.n_files = 0

        # List of additional variables needed for simulations
        self.ext_variables = ["thickness", "nx", "final_time", "nt",
                              "fluence", "absp", "direction", "wavelength"]

        self.q = multiprocessing.Queue()
        self.qsm = QuicksimManager(self, self.q)

        self.sp = SecondaryParameters()
        self.status("Use Load File to select a file", clear=True)

    def set_mini_panel_commands(self):
        """Add functionality to mini_panel widgets"""
        widgets = self.mini_panel.widgets
        menu = widgets["chart menu"]["menu"]
        menu.add_checkbutton(label="1D Trace Plot", onvalue="1D Trace Plot", offvalue="1D Trace Plot",
                             variable=self.chart_type, command=self.chartselect)
        menu.add_checkbutton(label="2D Trace Plot", onvalue="2D Trace Plot", offvalue="2D Trace Plot",
                             variable=self.chart_type, command=self.chartselect)
        menu.add_checkbutton(label="1D Histogram", onvalue="1D Histogram", offvalue="1D Histogram",
                             variable=self.chart_type, command=self.chartselect)
        menu.add_checkbutton(label="2D Histogram", onvalue="2D Histogram", offvalue="2D Histogram",
                             variable=self.chart_type, command=self.chartselect)
        
        widgets["export all"].configure(command=partial(self.export, "all"))
        widgets["load button"].configure(command=self.loadfile)
        widgets["graph button"].configure(command=self.drawchart)
        widgets["quicksim button"].configure(command=self.quicksim)

    def set_side_panel_commands(self):
        """Add functionality to side_panel widgets"""
        widgets = self.side_panel.widgets
        variables = self.side_panel.variables
        widgets["chain_vis"].configure(command=self.do_select_chain_popup)

        variables["combined_hist"].trace("w", self.redraw)
        variables["scale"].trace("w", self.redraw)
        variables["bin_shape"].trace("w", self.redraw)
        widgets["hori_marker_entry"].bind("<FocusOut>", self.redraw)
        widgets["equi_entry"].bind("<FocusOut>", self.redraw)
        widgets["num_bins_entry"].bind("<FocusOut>", self.redraw)
        widgets["thickness"].bind("<FocusOut>", self.redraw)
        widgets["xlim_l"].bind("<FocusOut>", self.redraw)
        widgets["xlim_u"].bind("<FocusOut>", self.redraw)
        widgets["export this"].configure(command=partial(self.export, "this_variable"))
        widgets["calc_diffusion"].configure(command=self.chain_diffusion)

        variables["bins"].set(str(DEFAULT_HIST_BINS))
        variables["thickness"].set(str(DEFAULT_THICKNESS))

    def do_select_chain_popup(self) -> None:
        self.side_panel.widgets["chain_vis"].configure(state=tk.DISABLED) # type: ignore
        self.ac_popup = ActivateChainPopup(self, self.side_panel.widget)

    def get_n_chains(self) -> int:
        """Count how many active chains, as set by to ActivateChainPopup"""
        n_chains = 0
        for chain in self.chains:
            if chain.is_visible():
                n_chains += 1
        return n_chains

    def mainloop(self) -> None:
        self.widget.mainloop()

    def bind(self, event: str, command: FunctionType) -> None:
        self.widget.bind(event, command)

    def status(self, msg: str, clear=False) -> None:
        """Append a new message to the status panel"""
        if clear:
            self.status_msg = list[str]()

        if len(self.status_msg) >= MAX_STATUS_MSGS:
            self.status_msg.pop(0)

        self.status_msg.append(msg)

        self.base_panel.widgets["status_box"].delete(1.0, tk.END)
        self.base_panel.widgets["status_box"].insert(1.0, "\n".join(self.status_msg[::-1]))

    def do_quicksim_entry_popup(self) -> dict[str, str]:
        """Collect quicksim settings"""
        self.qse_popup = QuicksimEntryPopup(self, self.side_panel.widget,
                                            self.ext_variables)
        self.widget.wait_window(self.qse_popup.toplevel)
        return {"model": self.qse_popup.model.get(),
                "meas": self.qse_popup.meas.get(),
                }

    def do_quicksim_result_popup(self, n_chains, n_sims, qse_info) -> None:
        """Show quicksim results"""
        active_chain_inds = [i for i in range(len(self.chains)) if self.chains[i].is_visible()]
        self.qsr_popup = QuicksimResultPopup(self, self.side_panel.widget, n_chains, n_sims,
                                             active_chain_inds, qse_info)

    def query_quicksim(self, expected_num_sims : int) -> None:
        """Periodically check and plot completed quicksims"""
        self.qsr_popup.progress_text.set(f"{self.q.qsize()} of {expected_num_sims} complete")
        self.qsr_popup.progress.set(self.q.qsize())
        if not self.qsr_popup.is_open: # Closing the popup aborts the simulations
            while True:
                try:
                    # Flush the queue
                    self.q.get(timeout=1)
                except Empty:
                    break

            self.qsm.terminate()
            self.status("Sims canceled")
            return

        if self.q.qsize() != expected_num_sims: # Tasks are not finished yet
            self.widget.after(1000, self.query_quicksim, expected_num_sims)
            return

        while True:
            try:
                sim_result = self.q.get(timeout=1)
                self.qsr_popup.sim_results.append(sim_result[:-1])
                message = sim_result[-1]
                if len(message) > 0:
                    self.status(message)
            except Empty:
                pass

            if len(self.qsr_popup.sim_results) == expected_num_sims:
                break

        self.qsm.join()
        self.qsr_popup.finalize()
        self.status("Sims finished")
        self.qsr_popup.toplevel.attributes('-topmost', 'true')
        self.qsr_popup.toplevel.attributes('-topmost', 'false')

    def quicksim(self) -> None:
        """Start a quicksim and periodically check for completion"""
        self.mini_panel.widgets["quicksim button"].configure(state=tk.DISABLED) # type: ignore
        self.mini_panel.widgets["load button"].configure(state=tk.DISABLED)  # type: ignore

        qse_info = self.do_quicksim_entry_popup()

        if not self.qse_popup.continue_:
            self.mini_panel.widgets["quicksim button"].configure(state=tk.NORMAL) # type: ignore
            self.mini_panel.widgets["load button"].configure(state=tk.NORMAL)  # type: ignore
            return

        sim_tasks = {}
        for ev in self.ext_variables:
            sim_tasks[ev] = []
            for i in range(self.qse_popup.n_sims):
                if ev == "nx" or ev == "nt": # Number of steps must be int
                    sim_tasks[ev].append(int(float(self.qse_popup.ext_var[ev][i].get())))
                else:
                    sim_tasks[ev].append(float(self.qse_popup.ext_var[ev][i].get()))

        self.do_quicksim_result_popup(self.get_n_chains(), self.qse_popup.n_sims, qse_info)
        self.qsr_popup.toplevel.attributes('-topmost', 'false')
        self.widget.after(10, self.qsm.quicksim, sim_tasks,
                          self.qse_popup.model.get(), self.qse_popup.meas.get())
        self.widget.after(1000, self.query_quicksim, self.qse_popup.n_sims * self.get_n_chains())

    def loadfile(self) -> None:
        file_names = filedialog.askopenfilenames(filetypes=[("Pickle File", "*.pik")],
                                                 title="Select File(s)", initialdir=PICKLE_FILE_LOCATION)
        if file_names == "":
            return

        for file_name in file_names:
            self.status(f"Loaded file {file_name}")

        self.widget.title(f"{self.application_name} - {file_names}")

        self.chains = []
        self.n_files = len(file_names)
        for file_name in file_names:
            with open(file_name, "rb") as rfile:
                MS_list = pickle.load(rfile)
                # Compatibility prior to ensembles
                if isinstance(MS_list, sim_utils.MetroState):
                    active = MS_list.param_info["active"]
                    names = MS_list.param_info["names"]
                    history = MS_list.H
                    MS_list = [MS_list]
                else:
                    active = MS_list.ensemble_fields["active"]
                    names = MS_list.ensemble_fields["names"]
                    history = MS_list.H
                    try:
                        MS_list = MS_list.unique_fields  # Might have to accomodate outdated MS_list.MS
                    except AttributeError:
                        MS_list = MS_list.MS

            for i in range(len(MS_list)):
                chain = Chain(file_name + f"-{i}")

                chain.active_sampled = active
                chain.param_names = names

                logl = getattr(history, "loglikelihood")
                if logl.ndim == 2:
                    logl = logl[i]
                elif logl.ndim == 1:
                    pass
                else:
                    raise ValueError("Invalid chain states format - "
                                        "must be 1D or 2D of size (1, num_states)")

                chain.data["log likelihood"] = logl[1:]

                accept = getattr(history, "accept")
                if accept.ndim == 2:
                    accept = accept[i]
                elif accept.ndim == 1:
                    pass
                else:
                    raise ValueError("Invalid chain states format - "
                                        "must be 1D or 2D of size (1, num_states)")
                
                bins = np.arange(0, len(accept), int(ACC_BIN_SIZE))
                accepted_subs = np.split(accept, bins)
                num_bins = len(accepted_subs)
                sub_means = np.zeros((num_bins))
                for s, sub in enumerate(accepted_subs):
                    sub_means[s] = np.mean(sub)
                chain.data["accept"] = sub_means

                try:
                    for key in chain.param_names:
                        mean_states = getattr(history, f"mean_{key}")
                        if mean_states.ndim == 2:
                            mean_states = mean_states[i]
                        elif mean_states.ndim == 1:
                            pass
                        else:
                            raise ValueError("Invalid chain states format - "
                                                "must be 1D or 2D of size (1, num_states)")

                        chain.data[key] = mean_states

                    for key in self.sp.func:
                        # TODO: Option to precalculate all of these
                        chain.data[key] = np.zeros(0)
                except ValueError as err:
                    self.status(f"Error: {err}")
                    continue   

                chain.visible.trace_id = chain.visible.trace("w", self.on_active_chain_update)
                self.chains.append(chain)         

        # Generate a button for each parameter
        self.mini_panel.widgets["chart menu"].configure(state=tk.NORMAL) # type: ignore
        self.chart_type.set("select")
        self.mini_panel.widgets["graph button"].configure(state=tk.DISABLED) # type: ignore
        menu: tk.Menu = self.side_panel.widgets["variable 1"]["menu"]
        self.side_panel.variables["variable_1"].set("select")
        menu.delete(0, tk.END)

        # TODO: Require all file_names have same set of keys, or track only unique keys
        for key in self.chains[0].data:
            menu.add_checkbutton(label=key, onvalue=key, offvalue=key,
                                 variable=self.side_panel.variables["variable_1"])

        self.side_panel.variables["variable_1"].trace("w", self.redraw)
        menu: tk.Menu = self.side_panel.widgets["variable 2"]["menu"]
        self.side_panel.variables["variable_2"].set("select")
        menu.delete(0, tk.END)
        for key in self.chains[0].data:
            menu.add_checkbutton(label=key, onvalue=key, offvalue=key,
                                 variable=self.side_panel.variables["variable_2"])

        self.side_panel.variables["variable_2"].trace("w", self.redraw)
        self.mini_panel.widgets["quicksim button"].configure(state=tk.NORMAL) # type: ignore

    def chartselect(self) -> None:
        """ Refresh on choosing a new type of plot """
        self.side_panel.loadstate(self.chart_type.get())
        self.mini_panel.widgets["export all"].configure(state=tk.NORMAL) # type: ignore
        self.mini_panel.widgets["graph button"].configure(state=tk.NORMAL) # type: ignore
        self.chart.figure.clear()
        self.chart.canvas.draw()

    def on_active_chain_update(self, *args) -> None:
        self.redraw()
        if hasattr(self, "qse_popup") and self.qse_popup.is_open:
            self.qse_popup.calc_total_sims()

    def redraw(self, *args) -> None:
        """
        Callback for drawchart()

        Updates whenever the values of certain checkbuttons are changed
        """
        self.drawchart()

    def drawchart(self) -> None:
        """Draw the plot"""
        self.chart.figure.clear()

        # Collect common entries
        x_val = self.side_panel.variables["variable_1"].get()
        scale = self.side_panel.variables["scale"].get()
        equi = self.side_panel.variables["equi"].get()
        thickness = self.side_panel.variables["thickness"].get()
        xlim_l = self.side_panel.variables["xlim_l"].get()
        xlim_u = self.side_panel.variables["xlim_u"].get()

        # Parse common entries
        if x_val == "select":
            return

        if scale == "Log":
            scale = "log"
        elif scale == "Symlog":
            scale = "symlog"
        else:
            scale = "linear"

        try:
            equi = int(equi)
            equi = max(0, equi)
        except ValueError:
            equi = 0

        try:
            xlim_l = float(xlim_l)
        except ValueError:
            xlim_l = None

        try:
            xlim_u = float(xlim_u)
        except ValueError:
            xlim_u = None

        xlim = (xlim_l, xlim_u)

        # Histogram specific entries
        bins = DEFAULT_HIST_BINS
        bin_shape = "linear"
        if "Histogram" in self.side_panel.state:
            bins = self.side_panel.variables["bins"].get()
            try:
                bins = int(bins)
            except ValueError:
                pass
            bin_shape = self.side_panel.variables["bin_shape"].get()

            if bin_shape == "Linear":
                bin_shape = "linear"
            elif bin_shape == "Log":
                bin_shape = "log"


        # 2D specific entries
        y_val = "select"
        if "2D" in self.side_panel.state:
            y_val = self.side_panel.variables["variable_2"].get()
            if y_val == "select":
                return

        axes = self.chart.figure.add_subplot()
        match self.side_panel.state:
            case "1D Trace Plot":
                hline = self.side_panel.variables["hori_marker"].get()
                try:
                    if "," in hline:
                        hline = tuple(map(float, hline.split(",")))
                    else:
                        hline = (float(hline),)
                except ValueError:
                    hline = tuple()

                title = f"{x_val}"

                for i, chain in enumerate(self.chains):

                    if not chain.is_visible():
                        continue
                    color = PLOT_COLOR_CYCLE[i % len(PLOT_COLOR_CYCLE)]

                    y = chain.data[x_val]
                    if (len(y) == 0 or thickness != self.sp.last_thickness.get(x_val, thickness)) and x_val in self.sp.func:
                        # Calculate and cache the secondary parameter
                        try:
                            self.sp.get(chain.data, x_val, thickness)
                        except (ValueError, KeyError) as err:
                            self.status(str(err))
                    mc_plot.traceplot1d(axes, chain.data[x_val],
                                        title, scale, xlim, hline, (equi,), color)
                    
                    if 0 <= equi < len(chain.data[x_val]):
                        if 1e-3 < np.abs(chain.data[x_val][equi]) < 1e6:
                            self.status(f"Chain {i} {x_val}({equi}): {chain.data[x_val][equi]:.3f}")
                        else:
                            self.status(f"Chain {i} {x_val}({equi}): {chain.data[x_val][equi]:.3e}")

            case "2D Trace Plot":
                xy_val = {"x": x_val, "y": y_val}

                for i, chain in enumerate(self.chains):
                    if not chain.is_visible():
                        continue
                    color = PLOT_COLOR_CYCLE[i % len(PLOT_COLOR_CYCLE)]

                    success = {"x": False, "y": False}
                    for s, val in xy_val.items():
                        y =  chain.data[val]
                        if (len(y) == 0 or thickness != self.sp.last_thickness.get(val, thickness)) and val in self.sp.func:
                            try:
                                self.sp.get(chain.data, val, thickness)
                            except (ValueError, KeyError) as err:
                                self.status(str(err))
                                continue
                        success[s] = True

                    if success["x"] and success["y"]: # Successfully obtained data for both params
                        mc_plot.traceplot2d(axes, chain.data[x_val][equi:],
                                            chain.data[y_val][equi:],
                                            x_val, y_val, scale, color)

            case "1D Histogram":
                combined_hist = self.side_panel.variables["combined_hist"].get()

                if combined_hist:
                    vals = np.zeros(0)
                    for chain in self.chains:
                        if not chain.is_visible(): # This value display disabled
                            continue

                        y = chain.data[x_val]
                        if (len(y) == 0 or thickness != self.sp.last_thickness.get(x_val, thickness)) and x_val in self.sp.func:
                            try:
                                self.sp.get(chain.data, x_val, thickness)
                            except (ValueError, KeyError) as err:
                                self.status(str(err))
                                continue

                        vals = np.hstack((vals, chain.data[x_val][equi:]))

                    # Print some statistics
                    
                    if bin_shape == "linear":
                        mean = np.nanmean(vals)
                        stdev = np.nanstd(vals, ddof=1)
                        self.status(f"Mean: {mean:.3e}, stdev: {stdev:.3e}")
                    elif bin_shape == "log":
                        nonzero = vals > 0
                        mean = np.nanmean(np.log10(vals[nonzero]))
                        stdev = np.nanstd(np.log10(vals[nonzero]), ddof=1)
                        self.status(f"Ignored {len(vals) - np.sum(nonzero)} zero values")
                        self.status(f"Log mean: {mean:.3e}, stdev: {stdev:.3e}")
                        self.status(f"Mean: {10 ** mean:.3e}, 1-sigma: ({10 ** (mean - stdev):.3e}, {10 ** (mean + stdev):.3e})")

                    color = PLOT_COLOR_CYCLE[0]
                    mc_plot.histogram1d(axes, vals, f"{x_val}", x_val, scale, bins, bin_shape, color)
                else:
                    for i, chain in enumerate(self.chains):
                        if not chain.is_visible():
                            continue
                        color = PLOT_COLOR_CYCLE[i % len(PLOT_COLOR_CYCLE)]

                        y = chain.data[x_val]
                        if (len(y) == 0 or thickness != self.sp.last_thickness.get(x_val, thickness)) and x_val in self.sp.func:
                            try:
                                self.sp.get(chain.data, x_val, thickness)
                            except (ValueError, KeyError) as err:
                                self.status(str(err))
                                continue

                        mc_plot.histogram1d(axes, chain.data[x_val][equi:],
                                            f"{x_val}", x_val, scale, bins, bin_shape, color)

            case "2D Histogram":
                xy_val = {"x": x_val, "y": y_val}

                # Always combine samples before plotting (essentially combined_hist=True)
                vals_x = np.zeros(0)
                vals_y = np.zeros(0)
                for chain in self.chains:
                    if not chain.is_visible(): # This value display disabled
                        continue

                    success = {"x": False, "y": False}
                    for s, val in xy_val.items():
                        y = chain.data[val]
                        if (len(y) == 0 or thickness != self.sp.last_thickness.get(val, thickness)) and val in self.sp.func:
                            try:
                                self.sp.get(chain.data, val, thickness)
                            except (ValueError, KeyError) as err:
                                self.status(str(err))
                                continue
                        success[s] = True

                    if success["x"] and success["y"]:
                        vals_x = np.hstack((vals_x, chain.data[x_val][equi:]))
                        vals_y = np.hstack((vals_y, chain.data[y_val][equi:]))
                mc_plot.histogram2d(axes, vals_x, vals_y,
                                    x_val, y_val, scale, bins)
                # colorbar = axes.imshow(hist2d, cmap="Blues")
                # self.chart.figure.colorbar(colorbar, ax=axes, fraction=0.04)

        # Record most recently used thickness
        if x_val in self.sp.last_thickness:
            self.sp.last_thickness[x_val] = thickness

        if "2D" in self.side_panel.state and y_val in self.sp.last_thickness:
            self.sp.last_thickness[y_val] = thickness

        self.chart.figure.tight_layout()
        self.chart.canvas.draw()

    def export(self, which) -> None:
        """ Export currently plotted values as .csv or .npy """
        if which == "all":
            # Make an empty folder next to the loaded pickle files
            file_name = self.chains[0].fname
            tstamp = str(datetime.now()).replace(":", "-")
            out_dir = os.path.join(os.path.dirname(file_name), f"export-{tstamp}")
            
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            if len(os.listdir(out_dir)) > 0:
                self.status(f"Error - dir {out_dir} must be empty")
                return
            # One output per chain
            for chain in self.chains:
                # Reasons to not export a file
                if not chain.is_visible(): # This value display disabled
                    continue

                out_name = os.path.basename(chain.fname)
                if out_name.endswith(".pik"):
                    out_name = out_name[:-4]
                out_name += ".csv"

                equi = self.side_panel.variables["equi"].get()
                try:
                    equi = int(equi)
                    equi = max(0, equi)
                except ValueError:
                    equi = 0

                data = np.zeros((0, 0))
                header = []
                for x_val in chain.data:
                    if x_val in self.sp.func:
                        continue
                    if x_val == "log likelihood" or x_val == "accept":
                        continue

                    vals = np.log10(chain.data[x_val][equi:])
                    if len(data) == 0:
                        data = np.vstack((np.arange(len(vals)) + equi, np.array(vals)))
                        header.append("Index")
                    else:
                        data = np.vstack((data, vals))
                    header.append(x_val)

                np.savetxt(os.path.join(out_dir, out_name), data.T, delimiter=",",
                                    header=",".join(header))
            self.status(f"Export complete - {out_dir}")

        elif which == "this_variable":
            # Collect common entries
            x_val = self.side_panel.variables["variable_1"].get()
            equi = self.side_panel.variables["equi"].get()

            # Parse common entries
            if x_val == "select":
                return

            try:
                equi = int(equi)
                equi = max(0, equi)
            except ValueError:
                equi = 0

            # Histogram specific entries
            bins = DEFAULT_HIST_BINS
            if "Histogram" in self.side_panel.state:
                bins = self.side_panel.variables["bins"].get()
                try:
                    bins = int(bins)
                except ValueError:
                    pass

            # 2D specific entries
            y_val = "select"
            if "2D" in self.side_panel.state:
                y_val = self.side_panel.variables["variable_2"].get()
                if y_val == "select":
                    return
                
            # Make an empty folder next to the loaded pickle files
            default_fname = self.chains[0].fname
            tstamp = str(datetime.now()).replace(":", "-")
            out_dir = os.path.join(os.path.dirname(default_fname), f"export-{tstamp}")
            
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            if len(os.listdir(out_dir)) > 0:
                self.status(f"Error - dir {out_dir} must be empty")
                return

            match self.side_panel.state:
                case "1D Trace Plot":
                    # One output per chain
                    for chain in self.chains:
                        # Reasons to not export a file
                        if not chain.is_visible(): # This value display disabled
                            continue
                        if x_val not in chain.data:
                            continue

                        out_name = os.path.basename(chain.fname)
                        if out_name.endswith(".pik"):
                            out_name = out_name[:-4]
                        out_name += ".csv"

                        if out_name.endswith(".npy"):
                            out_format = "npy"
                        elif out_name.endswith(".csv"):
                            out_format = "csv"
                        else:
                            raise ValueError("Invalid output file extension - must be .npy or .csv")

                        # (N x 2) array - (iter #, vals)
                        vals = chain.data[x_val][equi:]

                        if out_format == "npy":
                            np.save(os.path.join(out_dir, out_name), np.vstack((np.arange(len(vals)) + equi, vals)).T)
                        elif out_format == "csv":
                            np.savetxt(os.path.join(out_dir, out_name), np.vstack((np.arange(len(vals)) + equi, vals)).T, delimiter=",",
                                    header=f"N,{x_val}")
                        else:
                            continue
                        self.status(f"Export complete - {os.path.join(out_dir, out_name)}")

                case "2D Trace Plot":
                    for chain in self.chains:
                        # Reasons to not export a file
                        if not chain.is_visible(): # This value display disabled
                            continue
                        if x_val not in chain.data:
                            continue
                        if y_val not in chain.data:
                            continue

                        out_name = os.path.basename(chain.fname)
                        if out_name.endswith(".pik"):
                            out_name = out_name[:-4]
                        out_name += ".csv"

                        if out_name.endswith(".npy"):
                            out_format = "npy"
                        elif out_name.endswith(".csv"):
                            out_format = "csv"
                        else:
                            raise ValueError("Invalid output file extension - must be .npy or .csv")

                        vals_x = chain.data[x_val][equi:]
                        vals_y = chain.data[y_val][equi:]

                        if len(vals_x) == 0:
                            self.status(f"Missing {x_val}")
                            continue
                        if len(vals_y) == 0:
                            self.status(f"Missing {y_val}")
                            continue

                        # (N x 3) array - (iter #, vals_x, vals_y)
                        if out_format == "npy":
                            np.save(os.path.join(out_dir, out_name), np.vstack((np.arange(len(vals_x)) + equi, vals_x, vals_y)).T)
                        elif out_format == "csv":
                            np.savetxt(os.path.join(out_dir, out_name), np.vstack((np.arange(len(vals_x)) + equi, vals_x, vals_y)).T, delimiter=",",
                                    header=f"N,{x_val},{y_val}")
                        else:
                            continue
                        self.status(f"Export complete - {os.path.join(out_dir, out_name)}")

                case "1D Histogram":
                    combined_hist = self.side_panel.variables["combined_hist"].get()
                    if combined_hist:
                        out_name = os.path.basename(default_fname)
                        if out_name.endswith(".pik"):
                            out_name = out_name[:-4]
                        out_name += ".csv"

                        if out_name.endswith(".npy"):
                            out_format = "npy"
                        elif out_name.endswith(".csv"):
                            out_format = "csv"
                        else:
                            raise ValueError("Invalid output file extension - must be .npy or .csv")

                        vals = np.zeros(0)
                        for chain in self.chains:
                            if not chain.is_visible(): # This value display disabled
                                continue
                            if x_val not in chain.data:
                                continue

                            vals = np.hstack((vals, chain.data[x_val][equi:]))

                        freq, bin_centres = np.histogram(vals, bins)
                        bin_centres = (bin_centres + np.roll(bin_centres, -1))[:-1] / 2
                        if out_format == "npy":
                            np.save(os.path.join(out_dir, out_name), np.vstack((bin_centres, freq)).T)
                        elif out_format == "csv":
                            np.savetxt(os.path.join(out_dir, out_name), np.vstack((bin_centres, freq)).T, delimiter=",",
                                    header="bin_centre,freq")

                        self.status(f"Export complete - {os.path.join(out_dir, out_name)}")

                    else:
                        for chain in self.chains:
                            if not chain.is_visible(): # This value display disabled
                                continue
                            if x_val not in chain.data:
                                continue

                            out_name = os.path.basename(chain.fname)
                            if out_name.endswith(".pik"):
                                out_name = out_name[:-4]
                            out_name += ".csv"

                            if out_name.endswith(".npy"):
                                out_format = "npy"
                            elif out_name.endswith(".csv"):
                                out_format = "csv"
                            else:
                                raise ValueError("Invalid output file extension - must be .npy or .csv")

                            # (b x 2 array) - (bin centres, freq)
                            # Use a bar plot to regenerate the histogram shown in the GUI
                            vals = chain.data[x_val][equi:]
                            freq, bin_centres = np.histogram(vals, bins)
                            bin_centres = (bin_centres + np.roll(bin_centres, -1))[:-1] / 2
                            if out_format == "npy":
                                np.save(os.path.join(out_dir, out_name), np.vstack((bin_centres, freq)).T)
                            elif out_format == "csv":
                                np.savetxt(os.path.join(out_dir, out_name), np.vstack((bin_centres, freq)).T, delimiter=",",
                                        header="bin_centre,freq")
                            else:
                                continue
                            self.status(f"Export complete - {os.path.join(out_dir, out_name)}")

                case "2D Histogram":
                    out_name = os.path.basename(default_fname)
                    if out_name.endswith(".pik"):
                        out_name = out_name[:-4]
                    out_name += ".csv"

                    if out_name.endswith(".npy"):
                        out_format = "npy"
                    elif out_name.endswith(".csv"):
                        out_format = "csv"
                    else:
                        raise ValueError("Invalid output file extension - must be .npy or .csv")

                    vals_x = np.zeros(0)
                    vals_y = np.zeros(0)
                    for chain in self.chains:
                        # Reasons to not export a file
                        if not chain.is_visible(): # This value display disabled
                            continue
                        if x_val not in chain.data:
                            continue
                        if y_val not in chain.data:
                            continue

                        vals_x = np.hstack((vals_x, chain.data[x_val][equi:]))
                        vals_y = np.hstack((vals_y, chain.data[y_val][equi:]))

                    if len(vals_x) == 0:
                        self.status(f"Missing {x_val}")
                        return
                    if len(vals_y) == 0:
                        self.status(f"Missing {y_val}")
                        return
                    # (b0+1 x b1+1) array - (freq matrix), one row/col as bin headers
                    freq, bins_x, bins_y = np.histogram2d(vals_x, vals_y, bins)
                    bins_x = (bins_x + np.roll(bins_x, -1))[:-1] / 2
                    bins_y = (bins_y + np.roll(bins_y, -1))[:-1] / 2

                    freq_matrix = np.zeros((bins+1, bins+1))
                    freq_matrix[0, 0] = -1
                    freq_matrix[0, 1:] = bins_x
                    freq_matrix[1:, 0] = bins_y
                    freq_matrix[1:, 1:] = freq

                    if out_format == "npy":
                        np.save(os.path.join(out_dir, out_name), freq_matrix)
                    elif out_format == "csv":
                        np.savetxt(os.path.join(out_dir, out_name), freq_matrix, delimiter=",")

                    self.status(f"Export complete - {os.path.join(out_dir, out_name)}")

    def chain_diffusion(self):
        """Calculate diffusion coefficient of chains"""
        equi = self.side_panel.variables["equi"].get()
        try:
            equi = int(equi)
            equi = max(0, equi)
        except ValueError:
            equi = 0

        for chain in self.chains:
            if not chain.is_visible():
                continue
            num_active = sum(chain.active_sampled.values())
            num_samples = len(chain.data["log likelihood"]) + 1 - equi
            diffusion_coef = 0

            for param in chain.param_names:
                if not chain.active_sampled[param]:
                    continue

                x = chain.data[param][equi:]
                x = np.log10(x)
                x = np.diff(x)
                x = x ** 2
                diffusion_coef += np.sum(x)

            diffusion_coef /= (num_samples * num_active)
            self.status(f"Chain {os.path.basename(chain.fname)} Diffusion coef: {diffusion_coef}")
