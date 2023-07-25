from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog
from types import FunctionType
import tkinter as tk
import numpy as np
import os
import sys
sys.path.append("..")

import sim_utils
import mc_plot
import pickle


def rgb(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


events = {"key": {"escape": "<Escape>", "enter": "<Return>"},
          "click": {"left": "<Button-1>", "right": "<Button-3>"}}

PICKLE_FILE_LOCATION = "../output/TEST_REAL_STAUB"
APPLICATION_NAME = "MCMC Visualization"
WHITE = rgb(255, 255, 255)
LIGHT_GREY = rgb(191, 191, 191)
GREY = rgb(127, 127, 127)
DARK_GREY = rgb(63, 63, 63)
BLACK = rgb(0, 0, 0)
RED = rgb(127, 0, 0)
GREEN = rgb(0, 127, 0)

MENU_KWARGS = {"width": 10, "background": BLACK, "highlightbackground": BLACK, "foreground": WHITE}
LABEL_KWARGS = {"width": 14, "background": LIGHT_GREY}

DEFAULT_HIST_BINS = 96


class Window:
    """ The main GUI object"""

    class Popup:
        """
        For popups?
        TODO: Is this class needed? 
        """
        def __init__(self, master: tk.Tk, width: int, height: int, text: str,
                     colors: tuple[str, str]) -> None:
            self.widget = tk.Toplevel(master=master)
            x_offset = (self.widget.winfo_screenwidth() - width) // 2
            y_offset = (self.widget.winfo_screenheight() - height) // 2
            self.widget.geometry(f"{width}x{height}+{x_offset}+{y_offset}")
            self.widget.resizable(False, False)
            self.widget.title("")
            label = tk.Label(master=self.widget, width=width, height=height, text=text,
                             background=colors[0], foreground=colors[1])
            label.pack()
            self.widget.grab_set()
            self.widget.bind(events["click"]["left"], lambda code: self.widget.destroy())

    class Panel:
        """ Creates the frames for 1) the plot, 2) the plot options, 3) the import/export buttons, etc..."""
        def __init__(self, master: tk.Tk, width: int, height: int, color: str) -> None:
            self.widget = tk.Frame(master=master, width=width, height=height,
                                   background=color, border=4, relief="raised")
            self.states = {"blank": list[tuple[tk.Widget, dict[str, int | str]]]()}
            self.state = "blank"
            self.widgets = dict[str, tk.Widget]()
            self.variables = dict[str, tk.Variable]()

        def place(self, x: int, y: int) -> None:
            self.widget.place(x=x, y=y)

        def addstate(self, state: str, widgets: list[tuple[tk.Widget, dict[str, int | str]]]) -> None:
            self.states[state] = widgets

        def loadstate(self, state: str) -> None:
            if state == self.state:
                return
            for widget, placement in self.states[self.state]:
                widget.place_forget()
            for widget, placement in self.states[state]:
                widget.place(**placement) # type: ignore
            self.state = state


    class Chart:
        """ tk embedded matplotlib Figure """
        def __init__(self, master: tk.Tk, width: int, height: int) -> None:
            self.figure = Figure(figsize=(4, 4))
            self.canvas = FigureCanvasTkAgg(master=master, figure=self.figure)
            self.widget = self.canvas.get_tk_widget()
            self.widget.configure(width=width, height=height, border=4, relief="sunken")
            self.can_save = False

        def place(self, x: int, y: int) -> None:
            self.widget.place(x=x, y=y)

    def __init__(self, width: int, height: int, title: str) -> None:
        self.widget = tk.Tk()
        x_offset = (self.widget.winfo_screenwidth() - width) // 2
        y_offset = (self.widget.winfo_screenheight() - height) // 2
        self.widget.geometry(f"{width}x{height}+{x_offset}+{y_offset}")
        self.widget.resizable(False, False)
        self.widget.title(title)
        self.widget.option_add("*tearOff", False)
        self.chart = self.Chart(self.widget, 600, 600)

        # Stores all MCMC states - self.data[fname][param_name][accepted]
        # param_name e.g. p0, mu_n, mu_p
        # accepted - 0 for all proposed states, 1 for only accepted states
        self.data = dict[str, dict[str, dict[bool, np.ndarray]]]()
        self.file_names = dict[str, tk.IntVar]()

        self.chart.place(0, 0)
        self.side_panel = self.Panel(self.widget, 400, 480, GREY)
        self.side_panel.place(600, 0)
        self.mini_panel = self.Panel(self.widget, 400, 120, DARK_GREY)
        self.mini_panel.place(600, 480)
        self.base_panel = self.Panel(self.widget, 1000, 200, GREY)
        self.base_panel.place(0, 600)

        # Status box - use self.status() to add messages
        self.status_msg = list[str]()
        self.base_panel.variables["status_msg"] = tk.StringVar(value="")
        data_label = tk.Label(master=self.base_panel.widget, textvariable=self.base_panel.variables["status_msg"],
                              width=138, height=11,
                              background=LIGHT_GREY, relief="sunken", border=2, anchor="nw", justify="left")
        data_label.place(x=10, y=10)
        self.base_panel.widgets["data label"] = data_label
        self.status("Use Load File to select a file", clear=True)
        self.populate_mini_panel()
        self.populate_side_panel()
        self.mount_side_panel_states()

    def populate_mini_panel(self) -> None:
        """ Build the small panel that loads data files and refreshes the plot. """
        # Plot type selection
        self.chart_type = tk.StringVar(value="select")
        self.mini_panel.widgets["chart menu"] = tk.OptionMenu(self.mini_panel.widget,
                                                              self.chart_type, "select")
        menu: tk.Menu = self.mini_panel.widgets["chart menu"]["menu"]
        menu.delete(0)
        menu.add_checkbutton(label="1D Trace Plot", onvalue="1D Trace Plot", offvalue="1D Trace Plot",
                             variable=self.chart_type, command=self.chartselect)
        menu.add_checkbutton(label="2D Trace Plot", onvalue="2D Trace Plot", offvalue="2D Trace Plot",
                             variable=self.chart_type, command=self.chartselect)
        menu.add_checkbutton(label="1D Histogram", onvalue="1D Histogram", offvalue="1D Histogram",
                             variable=self.chart_type, command=self.chartselect)
        menu.add_checkbutton(label="2D Histogram", onvalue="2D Histogram", offvalue="2D Histogram",
                             variable=self.chart_type, command=self.chartselect)
        self.mini_panel.widgets["chart menu"].configure(**MENU_KWARGS, state=tk.DISABLED)
        self.mini_panel.widgets["chart menu"].place(x=20, y=40, anchor="sw")

        # Opens file dialog
        load_button = tk.Button(master=self.mini_panel.widget, width=10, text="Load File",
                                background=BLACK, foreground=WHITE, command=self.loadfile, border=4)
        load_button.place(x=200, y=40, anchor="s")
        self.mini_panel.widgets["load button"] = load_button

        # Export
        export_button = tk.Button(master=self.mini_panel.widget, width=10, text="Export",
                                  background=BLACK, foreground=WHITE, command=self.export, border=4)
        export_button.place(x=200, y=100, anchor="s")
        self.mini_panel.widgets["export button"] = load_button

        # Refreshes the plot
        graph_button = tk.Button(master=self.mini_panel.widget, width=10, text="Graph",
                                 background=BLACK, foreground=WHITE, command=self.drawchart, border=4)
        graph_button.place(x=380, y=40, anchor="se")
        graph_button.configure(state=tk.DISABLED)
        self.mini_panel.widgets["graph button"] = graph_button

    def populate_side_panel(self) -> None:
        """ Build the plot control panel to the right of the plotting frame. """
        widgets = self.side_panel.widgets
        variables = self.side_panel.variables
        panel = self.side_panel.widget

        # Text labels
        widgets["x_axis_label"] = tk.Label(master=panel, text="X Axis", **LABEL_KWARGS)
        widgets["y_axis_label"] = tk.Label(master=panel, text="Y Axis", **LABEL_KWARGS)
        widgets["scale_label"] = tk.Label(master=panel, text="Axis Scale", **LABEL_KWARGS)
        widgets["accept_label"] = tk.Label(master=panel, text="Filter", **LABEL_KWARGS)
        widgets["hori_marker_label"] = tk.Label(master=panel,
                                                text="Horizontal Line", **LABEL_KWARGS)
        widgets["equi_label"] = tk.Label(master=panel, text="Equilibration Period", **LABEL_KWARGS)
        widgets["num_bins_label"] = tk.Label(master=panel, text="Bins", **LABEL_KWARGS)

        # User select menus
        variables["variable_1"] = tk.StringVar(value="select")
        variables["variable_2"] = tk.StringVar(value="select")
        variables["accepted"] = tk.StringVar(value="Accepted")
        variables["scale"] = tk.StringVar(value="Linear")
        widgets["variable 1"] = tk.OptionMenu(panel, variables["variable_1"], "")
        widgets["variable 2"] = tk.OptionMenu(panel, variables["variable_2"], "")
        widgets["scale"] = tk.OptionMenu(panel, variables["scale"], "")
        widgets["accepted"] = tk.OptionMenu(panel, variables["accepted"], "")
        widgets["chain_vis"] = tk.Button(panel, text="Select Chains",
                                         command=self.do_select_chain_popup,
                                         width=13, border=4, background=BLACK, foreground=WHITE)

        widgets["variable 1"].configure(**MENU_KWARGS)
        widgets["variable 2"].configure(**MENU_KWARGS)
        widgets["scale"].configure(**MENU_KWARGS)
        widgets["accepted"].configure(**MENU_KWARGS)

        # Add fixed items to specific OptionMenus
        menu: tk.Menu = widgets["accepted"]["menu"]
        menu.delete(0)
        menu.add_checkbutton(label="Accepted", onvalue="Accepted", offvalue="Accepted",
                             variable=variables["accepted"])
        menu.add_checkbutton(label="All Proposed", onvalue="All Proposed", offvalue="All Proposed",
                             variable=variables["accepted"])
        variables["accepted"].trace("w", self.redraw)

        menu: tk.Menu = widgets["scale"]["menu"]
        menu.delete(0)
        menu.add_checkbutton(label="Linear", onvalue="Linear", offvalue="Linear", variable=variables["scale"])
        menu.add_checkbutton(label="Logarithmic", onvalue="Logarithmic",
                             offvalue="Logarithmic", variable=variables["scale"])
        variables["scale"].trace("w", self.redraw)

        # Entry for horizontal marker
        variables["hori_marker"] = tk.StringVar()
        widgets["hori_marker_entry"] = tk.Entry(master=panel, width=16, border=3, textvariable=variables["hori_marker"])
        widgets["hori_marker_entry"].bind("<FocusOut>", self.redraw)

        # Entry to designate initial equilibration period (of states to discard from the statistics)
        variables["equi"] = tk.StringVar()
        widgets["equi_entry"] = tk.Entry(master=panel, width=16, border=3, textvariable=variables["equi"])
        widgets["equi_entry"].bind("<FocusOut>", self.redraw)

        # Entry for number of bins
        variables["bins"] = tk.StringVar()
        variables["bins"].set(str(DEFAULT_HIST_BINS))
        widgets["num_bins_entry"] = tk.Entry(master=panel, width=16, border=3, textvariable=variables["bins"])
        widgets["num_bins_entry"].bind("<FocusOut>", self.redraw)

    def mount_side_panel_states(self):
        widgets = self.side_panel.widgets

        # TODO: More descriptive way to index these
        locations = [{"x": 20, "y": 20, "anchor": "nw"},
                     {"x": 200, "y": 20, "anchor": "n"},
                     {"x": 380, "y": 20, "anchor": "ne"},
                     {"x": 20, "y": 48, "anchor": "nw"},

                     {"x": 200, "y": 48, "anchor": "n"},
                     {"x": 380, "y": 48, "anchor": "ne"},
                     {"x": 20, "y": 88, "anchor": "nw"},
                     {"x": 200, "y": 88, "anchor": "n"},

                     {"x": 380, "y": 88, "anchor": "ne"},
                     {"x": 20, "y": 116, "anchor": "nw"},
                     {"x": 200, "y": 116, "anchor": "n"},
                     {"x": 380, "y": 116, "anchor": "ne"},

                     {"x": 20, "y": 184, "anchor": "nw"},
                     ]

        self.side_panel.addstate("1D Trace Plot", [(widgets["x_axis_label"], locations[0]),
                                                   (widgets["accept_label"], locations[1]),
                                                   (widgets["scale_label"], locations[2]),
                                                   (widgets["variable 1"], locations[3]),
                                                   (widgets["accepted"], locations[4]),
                                                   (widgets["scale"], locations[5]),
                                                   (widgets["hori_marker_label"], locations[6]),
                                                   (widgets["hori_marker_entry"], locations[9]),
                                                   (widgets["equi_label"], locations[7]),
                                                   (widgets["equi_entry"], locations[10]),
                                                   (widgets["chain_vis"], locations[12])]
                                 )

        self.side_panel.addstate("2D Trace Plot", [(widgets["x_axis_label"], locations[0]),
                                                   (widgets["y_axis_label"], locations[1]),
                                                   (widgets["scale_label"], locations[2]),
                                                   (widgets["variable 1"], locations[3]),
                                                   (widgets["variable 2"], locations[4]),
                                                   (widgets["scale"], locations[5]),
                                                   (widgets["equi_label"], locations[7]),
                                                   (widgets["equi_entry"], locations[10]),
                                                   (widgets["chain_vis"], locations[12])]
                                 )

        self.side_panel.addstate("1D Histogram", [(widgets["x_axis_label"], locations[0]),
                                                  (widgets["scale_label"], locations[1]),
                                                  (widgets["variable 1"], locations[3]),
                                                  (widgets["scale"], locations[4]),
                                                  (widgets["chain_vis"], locations[12]),
                                                  (widgets["equi_label"], locations[7]),
                                                  (widgets["equi_entry"], locations[10]),
                                                  (widgets["num_bins_label"], locations[8]),
                                                  (widgets["num_bins_entry"], locations[11])]
                                 )

        self.side_panel.addstate("2D Histogram", [(widgets["x_axis_label"], locations[0]),
                                                  (widgets["y_axis_label"], locations[1]),
                                                  (widgets["scale_label"], locations[2]),
                                                  (widgets["variable 1"], locations[3]),
                                                  (widgets["variable 2"], locations[4]),
                                                  (widgets["scale"], locations[5]),
                                                  (widgets["chain_vis"], locations[12]),
                                                  (widgets["equi_label"], locations[7]),
                                                  (widgets["equi_entry"], locations[10]),
                                                  (widgets["num_bins_label"], locations[8]),
                                                  (widgets["num_bins_entry"], locations[11])]
                                 )

    def do_select_chain_popup(self) -> None:
        toplevel = tk.Toplevel(self.side_panel.widget)
        toplevel.configure(**{"background": LIGHT_GREY})
        tk.Label(toplevel, text="Display:", background=LIGHT_GREY).grid(row=0, column=0)
        for i, file_name in enumerate(self.file_names):
            tk.Checkbutton(toplevel, text=os.path.basename(file_name),
                           variable=self.file_names[file_name],
                           onvalue=1, offvalue=0, background=LIGHT_GREY).grid(row=i+1, column=0)

    def mainloop(self) -> None:
        self.widget.mainloop()

    def bind(self, event: str, command: FunctionType) -> None:
        self.widget.bind(event, command)

    def status(self, msg: str, clear=False):
        """Append a new message to the status panel"""
        if clear:
            self.status_msg = list[str]()

        self.status_msg.append(msg)
        self.base_panel.variables["status_msg"].set("\n".join(self.status_msg))

    def loadfile(self) -> None:
        file_names = filedialog.askopenfilenames(filetypes=[("Pickle File", "*.pik")],
                                                 title="Select File(s)", initialdir=PICKLE_FILE_LOCATION)
        if file_names == "":
            return

        for file_name in file_names:
            self.status(f"Loaded file {file_name}")

        self.widget.title(f"{APPLICATION_NAME} - {file_names}")
        self.data.clear()

        for file_name in file_names:
            self.data[file_name] = {}
            with open(file_name, "rb") as rfile:
                metrostate: sim_utils.MetroState = pickle.load(rfile)

            try:
                for key in metrostate.param_info["names"]:
                    if metrostate.param_info["active"][key]:
                        states = getattr(metrostate.H, key)
                        # Always downcast to 1D
                        if states.ndim == 2 and states.shape[0] == 1:
                            states = states[0]
                        elif states.ndim == 1:
                            pass
                        else:
                            raise ValueError("Invalid chain states format - "
                                             "must be 1D or 2D of size (1, num_states)")

                        mean_states = getattr(metrostate.H, f"mean_{key}")
                        if mean_states.ndim == 2 and mean_states.shape[0] == 1:
                            mean_states = mean_states[0]
                        elif mean_states.ndim == 1:
                            pass
                        else:
                            raise ValueError("Invalid chain states format - "
                                             "must be 1D or 2D of size (1, num_states)")

                        self.data[file_name][key] = {False: states,
                                                     True: mean_states}
            except ValueError as err:
                self.status(f"Error: {err}")
                continue

        self.file_names = {file_name: tk.IntVar(value=1) for file_name in file_names}
        for file_name in self.file_names:
            self.file_names[file_name].trace("w", self.redraw)

        # Generate a button for each parameter
        self.mini_panel.widgets["chart menu"].configure(state=tk.NORMAL) # type: ignore
        self.chart_type.set("select")
        self.mini_panel.widgets["graph button"].configure(state=tk.DISABLED) # type: ignore
        menu: tk.Menu = self.side_panel.widgets["variable 1"]["menu"]
        self.side_panel.variables["variable_1"].set("select")
        menu.delete(0, tk.END)
        for key in self.data[file_names[0]]: # TODO: Require all file_names have same set of keys, or track only unique keys
            menu.add_checkbutton(label=key, onvalue=key, offvalue=key, variable=self.side_panel.variables["variable_1"])
        self.side_panel.variables["variable_1"].trace("w", self.redraw)
        menu: tk.Menu = self.side_panel.widgets["variable 2"]["menu"]
        self.side_panel.variables["variable_2"].set("select")
        menu.delete(0, tk.END)
        for key in self.data[file_names[0]]:
            menu.add_checkbutton(label=key, onvalue=key, offvalue=key, variable=self.side_panel.variables["variable_2"])
        self.side_panel.variables["variable_2"].trace("w", self.redraw)

    def chartselect(self) -> None:
        """ Refresh on choosing a new type of plot """
        self.side_panel.loadstate(self.chart_type.get())
        self.mini_panel.widgets["graph button"].configure(state=tk.NORMAL) # type: ignore
        self.chart.figure.clear()
        self.chart.canvas.draw()

    def redraw(self, *args) -> None:
        """
        Callback for drawchart()

        Updates whenever the values of certain checkbuttons are changed
        """
        self.drawchart()

    def drawchart(self) -> None:
        self.chart.figure.clear()
        match key := self.side_panel.state:
            case "1D Trace Plot":
                value = self.side_panel.variables["variable_1"].get()
                accepted = self.side_panel.variables["accepted"].get()
                scale = self.side_panel.variables["scale"].get()
                hline = self.side_panel.variables["hori_marker"].get()
                equi = self.side_panel.variables["equi"].get()
                try:
                    if "," in hline:
                        hline = tuple(map(float, hline.split(",")))
                    else:
                        hline = (float(hline),)
                except ValueError:
                    hline = tuple()

                try:
                    equi = (int(equi),)
                except ValueError:
                    equi = tuple()

                if value == "select":
                    return
                if accepted == "Accepted":
                    accepted = True
                    title = f"Accepted {value}"
                else:
                    accepted = False
                    title = f"Raw {value}"
                if scale == "Logarithmic":
                    scale = "log"
                else:
                    scale = "linear"
                axes = self.chart.figure.add_subplot()
                for file_name in self.file_names:
                    if self.file_names[file_name].get() == 0: # This value display disabled
                        continue
                    mc_plot.traceplot1d(axes, self.data[file_name][value][accepted],
                                        title, scale, hline, equi)
            case "2D Trace Plot":
                x_val = self.side_panel.variables["variable_1"].get()
                y_val = self.side_panel.variables["variable_2"].get()
                accepted = self.side_panel.variables["accepted"].get()
                scale = self.side_panel.variables["scale"].get()
                equi = self.side_panel.variables["equi"].get()

                try:
                    equi = int(equi)
                    equi = max(0, equi)
                except ValueError:
                    equi = 0

                if x_val == "select" or y_val == "select":
                    return
                if scale == "Logarithmic":
                    scale = "log"
                else:
                    scale = "linear"

                axes = self.chart.figure.add_subplot()
                for file_name in self.file_names:
                    if self.file_names[file_name].get() == 0: # This value display disabled
                        continue
                    mc_plot.traceplot2d(axes, self.data[file_name][x_val][True][equi:],
                                        self.data[file_name][y_val][True][equi:],
                                        x_val, y_val, scale)
            case "1D Histogram":
                value = self.side_panel.variables["variable_1"].get()
                accepted = self.side_panel.variables["accepted"].get()
                scale = self.side_panel.variables["scale"].get()
                bins = self.side_panel.variables["bins"].get()
                equi = self.side_panel.variables["equi"].get()
                try:
                    bins = int(bins)
                except ValueError:
                    bins = DEFAULT_HIST_BINS

                try:
                    equi = int(equi)
                    equi = max(0, equi)
                except ValueError:
                    equi = 0

                if value == "select":
                    return
                if scale == "Logarithmic":
                    scale = "log"
                else:
                    scale = "linear"

                axes = self.chart.figure.add_subplot()
                for file_name in self.file_names:
                    if self.file_names[file_name].get() == 0: # This value display disabled
                        continue
                    mc_plot.histogram1d(axes, self.data[file_name][value][True][equi:],
                                        f"Accepted {value}", scale, bins)
            case "2D Histogram":
                x_val = self.side_panel.variables["variable_1"].get()
                y_val = self.side_panel.variables["variable_2"].get()
                accepted = self.side_panel.variables["accepted"].get()
                scale = self.side_panel.variables["scale"].get()
                bins = self.side_panel.variables["bins"].get()
                equi = self.side_panel.variables["equi"].get()
                try:
                    bins = int(bins)
                except ValueError:
                    bins = DEFAULT_HIST_BINS

                try:
                    equi = int(equi)
                    equi = max(0, equi)
                except ValueError:
                    equi = 0

                if x_val == "select" or y_val == "select":
                    return
                if scale == "Logarithmic":
                    scale = "log"
                else:
                    scale = "linear"

                axes = self.chart.figure.add_subplot()
                for file_name in self.file_names:
                    if self.file_names[file_name].get() == 0: # This value display disabled
                        continue
                    mc_plot.histogram2d(axes, self.data[file_name][x_val][True][equi:],
                                        self.data[file_name][y_val][True][equi:],
                                        x_val, y_val, scale, bins)

                    # colorbar = axes.imshow(hist2d, cmap="Blues")
                    # self.chart.figure.colorbar(colorbar, ax=axes, fraction=0.04)

        self.chart.figure.tight_layout()
        self.chart.canvas.draw()

    def export(self) -> None:
        """ Export currently plotted values as .csv or .npy """
        match key := self.side_panel.state:
            case "1D Trace Plot":
                value = self.side_panel.variables["variable_1"].get()
                accepted = self.side_panel.variables["accepted"].get() == "Accepted"
                equi = self.side_panel.variables["equi"].get()

                try:
                    equi = int(equi)
                    equi = max(0, equi)
                except ValueError:
                    equi = 0

                # One output per chain
                for file_name in self.file_names:
                    # Reasons to not export a file
                    if self.file_names[file_name].get() == 0: # This value display disabled
                        continue
                    if value not in self.data[file_name]:
                        continue

                    out_name = filedialog.asksaveasfilename(filetypes=[("binary", "*.npy"),
                                                                       ("Text", "*.csv")],
                                                            defaultextension=".csv",
                                                            title=f"{os.path.basename(file_name)} - Save as",
                                                            initialdir=PICKLE_FILE_LOCATION)
                    if out_name == "":
                        continue

                    if out_name.endswith(".npy"):
                        out_format = "npy"
                    elif out_name.endswith(".csv"):
                        out_format = "csv"
                    else:
                        raise ValueError("Invalid output file extension - must be .npy or .csv")

                    # (N x 2) array - (iter #, vals)
                    vals = self.data[file_name][value][accepted][equi:]

                    if out_format == "npy":
                        np.save(out_name, np.vstack((np.arange(len(vals)) + equi, vals)).T)
                    elif out_format == "csv":
                        np.savetxt(out_name, np.vstack((np.arange(len(vals)) + equi, vals)).T, delimiter=",",
                                   header=f"N,{value}")
                    else:
                        continue
                    self.status(f"Export complete - {out_name}")

            case "2D Trace Plot":
                x_val = self.side_panel.variables["variable_1"].get()
                y_val = self.side_panel.variables["variable_2"].get()
                equi = self.side_panel.variables["equi"].get()

                try:
                    equi = int(equi)
                    equi = max(0, equi)
                except ValueError:
                    equi = 0

                for file_name in self.file_names:
                    # Reasons to not export a file
                    if self.file_names[file_name].get() == 0: # This value display disabled
                        continue
                    if x_val not in self.data[file_name]:
                        continue
                    if y_val not in self.data[file_name]:
                        continue

                    out_name = filedialog.asksaveasfilename(filetypes=[("binary", "*.npy"),
                                                                       ("Text", "*.csv")],
                                                            defaultextension=".csv",
                                                            title=f"{os.path.basename(file_name)} - Save as",
                                                            initialdir=PICKLE_FILE_LOCATION)
                    if out_name == "":
                        continue

                    if out_name.endswith(".npy"):
                        out_format = "npy"
                    elif out_name.endswith(".csv"):
                        out_format = "csv"
                    else:
                        raise ValueError("Invalid output file extension - must be .npy or .csv")

                    vals_x = self.data[file_name][x_val][True][equi:]
                    vals_y = self.data[file_name][y_val][True][equi:]

                    # (N x 3) array - (iter #, vals_x, vals_y)
                    if out_format == "npy":
                        np.save(out_name, np.vstack((np.arange(len(vals_x)) + equi, vals_x, vals_y)).T)
                    elif out_format == "csv":
                        np.savetxt(out_name, np.vstack((np.arange(len(vals_x)) + equi, vals_x, vals_y)).T, delimiter=",",
                                   header=f"N,{x_val},{y_val}")
                    else:
                        continue
                    self.status(f"Export complete - {out_name}")

            case "1D Histogram":
                value = self.side_panel.variables["variable_1"].get()
                bins = self.side_panel.variables["bins"].get()
                equi = self.side_panel.variables["equi"].get()

                try:
                    equi = int(equi)
                    equi = max(0, equi)
                except ValueError:
                    equi = 0

                try:
                    bins = int(bins)
                except ValueError:
                    bins = DEFAULT_HIST_BINS

                for file_name in self.file_names:
                    if self.file_names[file_name].get() == 0: # This value display disabled
                        continue
                    if value not in self.data[file_name]:
                        continue

                    out_name = filedialog.asksaveasfilename(filetypes=[("binary", "*.npy"),
                                                                       ("Text", "*.csv")],
                                                            defaultextension=".csv",
                                                            title=f"{os.path.basename(file_name)} - Save as",
                                                            initialdir=PICKLE_FILE_LOCATION)
                    if out_name == "":
                        continue

                    if out_name.endswith(".npy"):
                        out_format = "npy"
                    elif out_name.endswith(".csv"):
                        out_format = "csv"
                    else:
                        raise ValueError("Invalid output file extension - must be .npy or .csv")

                    # (b x 2 array) - (bin centres, freq)
                    # Use a bar plot to regenerate the histogram shown in the GUI
                    vals = self.data[file_name][value][True][equi:]
                    freq, bin_centres = np.histogram(vals, bins)
                    bin_centres = (bin_centres + np.roll(bin_centres, -1))[:-1] / 2
                    if out_format == "npy":
                        np.save(out_name, np.vstack((bin_centres, freq)).T)
                    elif out_format == "csv":
                        np.savetxt(out_name, np.vstack((bin_centres, freq)).T, delimiter=",",
                                   header="bin_centre,freq")
                    else:
                        continue
                    self.status(f"Export complete - {out_name}")

            case "2D Histogram":
                x_val = self.side_panel.variables["variable_1"].get()
                y_val = self.side_panel.variables["variable_2"].get()
                bins = self.side_panel.variables["bins"].get()
                equi = self.side_panel.variables["equi"].get()

                try:
                    equi = int(equi)
                    equi = max(0, equi)
                except ValueError:
                    equi = 0

                try:
                    bins = int(bins)
                except ValueError:
                    bins = DEFAULT_HIST_BINS

                for file_name in self.file_names:
                    # Reasons to not export a file
                    if self.file_names[file_name].get() == 0: # This value display disabled
                        continue
                    if x_val not in self.data[file_name]:
                        continue
                    if y_val not in self.data[file_name]:
                        continue

                    out_name = filedialog.asksaveasfilename(filetypes=[("binary", "*.npy"),
                                                                       ("Text", "*.csv")],
                                                            defaultextension=".csv",
                                                            title=f"{os.path.basename(file_name)} - Save as",
                                                            initialdir=PICKLE_FILE_LOCATION)
                    if out_name == "":
                        continue

                    if out_name.endswith(".npy"):
                        out_format = "npy"
                    elif out_name.endswith(".csv"):
                        out_format = "csv"
                    else:
                        raise ValueError("Invalid output file extension - must be .npy or .csv")

                    vals_x = self.data[file_name][x_val][True][equi:]
                    vals_y = self.data[file_name][y_val][True][equi:]
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
                        np.save(out_name, freq_matrix)
                    elif out_format == "csv":
                        np.savetxt(out_name, freq_matrix, delimiter=",")
                    else:
                        continue                    
                    self.status(f"Export complete - {out_name}")

window = Window(1000, 800, APPLICATION_NAME)
window.bind(events["key"]["escape"], sys.exit) # type: ignore

window.mainloop()
