from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure, Axes
from tkinter import filedialog
from types import FunctionType
import tkinter as tk
import numpy as np
import os
import sys
sys.path.append("..")

import sim_utils
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


class Plot:
    def traceplot1d(axes: Axes, x_list: np.ndarray, title: str, scale: str, *hline) -> None:
        axes.plot(x_list)
        if len(hline) == 1:
            if min(x_list) < hline and hline < max(x_list):
                axes.hlines(hline[0], 0, len(x_list), colors=BLACK, linestyles="dashed")
        axes.set_title(title)
        axes.set_yscale(scale)
        axes.set_xlabel("n", fontstyle="italic")

    def traceplot2d(axes: Axes, x_list: np.ndarray, y_list: np.ndarray,
                    x_label: str, y_label: str, scale: str) -> None:
        axes.plot(x_list, y_list)
        axes.plot(x_list[0], y_list[0], marker=".", linestyle=" ", color=GREEN,
                  label="Start", markersize=10)
        axes.plot(x_list[-1], y_list[-1], marker=".", linestyle=" ", color=RED,
                  label="End", markersize=10)
        axes.set_xscale(scale)
        axes.set_yscale(scale)
        axes.legend()
        axes.set_xlabel(f"Accepted {x_label}")
        axes.set_ylabel(f"Accepted {y_label}")

    def histogram1d(axes: Axes, x_list: np.ndarray, title: str, scale: str) -> None:
        axes.hist(x_list, 100, edgecolor=BLACK)
        axes.set_yscale(scale)
        axes.set_title(title)

    def histogram2d(axes: Axes, x_list: np.ndarray, y_list: np.ndarray,
                    x_label: str, y_label: str, scale: str) -> None:
        data = axes.hist2d(x_list, y_list, 100, cmap="Blues")[0]
        axes.set_xscale(scale)
        axes.set_yscale(scale)
        axes.set_xlabel(f"Accepted {x_label}")
        axes.set_ylabel(f"Accepted {y_label}")
        return data


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
                widget.place(**placement)
            self.state = state

        def setvar(self, widget_name: str, value: str) -> None:
            name = self.widgets[widget_name]["textvariable"]
            self.widget.setvar(name, value)

        def getvar(self, widget_name: str) -> str:
            name = self.widgets[widget_name]["textvariable"]
            return self.widget.getvar(name)

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
        self.data = dict[str, dict[bool]]()

        self.chart.place(0, 0)
        self.side_panel = self.Panel(self.widget, 400, 480, GREY)
        self.side_panel.place(600, 0)
        self.mini_panel = self.Panel(self.widget, 400, 120, DARK_GREY)
        self.mini_panel.place(600, 480)
        self.base_panel = self.Panel(self.widget, 1000, 200, GREY)
        self.base_panel.place(0, 600)

        # Status box
        self.status_msg = tk.StringVar(value="")
        data_label = tk.Label(master=self.base_panel.widget, textvariable=self.status_msg, width=138, height=11,
                              background=LIGHT_GREY, relief="sunken", border=2, anchor="nw", justify="left")
        data_label.place(x=10, y=10)
        self.base_panel.widgets["data label"] = data_label
        self.base_panel.setvar("data label", "Use Load File to select a file")
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
        panel = self.side_panel.widget

        # Text labels
        widgets["x_axis_label"] = tk.Label(master=panel, text="X Axis", **LABEL_KWARGS)
        widgets["y_axis_label"] = tk.Label(master=panel, text="Y Axis", **LABEL_KWARGS)
        widgets["scale_label"] = tk.Label(master=panel, text="Axis Scale", **LABEL_KWARGS)
        widgets["accept_label"] = tk.Label(master=panel, text="Filter", **LABEL_KWARGS)
        widgets["hori_marker_label"] = tk.Label(master=panel,
                                                text="Horizontal Line", **LABEL_KWARGS)

        # User select menus
        variable_1 = tk.StringVar(value="select")
        variable_2 = tk.StringVar(value="select")
        accepted = tk.StringVar(value="Accepted")
        scale = tk.StringVar(value="Linear")
        widgets["variable 1"] = tk.OptionMenu(panel, variable_1, "")
        widgets["variable 2"] = tk.OptionMenu(panel, variable_2, "")
        widgets["scale"] = tk.OptionMenu(panel, scale, "")
        widgets["accepted"] = tk.OptionMenu(panel, accepted, "")
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
                             variable=accepted)
        menu.add_checkbutton(label="All Proposed", onvalue="All Proposed", offvalue="All Proposed",
                             variable=accepted)

        menu: tk.Menu = widgets["scale"]["menu"]
        menu.delete(0)
        menu.add_checkbutton(label="Linear", onvalue="Linear", offvalue="Linear", variable=scale)
        menu.add_checkbutton(label="Logarithmic", onvalue="Logarithmic",
                             offvalue="Logarithmic", variable=scale)

        # Entry for horizontal marker
        widgets["hori_marker_entry"] = tk.Entry(master=panel, width=16, border=3)

    def mount_side_panel_states(self):
        widgets = self.side_panel.widgets

        # TODO: Make this a window property or something
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

                     {"x": 20, "y": 184, "anchor": "nw"}
                     ]

        self.side_panel.addstate("1D Trace Plot", [(widgets["x_axis_label"], locations[0]),
                                                   (widgets["accept_label"], locations[1]),
                                                   (widgets["scale_label"], locations[2]),
                                                   (widgets["variable 1"], locations[3]),
                                                   (widgets["accepted"], locations[4]),
                                                   (widgets["scale"], locations[5]),
                                                   (widgets["hori_marker_label"], locations[6]),
                                                   (widgets["hori_marker_entry"], locations[9]),
                                                   (widgets["chain_vis"], locations[12])]
                                 )

        self.side_panel.addstate("2D Trace Plot", [(widgets["x_axis_label"], locations[0]),
                                                   (widgets["y_axis_label"], locations[1]),
                                                   (widgets["scale_label"], locations[2]),
                                                   (widgets["variable 1"], locations[3]),
                                                   (widgets["variable 2"], locations[4]),
                                                   (widgets["scale"], locations[5]),
                                                   (widgets["chain_vis"], locations[12])]
                                 )

        self.side_panel.addstate("1D Histogram", [(widgets["x_axis_label"], locations[0]),
                                                  (widgets["scale_label"], locations[1]),
                                                  (widgets["variable 1"], locations[3]),
                                                  (widgets["scale"], locations[4]),
                                                  (widgets["chain_vis"], locations[12])]
                                 )

        self.side_panel.addstate("2D Histogram", [(widgets["x_axis_label"], locations[0]),
                                                  (widgets["y_axis_label"], locations[1]),
                                                  (widgets["scale_label"], locations[2]),
                                                  (widgets["variable 1"], locations[3]),
                                                  (widgets["variable 2"], locations[4]),
                                                  (widgets["scale"], locations[5]),
                                                  (widgets["chain_vis"], locations[12])]
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

    def loadfile(self) -> None:
        file_names = filedialog.askopenfilenames(filetypes=[("Pickle File", "*.pik")],
                                                 title="Select File(s)", initialdir=PICKLE_FILE_LOCATION)
        if file_names == "":
            return
        # TODO: Prefer a list of strs instead of a giant concatenated str
        self.base_panel.setvar("data label", self.base_panel.getvar(
            "data label") + f"\nLoaded files {file_names}")
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

                        self.data[file_name][key] = {0: states,
                                                     1: mean_states}
            except ValueError as e:
                self.base_panel.setvar("data label", self.base_panel.getvar(
                    "data label") + f"\nError: {e}")
                continue

        self.file_names = {file_name: tk.IntVar(value=1) for file_name in file_names}
        for file_name in self.file_names:
            self.file_names[file_name].trace("w", self.redraw)

        # TODO: Require all file_names have same set of keys, or track only unique keys

        # Generate a button for each parameter
        self.mini_panel.widgets["chart menu"].configure(state=tk.NORMAL)
        self.chart_type.set("select")
        self.mini_panel.widgets["graph button"].configure(state=tk.DISABLED)
        menu: tk.Menu = self.side_panel.widgets["variable 1"]["menu"]
        variable = self.side_panel.widgets["variable 1"]["textvariable"]
        self.widget.setvar(variable, "select")
        menu.delete(0, tk.END)
        for key in self.data[file_names[0]]:
            menu.add_checkbutton(label=key, onvalue=key, offvalue=key, variable=variable)
        menu: tk.Menu = self.side_panel.widgets["variable 2"]["menu"]
        variable = self.side_panel.widgets["variable 2"]["textvariable"]
        self.widget.setvar(variable, "select")
        menu.delete(0, tk.END)
        for key in self.data[file_names[0]]:
            menu.add_checkbutton(label=key, onvalue=key, offvalue=key, variable=variable)

    def chartselect(self) -> None:
        """ Refresh on choosing a new type of plot """
        self.side_panel.loadstate(self.chart_type.get())
        self.mini_panel.widgets["graph button"].configure(state=tk.NORMAL)
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
                name = self.side_panel.widgets["variable 1"]["textvariable"]
                value = self.widget.getvar(name)
                name = self.side_panel.widgets["accepted"]["textvariable"]
                accepted = self.widget.getvar(name)
                name = self.side_panel.widgets["scale"]["textvariable"]
                scale = self.widget.getvar(name)
                entry: tk.Entry = self.side_panel.widgets["hori_marker_entry"]
                hline = entry.get()
                try:
                    hline = (float(hline),)
                except:
                    hline = tuple()
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
                    Plot.traceplot1d(axes, self.data[file_name][value][accepted],
                                     title, scale, *hline)
            case "2D Trace Plot":
                name = self.side_panel.widgets["variable 1"]["textvariable"]
                x_val = self.widget.getvar(name)
                name = self.side_panel.widgets["variable 2"]["textvariable"]
                y_val = self.widget.getvar(name)
                name = self.side_panel.widgets["scale"]["textvariable"]
                scale = self.widget.getvar(name)
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
                    Plot.traceplot2d(axes, self.data[file_name][x_val][True],
                                     self.data[file_name][y_val][True],
                                     x_val, y_val, scale)
            case "1D Histogram":
                name = self.side_panel.widgets["variable 1"]["textvariable"]
                value = self.widget.getvar(name)
                name = self.side_panel.widgets["scale"]["textvariable"]
                scale = self.widget.getvar(name)
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
                    Plot.histogram1d(axes, self.data[file_name][value][True],
                                     f"Accepted {value}", scale)
            case "2D Histogram":
                name = self.side_panel.widgets["variable 1"]["textvariable"]
                x_val = self.widget.getvar(name)
                name = self.side_panel.widgets["variable 2"]["textvariable"]
                y_val = self.widget.getvar(name)
                name = self.side_panel.widgets["scale"]["textvariable"]
                scale = self.widget.getvar(name)
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
                    Plot.histogram2d(axes, self.data[file_name][x_val][True],
                                     self.data[file_name][y_val][True],
                                     x_val, y_val, scale)

                    # colorbar = axes.imshow(hist2d, cmap="Blues")
                    # self.chart.figure.colorbar(colorbar, ax=axes, fraction=0.04)

        self.chart.figure.tight_layout()
        self.chart.canvas.draw()

    def export(self) -> None:
        """ Export currently plotted values as .csv or .npy """
        match key := self.side_panel.state:
            case "1D Trace Plot":
                name = self.side_panel.widgets["variable 1"]["textvariable"]
                value = self.widget.getvar(name)
                name = self.side_panel.widgets["accepted"]["textvariable"]
                accepted = self.widget.getvar(name) == "Accepted"

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
                    vals = self.data[file_name][value][accepted]

                    if out_format == "npy":
                        np.save(out_name, np.vstack((np.arange(len(vals)), vals)).T)
                    elif out_format == "csv":
                        np.savetxt(out_name, np.vstack((np.arange(len(vals)), vals)).T, delimiter=",",
                                   header=f"N,{value}")
                    else:
                        continue

            case "2D Trace Plot":
                name = self.side_panel.widgets["variable 1"]["textvariable"]
                x_val = self.widget.getvar(name)
                name = self.side_panel.widgets["variable 2"]["textvariable"]
                y_val = self.widget.getvar(name)
                accepted = self.widget.getvar(name) == "Accepted"

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

                    vals_x = self.data[file_name][x_val][accepted]
                    vals_y = self.data[file_name][y_val][accepted]

                    # (N x 3) array - (iter #, vals_x, vals_y)
                    if out_format == "npy":
                        np.save(out_name, np.vstack((np.arange(len(vals_x)), vals_x, vals_y)).T)
                    elif out_format == "csv":
                        np.savetxt(out_name, np.vstack((np.arange(len(vals_x)), vals_x, vals_y)).T, delimiter=",",
                                   header=f"N,{x_val},{y_val}")
                    else:
                        continue

            case "1D Histogram":
                # (b x 2 array) - (bins, freq)
                pass
            case "2D Histogram":
                # (b0 x b1) array - (freq matrix), one row/col as bin headers
                pass

window = Window(1000, 800, APPLICATION_NAME)
window.bind(events["key"]["escape"], lambda code: exit())

window.mainloop()
