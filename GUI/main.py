from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog
from types import FunctionType
import tkinter as tk
import numpy as np
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
    def traceplot1d(figure: Figure, x_list: np.ndarray, title: str, scale: str, *hline) -> None:
        axes = figure.add_subplot()
        axes.plot(x_list)
        if len(hline) == 1:
            if min(x_list) < hline and hline < max(x_list):
                axes.hlines(hline[0], 0, len(x_list), colors=BLACK, linestyles="dashed")
        axes.set_title(title)
        axes.set_yscale(scale)
        axes.set_xlabel("n", fontstyle="italic")

    def traceplot2d(figure: Figure, x_list: np.ndarray, y_list: np.ndarray,
                    x_label: str, y_label: str, scale: str) -> None:
        axes = figure.add_subplot()
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

    def histogram1d(figure: Figure, x_list: np.ndarray, title: str, scale: str) -> None:
        axes = figure.add_subplot()
        axes.hist(x_list, 100, edgecolor=BLACK)
        axes.set_yscale(scale)
        axes.set_title(title)

    def histogram2d(figure: Figure, x_list: np.ndarray, y_list: np.ndarray,
                    x_label: str, y_label: str, scale: str) -> None:
        axes = figure.add_subplot()
        data = axes.hist2d(x_list, y_list, 100, cmap="Blues")[0]
        axes.set_xscale(scale)
        axes.set_yscale(scale)
        axes.set_xlabel(f"Accepted {x_label}")
        axes.set_ylabel(f"Accepted {y_label}")
        colorbar = axes.imshow(data, cmap="Blues")
        figure.colorbar(colorbar, ax=axes, fraction=0.04)


class Window:
    class Popup:
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
        self.data = dict[str, dict[bool]]()
        self.chart.place(0, 0)
        self.side_panel = self.Panel(self.widget, 400, 540, GREY)
        self.side_panel.place(600, 0)
        self.mini_panel = self.Panel(self.widget, 400, 60, DARK_GREY)
        self.mini_panel.place(600, 540)
        self.base_panel = self.Panel(self.widget, 1000, 200, GREY)
        self.base_panel.place(0, 600)
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
        load_button = tk.Button(master=self.mini_panel.widget, width=10, text="Load File",
                                background=BLACK, foreground=WHITE, command=self.loadfile, border=4)
        load_button.place(x=200, y=40, anchor="s")
        self.mini_panel.widgets["load button"] = load_button
        graph_button = tk.Button(master=self.mini_panel.widget, width=10, text="Graph",
                                 background=BLACK, foreground=WHITE, command=self.drawchart, border=4)
        graph_button.place(x=380, y=40, anchor="se")
        graph_button.configure(state=tk.DISABLED)
        self.mini_panel.widgets["graph button"] = graph_button
        self.file = tk.StringVar(value="")
        data_label = tk.Label(master=self.base_panel.widget, textvariable=self.file, width=138, height=11,
                              background=LIGHT_GREY, relief="sunken", border=2, anchor="nw", justify="left")
        data_label.place(x=10, y=10)
        self.base_panel.widgets["data label"] = data_label
        self.base_panel.setvar("data label", "Use Load File to select a file")
        self.make_side_panel()

    def make_side_panel(self) -> None:
        """ Build the plot control panel to the right of the plotting frame. """
        widgets = self.side_panel.widgets
        panel = self.side_panel.widget
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
                     {"x": 380, "y": 116, "anchor": "ne"}
                     ]

        # Text labels
        widgets["x_axis_label"] = tk.Label(master=panel, text="X Axis", **LABEL_KWARGS)
        widgets["y_axis_label"] = tk.Label(master=panel, text="Y Axis", **LABEL_KWARGS)
        widgets["scale_label"] = tk.Label(master=panel, text="Axis Scale", **LABEL_KWARGS)
        widgets["accept_label"] = tk.Label(master=panel, text="Filter", **LABEL_KWARGS)
        widgets["hori_marker_label"] = tk.Label(master=panel, text="Horizontal Line", **LABEL_KWARGS)

        # User select menus
        variable_1 = tk.StringVar(value="select")
        variable_2 = tk.StringVar(value="select")
        accepted = tk.StringVar(value="Accepted")
        scale = tk.StringVar(value="Linear")
        widgets["variable 1"] = tk.OptionMenu(panel, variable_1, "")
        widgets["variable 2"] = tk.OptionMenu(panel, variable_2, "")
        widgets["scale"] = tk.OptionMenu(panel, scale, "")
        widgets["accepted"] = tk.OptionMenu(panel, accepted, "")

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

        self.side_panel.addstate("1D Trace Plot", [(widgets["x_axis_label"], locations[0]), (widgets["accept_label"], locations[1]), (widgets["scale_label"], locations[2]),
                                                   (widgets["variable 1"], locations[3]), (widgets["accepted"],
                                                                                             locations[4]), (widgets["scale"], locations[5]),
                                                     (widgets["hori_marker_label"], locations[6]), (widgets["hori_marker_entry"], locations[9])])
        self.side_panel.addstate("2D Trace Plot", [(widgets["x_axis_label"], locations[0]), (widgets["y_axis_label"], locations[1]), (widgets["scale_label"], locations[2]),
                                                   (widgets["variable 1"], locations[3]), (widgets["variable 2"], locations[4]), (widgets["scale"], locations[5])])
        self.side_panel.addstate("1D Histogram", [(widgets["x_axis_label"], locations[0]), (widgets["scale_label"], locations[1]), (widgets["variable 1"], locations[3]),
                                                  (widgets["scale"], locations[4])])
        self.side_panel.addstate("2D Histogram", [(widgets["x_axis_label"], locations[0]), (widgets["y_axis_label"], locations[1]), (widgets["scale_label"], locations[2]),
                                                  (widgets["variable 1"], locations[3]), (widgets["variable 2"], locations[4]), (widgets["scale"], locations[5])])

    def mainloop(self) -> None:
        self.widget.mainloop()

    def bind(self, event: str, command: FunctionType) -> None:
        self.widget.bind(event, command)

    def loadfile(self) -> None:
        file_name = filedialog.askopenfilename(filetypes=[("Pickle File", "*.pik")],
                                               title="Select File", initialdir=PICKLE_FILE_LOCATION)
        if file_name == "":
            return
        self.base_panel.setvar("data label", self.base_panel.getvar(
            "data label") + "\nText can also be appened (or deleted).")
        self.widget.title(f"{APPLICATION_NAME} - {file_name}")
        with open(file_name, "rb") as rfile:
            metrostate: sim_utils.MetroState = pickle.load(rfile)
        self.data.clear()
        for key in metrostate.param_info["names"]:
            # TODO: add a method to select chains
            if metrostate.param_info["active"][key]:
                self.data[key] = {0: metrostate.H.__getattribute__(
                    key)[0], 1: metrostate.H.__getattribute__(f"mean_{key}")[0]}
        self.mini_panel.widgets["chart menu"].configure(state=tk.NORMAL)
        self.chart_type.set("select")
        self.mini_panel.widgets["graph button"].configure(state=tk.DISABLED)
        menu: tk.Menu = self.side_panel.widgets["variable 1"]["menu"]
        variable = self.side_panel.widgets["variable 1"]["textvariable"]
        self.widget.setvar(variable, "select")
        menu.delete(0, tk.END)
        for key in self.data:
            menu.add_checkbutton(label=key, onvalue=key, offvalue=key, variable=variable)
        menu: tk.Menu = self.side_panel.widgets["variable 2"]["menu"]
        variable = self.side_panel.widgets["variable 2"]["textvariable"]
        self.widget.setvar(variable, "select")
        menu.delete(0, tk.END)
        for key in self.data:
            menu.add_checkbutton(label=key, onvalue=key, offvalue=key, variable=variable)

    def chartselect(self) -> None:
        self.side_panel.loadstate(self.chart_type.get())
        self.mini_panel.widgets["graph button"].configure(state=tk.NORMAL)
        self.chart.figure.clear()
        self.chart.canvas.draw()

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
                Plot.traceplot1d(self.chart.figure,
                                 self.data[value][accepted], title, scale, *hline)
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
                Plot.traceplot2d(
                    self.chart.figure, self.data[x_val][True], self.data[y_val][True], x_val, y_val, scale)
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
                Plot.histogram1d(self.chart.figure,
                                 self.data[value][True], f"Accepted {value}", scale)
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
                Plot.histogram2d(
                    self.chart.figure, self.data[x_val][True], self.data[y_val][True], x_val, y_val, scale)
        self.chart.canvas.draw()


window = Window(1000, 800, APPLICATION_NAME)
window.bind(events["key"]["escape"], lambda code: exit())

window.mainloop()
