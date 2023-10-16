"""
First of a two-stage tkinter GUI construction - this one plots all of the widgets
where window.py then adds functionality
"""
import platform
from io import BytesIO
import tkinter as tk
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure

from rclickmenu import Clickmenu, CLICK_EVENTS
from gui_colors import BLACK, WHITE, LIGHT_GREY, GREY, DARK_GREY
from gui_styles import MENU_KWARGS, LABEL_KWARGS

OSTYPE = platform.system().lower()
if OSTYPE == "windows":
    import win32clipboard

class MainClickmenu(Clickmenu):

    def __init__(self, window, master, chart):
        super().__init__(window, master, target_widget=chart.widget)
        self.chart = chart
        self.menu.add_command(label="Copy", command=self.copy_fig)

    def copy_fig(self):
        """
        Adapted from: addcopyfighandler by joshburnett (09/14/2023)
        https://github.com/joshburnett/addcopyfighandler
        """
        if OSTYPE != "windows":
            raise NotImplementedError("Copy-paste only supported on Windows (WIP)")

        with BytesIO() as buf:
            self.chart.canvas.figure.savefig(buf, dpi=600, format="png")

            image = Image.open(buf)
            with BytesIO() as output:
                image.convert("RGB").save(output, "BMP")
                data = output.getvalue()[14:]  # The file header off-set of BMP is 14 bytes
                format_id = win32clipboard.CF_DIB  # DIB = device independent bitmap

        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(format_id, data)
        win32clipboard.CloseClipboard()

class TkGUI():
    """GUI with all widgets plotted"""

    class Panel:
        """
        Creates the frames for:
        1) the plot, 2) the plot options, 3) the import/export buttons, etc...
        """
        def __init__(self, master: tk.Tk, width: int, height: int, color: str) -> None:
            self.widget = tk.Frame(master=master, width=width, height=height,
                                   background=color, border=4, relief="raised")
            self.states = {"blank": list[tuple[tk.Widget, dict[str, int | str]]]()}
            self.state = "blank"
            self.widgets = dict[str, tk.Widget]()
            self.variables = dict[str, tk.Variable]()

        def place(self, x: int, y: int) -> None:
            self.widget.place(x=x, y=y)

        def addstate(self, state: str,
                     widgets: list[tuple[tk.Widget, dict[str, int | str]]]) -> None:
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
        def __init__(self, master: tk.Tk | tk.Toplevel, width: int, height: int) -> None:
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
        self.application_name = title
        self.widget.option_add("*tearOff", False)
        self.chart = self.Chart(self.widget, 600, 600)

        self.clickmenu = MainClickmenu(self, self.widget, self.chart)
        self.widget.bind(CLICK_EVENTS["click"]["right"], self.clickmenu.show)

        self.chart.place(0, 0)
        self.side_panel = self.Panel(self.widget, 400, 430, GREY)
        self.side_panel.place(600, 0)
        self.toolbar_panel = self.Panel(self.widget, 400, 50, DARK_GREY)
        self.toolbar_panel.place(600, 430)
        self.mini_panel = self.Panel(self.widget, 400, 120, DARK_GREY)
        self.mini_panel.place(600, 480)
        self.base_panel = self.Panel(self.widget, 1000, 200, GREY)
        self.base_panel.place(0, 600)

        # Status box - use self.status() to add messages
        self.status_msg = list[str]()
        self.base_panel.variables["status_msg"] = tk.StringVar(value="")
        data_label = tk.Label(master=self.base_panel.widget,
                              textvariable=self.base_panel.variables["status_msg"],
                              width=138, height=11,
                              background=LIGHT_GREY, relief="sunken", border=2,
                              anchor="nw", justify="left")
        data_label.place(x=10, y=10)
        self.base_panel.widgets["data label"] = data_label
        self.populate_mini_panel()
        self.populate_side_panel()
        self.mount_side_panel_states()

        # Figure toolbar
        toolbar = NavigationToolbar2Tk(self.chart.canvas, self.toolbar_panel.widget,
                                       pack_toolbar=False)
        toolbar.place(x=0, y=0, width=390)

    def populate_mini_panel(self) -> None:
        """ Build the small panel that loads data files and refreshes the plot. """
        # Plot type selection
        self.chart_type = tk.StringVar(value="select")
        self.mini_panel.widgets["chart menu"] = tk.OptionMenu(self.mini_panel.widget,
                                                              self.chart_type, "select")
        menu: tk.Menu = self.mini_panel.widgets["chart menu"]["menu"]
        menu.delete(0)
        self.mini_panel.widgets["chart menu"].configure(**MENU_KWARGS, state=tk.DISABLED)
        self.mini_panel.widgets["chart menu"].place(x=20, y=40, anchor="sw")

        # Opens file dialog
        load_button = tk.Button(master=self.mini_panel.widget, width=10, text="Load File",
                                background=BLACK, foreground=WHITE, border=4)
        load_button.place(x=200, y=40, anchor="s")
        self.mini_panel.widgets["load button"] = load_button

        # Export
        export_button = tk.Button(master=self.mini_panel.widget, width=10, text="Export All",
                                  background=BLACK, foreground=WHITE, border=4)
        self.mini_panel.widgets["export all"] = export_button
        self.mini_panel.widgets["export all"].configure(state=tk.DISABLED)
        self.mini_panel.widgets["export all"].place(x=200, y=100, anchor="s")
        

        # Refreshes the plot
        graph_button = tk.Button(master=self.mini_panel.widget, width=10, text="Graph",
                                 background=BLACK, foreground=WHITE, border=4)
        graph_button.place(x=380, y=40, anchor="se")
        graph_button.configure(state=tk.DISABLED)
        self.mini_panel.widgets["graph button"] = graph_button

        # Does a simulation using the state data
        qs_button = tk.Button(master=self.mini_panel.widget, width=10, text="Simulate",
                              background=BLACK, foreground=WHITE, border=4)
        qs_button.place(x=380, y=100, anchor="se")
        qs_button.configure(state=tk.DISABLED)
        self.mini_panel.widgets["quicksim button"] = qs_button

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
        widgets["thickness_label"] = tk.Label(master=panel, text="Thickness [nm]", **LABEL_KWARGS)

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
                                         width=13, border=4, background=BLACK, foreground=WHITE)
        variables["combined_hist"] = tk.IntVar(value=0)
        widgets["combined_hist"] = tk.Checkbutton(panel, text="Single Hist",
                                                  variable=variables["combined_hist"],
                                                  **{"width": 10, "background": LIGHT_GREY})

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

        menu: tk.Menu = widgets["scale"]["menu"]
        menu.delete(0)
        menu.add_checkbutton(label="Linear", onvalue="Linear", offvalue="Linear",
                             variable=variables["scale"])
        menu.add_checkbutton(label="Logarithmic", onvalue="Logarithmic",
                             offvalue="Logarithmic", variable=variables["scale"])


        # Entry for horizontal marker
        variables["hori_marker"] = tk.StringVar()
        widgets["hori_marker_entry"] = tk.Entry(master=panel, width=16, border=3,
                                                textvariable=variables["hori_marker"])

        # Entry to designate initial equilibration period (of states to discard from the statistics)
        variables["equi"] = tk.StringVar()
        widgets["equi_entry"] = tk.Entry(master=panel, width=16, border=3,
                                         textvariable=variables["equi"])

        # Entry for number of bins
        variables["bins"] = tk.StringVar()
        widgets["num_bins_entry"] = tk.Entry(master=panel, width=16, border=3,
                                             textvariable=variables["bins"])

        # Entry to designate sample thickness
        variables["thickness"] = tk.StringVar()
        widgets["thickness"] = tk.Entry(master=panel, width=16, border=3,
                                        textvariable=variables["thickness"])
        
        widgets["export this"] = tk.Button(master=panel, width=16, text="Export This Variable",
                                           background=BLACK, foreground=WHITE, border=4)

    def mount_side_panel_states(self) -> None:
        """Add a map of widget locations for each of the four plotting states"""
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

                     {"x": 20, "y": 156, "anchor": "nw"},
                     {"x": 20, "y": 188, "anchor": "nw"},

                     {"x": 380, "y": 156, "anchor": "e"},

                     {"x": 20, "y": 252, "anchor": "nw"},
                     {"x": 20, "y": 324, "anchor": "sw"},

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
                                                   (widgets["thickness_label"], locations[12]),
                                                   (widgets["thickness"], locations[13]),
                                                   (widgets["chain_vis"], locations[15]),
                                                   (widgets["export this"], locations[16])]
                                 )

        self.side_panel.addstate("2D Trace Plot", [(widgets["x_axis_label"], locations[0]),
                                                   (widgets["y_axis_label"], locations[1]),
                                                   (widgets["scale_label"], locations[2]),
                                                   (widgets["variable 1"], locations[3]),
                                                   (widgets["variable 2"], locations[4]),
                                                   (widgets["scale"], locations[5]),
                                                   (widgets["equi_label"], locations[7]),
                                                   (widgets["equi_entry"], locations[10]),
                                                   (widgets["thickness_label"], locations[12]),
                                                   (widgets["thickness"], locations[13]),
                                                   (widgets["chain_vis"], locations[15]),
                                                   (widgets["export this"], locations[16])]
                                 )

        self.side_panel.addstate("1D Histogram", [(widgets["x_axis_label"], locations[0]),
                                                  (widgets["scale_label"], locations[1]),
                                                  (widgets["variable 1"], locations[3]),
                                                  (widgets["scale"], locations[4]),
                                                  (widgets["chain_vis"], locations[15]),
                                                  (widgets["equi_label"], locations[7]),
                                                  (widgets["equi_entry"], locations[10]),
                                                  (widgets["num_bins_label"], locations[8]),
                                                  (widgets["num_bins_entry"], locations[11]),
                                                  (widgets["thickness_label"], locations[12]),
                                                  (widgets["thickness"], locations[13]),
                                                  (widgets["combined_hist"], locations[14]),
                                                  (widgets["export this"], locations[16])]
                                 )

        self.side_panel.addstate("2D Histogram", [(widgets["x_axis_label"], locations[0]),
                                                  (widgets["y_axis_label"], locations[1]),
                                                  (widgets["scale_label"], locations[2]),
                                                  (widgets["variable 1"], locations[3]),
                                                  (widgets["variable 2"], locations[4]),
                                                  (widgets["scale"], locations[5]),
                                                  (widgets["chain_vis"], locations[15]),
                                                  (widgets["equi_label"], locations[7]),
                                                  (widgets["equi_entry"], locations[10]),
                                                  (widgets["num_bins_label"], locations[8]),
                                                  (widgets["num_bins_entry"], locations[11]),
                                                  (widgets["thickness_label"], locations[12]),
                                                  (widgets["thickness"], locations[13]),
                                                  (widgets["export this"], locations[16])]
                                 )
