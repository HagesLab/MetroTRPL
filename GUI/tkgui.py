"""
First of a two-stage tkinter GUI construction - this one plots all of the widgets
where window.py then adds functionality
"""
import platform
from io import BytesIO
import tkinter as tk
import tkinter.scrolledtext as tkscrolled
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
        y_offset = 0 #(self.widget.winfo_screenheight() - height) // 2
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
        data_label = tkscrolled.ScrolledText(master=self.base_panel.widget,
                              width=120, height=10,
                              background=LIGHT_GREY, relief="sunken", bd=2,)
        data_label.place(x=10, y=10)
        self.base_panel.widgets["status_box"] = data_label
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
        widgets["hori_marker_label"] = tk.Label(master=panel,
                                                text="Horizontal Line", **LABEL_KWARGS)
        widgets["equi_label"] = tk.Label(master=panel, text="Equilibration Period", **LABEL_KWARGS)
        widgets["num_bins_label"] = tk.Label(master=panel, text="Bins", **LABEL_KWARGS)
        widgets["thickness_label"] = tk.Label(master=panel, text="Thickness [nm]", **LABEL_KWARGS)
        widgets["xlim_label"] = tk.Label(master=panel, text="X-limits (A, B)", **LABEL_KWARGS)
        widgets["bin_shape_label"] = tk.Label(master=panel, text="Bin Scale", **LABEL_KWARGS)

        # User select menus
        variables["variable_1"] = tk.StringVar(value="select")
        variables["variable_2"] = tk.StringVar(value="select")
        widgets["variable 1"] = tk.OptionMenu(panel, variables["variable_1"], "")
        widgets["variable 2"] = tk.OptionMenu(panel, variables["variable_2"], "")
        variables["scale"] = tk.StringVar(value="Linear")
        widgets["scale"] = tk.OptionMenu(panel, variables["scale"], "")
        variables["bin_shape"] = tk.StringVar(value="Linear")
        widgets["bin_shape"] = tk.OptionMenu(panel, variables["bin_shape"], "")
        widgets["chain_vis"] = tk.Button(panel, text="Select Chains",
                                         width=13, border=4, background=BLACK, foreground=WHITE)
        variables["combined_hist"] = tk.IntVar(value=0)
        widgets["combined_hist"] = tk.Checkbutton(panel, text="Single Hist",
                                                  variable=variables["combined_hist"],
                                                  **{"width": 10, "background": LIGHT_GREY})
        widgets["calc_diffusion"] = tk.Button(panel, text="Diffusion", width=13, border=4,
                                              background=BLACK, foreground=WHITE)

        widgets["variable 1"].configure(**MENU_KWARGS)
        widgets["variable 2"].configure(**MENU_KWARGS)
        widgets["scale"].configure(**MENU_KWARGS)
        widgets["bin_shape"].configure(**MENU_KWARGS)

        # Add fixed items to specific OptionMenus

        menu: tk.Menu = widgets["scale"]["menu"]
        menu.delete(0)
        menu.add_checkbutton(label="Linear", onvalue="Linear", offvalue="Linear",
                             variable=variables["scale"])
        menu.add_checkbutton(label="Log", onvalue="Log",
                             offvalue="Log", variable=variables["scale"])
        menu.add_checkbutton(label="Symlog", onvalue="Symlog",
                             offvalue="Symlog", variable=variables["scale"])
        
        # Menu for whether bins are logarithmically spaced
        menu: tk.Menu = widgets["bin_shape"]["menu"]
        menu.delete(0)
        menu.add_checkbutton(label="Linear", onvalue="Linear", offvalue="Linear",
                             variable=variables["bin_shape"])
        menu.add_checkbutton(label="Log", onvalue="Log",
                             offvalue="Log", variable=variables["bin_shape"])

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
        
        # Entries for x limit
        variables["xlim_l"] = tk.StringVar()
        widgets["xlim_l"] = tk.Entry(master=panel, width=7, border=3,
                                        textvariable=variables["xlim_l"])
        variables["xlim_u"] = tk.StringVar()
        widgets["xlim_u"] = tk.Entry(master=panel, width=7, border=3,
                                        textvariable=variables["xlim_u"])
        
        widgets["export this"] = tk.Button(master=panel, width=16, text="Export This Variable",
                                           background=BLACK, foreground=WHITE, border=4)

    def mount_side_panel_states(self) -> None:
        """Add a map of widget locations for each of the four plotting states"""
        widgets = self.side_panel.widgets

        cols = {"left": 20, "mid": 200, "right": 380,
                "mid_l": 198, "mid_r": 253}
        
        rows = {"1u": 20, "1d": 48, "2u": 88, "2d": 116, "3u": 156, "3d": 188,}

        l1u = {"x": cols["left"], "y": rows["1u"], "anchor": "nw"}
        m1u = {"x": cols["mid"], "y": rows["1u"], "anchor": "n"}
        r1u = {"x": cols["right"], "y": rows["1u"], "anchor": "ne"}

        l1d = {"x": cols["left"], "y": rows["1d"], "anchor": "nw"}
        m1d = {"x": cols["mid"], "y": rows["1d"], "anchor": "n"}
        r1d = {"x": cols["right"], "y": rows["1d"], "anchor": "ne"}

        l2u = {"x": cols["left"], "y": rows["2u"], "anchor": "nw"}
        m2u = {"x": cols["mid"], "y": rows["2u"], "anchor": "n"}
        r2u = {"x": cols["right"], "y": rows["2u"], "anchor": "ne"}

        l2d = {"x": cols["left"], "y": rows["2d"], "anchor": "nw"}
        m2d = {"x": cols["mid"], "y": rows["2d"], "anchor": "n"}
        r2d = {"x": cols["right"], "y": rows["2d"], "anchor": "ne"}

        l3u = {"x": cols["left"], "y": rows["3u"], "anchor": "nw"}
        m3u = {"x": cols["mid"], "y": rows["3u"], "anchor": "n"}
        r3u = {"x": cols["right"], "y": rows["3u"], "anchor": "ne"}

        l3d = {"x": cols["left"], "y": rows["3d"], "anchor": "nw"}
        ml3d = {"x": cols["mid_l"], "y": rows["3d"], "anchor": "ne"}
        mr3d = {"x": cols["mid_r"], "y": rows["3d"], "anchor": "ne"}
        r3d = {"x": cols["right"], "y": rows["3d"], "anchor": "ne"}

        singlehist = {"x": cols["right"], "y": 240, "anchor": "e"}
        chainvis = {"x": cols["left"], "y": 322, "anchor": "nw"}
        export = {"x": cols["left"], "y": 394, "anchor": "sw"}
        diffusion = {"x": cols["right"], "y": 394, "anchor": "se"}

        self.side_panel.addstate("1D Trace Plot", [(widgets["x_axis_label"], l1u),
                                                   (widgets["scale_label"], r1u),
                                                   (widgets["variable 1"], l1d),
                                                   (widgets["scale"], r1d),
                                                   (widgets["hori_marker_label"], l2u),
                                                   (widgets["hori_marker_entry"], l2d),
                                                   (widgets["equi_label"], m2u),
                                                   (widgets["equi_entry"], m2d),
                                                   (widgets["thickness_label"], l3u),
                                                   (widgets["thickness"], l3d),
                                                   (widgets["xlim_label"], m3u),
                                                   (widgets["xlim_l"], ml3d),
                                                   (widgets["xlim_u"], mr3d),
                                                   (widgets["chain_vis"], chainvis),
                                                   (widgets["export this"], export),
                                                   (widgets["calc_diffusion"], diffusion)]
                                 )

        self.side_panel.addstate("2D Trace Plot", [(widgets["x_axis_label"], l1u),
                                                   (widgets["y_axis_label"], m1u),
                                                   (widgets["scale_label"], r1u),
                                                   (widgets["variable 1"], l1d),
                                                   (widgets["variable 2"], m1d),
                                                   (widgets["scale"], r1d),
                                                   (widgets["equi_label"], m2u),
                                                   (widgets["equi_entry"], m2d),
                                                   (widgets["thickness_label"], l3u),
                                                   (widgets["thickness"], l3d),
                                                   (widgets["xlim_label"], m3u),
                                                   (widgets["xlim_l"], ml3d),
                                                   (widgets["xlim_u"], mr3d),
                                                   (widgets["chain_vis"], chainvis),
                                                   (widgets["export this"], export)]
                                 )

        self.side_panel.addstate("1D Histogram", [(widgets["x_axis_label"], l1u),
                                                  (widgets["scale_label"], r1u),
                                                  (widgets["variable 1"], l1d),
                                                  (widgets["scale"], r1d),
                                                  (widgets["equi_label"], m2u),
                                                  (widgets["equi_entry"], m2d),
                                                  (widgets["num_bins_label"], r2u),
                                                  (widgets["num_bins_entry"], r2d),
                                                  (widgets["thickness_label"], l3u),
                                                  (widgets["thickness"], l3d),
                                                  (widgets["xlim_label"], m3u),
                                                  (widgets["xlim_l"], ml3d),
                                                  (widgets["xlim_u"], mr3d),
                                                  (widgets["bin_shape_label"], r3u),
                                                  (widgets["bin_shape"], r3d),
                                                  (widgets["combined_hist"], singlehist),
                                                  (widgets["chain_vis"], chainvis),
                                                  (widgets["export this"], export)]
                                 )

        self.side_panel.addstate("2D Histogram", [(widgets["x_axis_label"], l1u),
                                                  (widgets["y_axis_label"], m1u),
                                                  (widgets["scale_label"], r1u),
                                                  (widgets["variable 1"], l1d),
                                                  (widgets["variable 2"], m1d),
                                                  (widgets["scale"], r1d),
                                                  (widgets["equi_label"], m2u),
                                                  (widgets["equi_entry"], m2d),
                                                  (widgets["num_bins_label"], r2u),
                                                  (widgets["num_bins_entry"], r2d),
                                                  (widgets["thickness_label"], l3u),
                                                  (widgets["thickness"], l3d),
                                                  (widgets["xlim_label"], m3u),
                                                  (widgets["xlim_l"], ml3d),
                                                  (widgets["xlim_u"], mr3d),
                                                  (widgets["chain_vis"], chainvis),
                                                  (widgets["export this"], export)]
                                 )
