# --- IMPORTS --- #
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog
from types import FunctionType
import tkinter as tk
import sim_utils
import pickle

# right click horizontal lines cannot be saved. If we add horizontal lines via a input box, then we could save them
# doing it this way would also allow us to remove horizontal lines and potentially change the appearance of graphs 

# could make variable labels display what variable they are 

# --- CONSTANTS --- #

WIDTH  = 800
HEIGHT = 800
BW     = 4
BASE_TITLE = "Graph Viewer"

# --- CLASSES --- #

class Chart:
    def __init__(self, figure_canvas: FigureCanvasTkAgg, figure: Figure) -> None:
        self.figure_canvas = figure_canvas
        self.figure        = figure
        self.data          = dict[str, dict[bool, list[float]]]()
        self.mean          = tk.BooleanVar()
        self.can_save      = False
        self.menus         = dict[str, tuple[tk.Menu, tk.StringVar, FunctionType]]()  
        self.window        : tk.Tk
        self.canvas        : tk.Canvas
        self.click_menu    : tk.Menu
        self.position      = 0, 0
        self.horizontal    = None
        self.vertical      = None

    def load(self) -> None:
        file_name = filedialog.askopenfilename(initialdir="../outputs/TEST_REAL_STAUB", title="File Select", filetypes=[("Pickle Files", "*.pik")])
        if file_name == "":
            return
        self.window.title(f"{BASE_TITLE} | {file_name}")
        popup(file_name)
        with open(file_name, "rb") as rfile:
            metrostate: sim_utils.MetroState = pickle.load(rfile)
        self.clean()
        self.figure_canvas.draw()
        for key in chart.menus:
            chart.menus[key][1].set("")
            if len(self.data) != 0:
                self.menus[key][0].delete(0, len(self.data)-1)
        self.data.clear()
        self.can_save = False

        for param in metrostate.param_info["names"]:
            if not metrostate.param_info["active"][param]:
                continue
            self.data[param] = dict()
            self.data[param][False] = metrostate.H.__getattribute__(param)[0]
            self.data[param][True] = metrostate.H.__getattribute__("mean_" + param)[0]

            for key in self.menus:
                menu, variable, func = self.menus[key]
                menu.add_checkbutton(label=param, onvalue=param, offvalue=param, variable=variable, command=func)

    def trace1d(self) -> None:
        if self.menus["trace1d"][1].get() == "":
            return
        self.can_save = True
        self.clean()
        axes = self.figure.add_subplot()
        axes.plot(self.data[self.menus["trace1d"][1].get()][self.mean.get()])
        text = self.menus["trace1d"][1].get()
        if self.mean.get():
            text = "Accepted " + text
        axes.set_title(text)
        axes.set_xlabel("n", fontstyle="italic")
        self.figure_canvas.draw()

    def rclick(self, data: tk.Event) -> None:
        if self.can_save:
            self.position = data.x, data.y
            self.click_menu.tk_popup(data.x_root, data.y_root)

    def hline(self) -> None:
        y = self.position[1]
        if self.horizontal != None:
            self.canvas.delete(self.horizontal)
        self.horizontal = self.canvas.create_line(0, y, WIDTH, y, fill=color.GREEN, width=4)

    def vline(self) -> None:
        x = self.position[0]
        if self.vertical != None:
            self.canvas.delete(self.vertical)
        self.vertical = self.canvas.create_line(x, 0, x, HEIGHT, fill=color.GREEN, width=4)


    def clear(self) -> None:
        self.clean()
        self.figure_canvas.draw()
        for key in self.menus:
            self.menus[key][1].set("")

    def clean(self) -> None:
        self.figure.clear()
        if self.horizontal != None:
            self.canvas.delete(self.horizontal)
        if self.vertical != None:
            self.canvas.delete(self.vertical)
        self.vertical = None
        self.horizontal = None


    def trace2d(self) -> None:
        if self.menus["trace2d 1"][1].get() == "" or self.menus["trace2d 2"][1].get() == "":
            return
        self.can_save = True
        self.clean()
        axes = self.figure.add_subplot()
        axes.plot(self.data[self.menus["trace2d 1"][1].get()][self.mean.get()], self.data[self.menus["trace2d 2"][1].get()][self.mean.get()])
        #text = self.menus["trace1d"][1].get()
        #if self.mean.get():
        #    text = "Accepted " + text
        #axes.set_title(text)
        #axes.set_xlabel("n", fontstyle="italic")
        self.figure_canvas.draw()

    def save(self) -> None:
        if not self.can_save:
            return
        file_name = filedialog.asksaveasfilename(initialdir="Figures", defaultextension="*.jpg", filetypes=[("JPEG File", "*.jpg"), ("PNG File", "*.png")])
        if file_name == "":
            return
        self.figure.savefig(file_name)

    def hist1d(self) -> None:
        if self.menus["hist1d"][1].get() == "":
            return
        self.can_save = True
        self.clean()
        axes = self.figure.add_subplot()
        axes.hist(self.data[self.menus["hist1d"][1].get()][True], 100)
        axes.set_title("Accepted " + self.menus["hist1d"][1].get())
        self.figure_canvas.draw()

    def hist2d(self) -> None:
        if self.menus["hist2d 1"][1].get() == "" or self.menus["hist2d 2"][1].get() == "":
            return
        self.can_save = True
        self.clean()
        axes = self.figure.add_subplot()
        x = axes.hist2d(self.data[self.menus["hist2d 1"][1].get()][True], self.data[self.menus["hist2d 2"][1].get()][True], 100, cmap="jet")[0]
        axes.set_xlabel("Accepted " + self.menus["hist2d 1"][1].get())
        axes.set_ylabel("Accepted " + self.menus["hist2d 2"][1].get())
        im = axes.imshow(x, cmap="jet")
        self.figure.colorbar(im, ax=axes, fraction=0.04)
        self.figure_canvas.draw()

class color:
    GREY       = "#606060"
    BLACK      = "#000000"
    DARK_GREY  = "#202020"
    LIGHT_GREY = "#808080"
    WHITE      = "#FFFFFF"
    GREEN      = "#009900"

# --- FUNCTIONS --- #

def popup(file_name: str) -> None:
    puwin = tk.Toplevel(master=window)
    x=window.winfo_rootx()
    y=window.winfo_rooty()
    puwin.geometry(f"200x100+{x+300}+{y+350}")
    puwin.resizable(False, False)
    puwin.title("")
    label = tk.Label(master=puwin, width=200, height=100, text="File Successfully Loaded.")
    label.pack()
    puwin.grab_set()

# --- MAIN --- #

window = tk.Tk()
window.title(BASE_TITLE)
window.resizable(False, False)
window.geometry(f"{WIDTH}x{HEIGHT}")
window.option_add("*tearOff", False)
window.bind("<Escape>", lambda code: exit())

figure = Figure(figsize=(5, 5))
figure_canvas = FigureCanvasTkAgg(master=window, figure=figure)
canvas = figure_canvas.get_tk_widget()
canvas.configure(width=WIDTH, height=HEIGHT, borderwidth=BW, background=color.GREY)
chart = Chart(figure_canvas=figure_canvas, figure=figure)
chart.window = window
chart.click_menu = tk.Menu(master=window)
chart.click_menu.add_command(label="Horizontal Line", command=chart.hline, underline=0)
chart.click_menu.add_command(label="Vertical Line", command=chart.vline, underline=0)
canvas.pack()
chart.canvas = canvas

menu_bar = tk.Menu(master=window)
window.configure(menu=menu_bar)
file_menu = tk.Menu(master=menu_bar)
menu_bar.add_cascade(label="File", menu=file_menu, underline=0)
file_menu.add_command(label="Load", command=chart.load, underline=0)
file_menu.add_command(label="Clear", command=chart.clear, underline=0)
file_menu.add_command(label="Save", command=chart.save, underline=0)

trace_plot_menu = tk.Menu(master=menu_bar)
menu_bar.add_cascade(label="Trace Plot", menu=trace_plot_menu, underline=0)
trace_plot_menu.add_checkbutton(label="Accepted", onvalue=True, offvalue=False, variable=chart.mean, command=chart.trace1d, underline=0)
chart.menus["trace1d"] = tk.Menu(master=trace_plot_menu), tk.StringVar(), chart.trace1d
trace1d_menu = tk.Menu(master=trace_plot_menu)
trace_plot_menu.add_cascade(label="1D", menu=trace1d_menu, underline=0)
trace1d_menu.add_cascade(label="Variable", menu=chart.menus["trace1d"][0], underline=0)

trace2d_menu = tk.Menu(master=trace_plot_menu)
trace_plot_menu.add_cascade(label="2D", menu=trace2d_menu, underline=0)
chart.menus["trace2d 1"] = tk.Menu(master=trace2d_menu), tk.StringVar(), chart.trace2d
trace2d_menu.add_cascade(label="Variable 1", menu=chart.menus["trace2d 1"][0])
chart.menus["trace2d 2"] = tk.Menu(master=trace2d_menu), tk.StringVar(), chart.trace2d
trace2d_menu.add_cascade(label="Variable 2", menu=chart.menus["trace2d 2"][0])


canvas.bind("<Button-3>", chart.rclick)

histogram_menu = tk.Menu(master=menu_bar)
menu_bar.add_cascade(label="Histogram", menu=histogram_menu, underline=0)

hist1d_menu = tk.Menu(master=histogram_menu)
histogram_menu.add_cascade(label="1D", menu=hist1d_menu, underline=0)
chart.menus["hist1d"] = tk.Menu(master=hist1d_menu), tk.StringVar(), chart.hist1d
hist1d_menu.add_cascade(label="Variable", menu=chart.menus["hist1d"][0], underline=0)

hist2d_menu = tk.Menu(master=histogram_menu)
histogram_menu.add_cascade(label="2D", menu=hist2d_menu, underline=0)
chart.menus["hist2d 1"] = tk.Menu(master=hist2d_menu), tk.StringVar(), chart.hist2d
hist2d_menu.add_cascade(label="Variable 1", menu=chart.menus["hist2d 1"][0])
chart.menus["hist2d 2"] = tk.Menu(master=hist2d_menu), tk.StringVar(), chart.hist2d
hist2d_menu.add_cascade(label="Variable 2", menu=chart.menus["hist2d 2"][0])

window.mainloop()