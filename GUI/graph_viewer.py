# --- IMPORTS --- #
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
from types import FunctionType
from matplotlib.figure import Figure
import tkinter as tk
import sim_utils
import pickle

# --- CONSTANTS --- #

WIDTH  = 800
HEIGHT = 800
BW     = 4

# --- CLASSES --- #

class Chart:
    def __init__(self, figure_canvas: FigureCanvasTkAgg, figure: Figure) -> None:
        self.figure_canvas = figure_canvas
        self.figure        = figure
        self.data          = dict[str, dict[bool, list[float]]]()
        self.mean          = tk.BooleanVar()
        self.can_save      = False
        self.menus         = dict[str, tuple[tk.Menu, tk.StringVar, FunctionType]]()

    def load(self) -> None:
        file_name = filedialog.askopenfilename(title="File Select", filetypes=[("Pickle Files", "*.pik")])
        if file_name == "":
            return
        with open(file_name, "rb") as rfile:
            metrostate: sim_utils.MetroState = pickle.load(rfile)
        self.figure.clear()
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
            self.data[param][False] = metrostate.H.__getattribute__(param)[0].tolist()
            self.data[param][True] = metrostate.H.__getattribute__("mean_"+param)[0].tolist()

            for key in self.menus:
                menu, variable, func = self.menus[key]
                menu.add_checkbutton(label=param, onvalue=param, offvalue=param, variable=variable, command=func)

    def graph(self) -> None:
        if self.menus["graph"][1].get() == "":
            return
        self.can_save = True
        self.figure.clear()
        axes = self.figure.add_subplot()
        axes.plot(self.data[self.menus["graph"][1].get()][self.mean.get()])
        text = self.menus["graph"][1].get()
        if self.mean.get():
            text = "mean " + text
        axes.set_title(text)
        self.figure_canvas.draw()

    def save(self) -> None:
        if self.can_save:
            return
        text = self.menus["graph"][1].get()
        if self.mean.get():
            text = "mean_" + text
        self.figure.savefig(f"Figures/{text}.jpg")

    def hist1d(self) -> None:
        if self.menus["hist1d"][1].get() == "":
            return
        self.can_save = True
        self.figure.clear()
        axes = self.figure.add_subplot()
        axes.hist(self.data[self.menus["hist1d"][1].get()][True], 100)
        axes.set_title("mean " + self.menus["hist1d"][1].get())
        self.figure_canvas.draw()

    def hist2d(self) -> None:
        if self.menus["hist2d 1"][1].get() == "" or self.menus["hist2d 2"][1].get() == "":
            return
        self.can_save = True
        self.figure.clear()
        axes = self.figure.add_subplot()
        x = axes.hist2d(self.data[self.menus["hist2d 1"][1].get()][True], self.data[self.menus["hist2d 2"][1].get()][True], 100, cmap="jet")[0]
        axes.set_xlabel("mean " + self.menus["hist2d 1"][1].get())
        axes.set_ylabel("mean " + self.menus["hist2d 2"][1].get())
        self.figure_canvas.draw()

class color:
    GREY       = "#606060"
    BLACK      = "#000000"
    DARK_GREY  = "#202020"
    LIGHT_GREY = "#808080"
    WHITE      = "#FFFFFF"

# --- FUNCTIONS --- #

# --- MAIN --- #

window = tk.Tk()
window.title("Graph Viewer")
window.resizable(False, False)
window.geometry(f"{WIDTH}x{HEIGHT}")
window.option_add("*tearOff", False)
window.bind("<Escape>", lambda code: exit())

figure = Figure(figsize=(5, 5))
figure_canvas = FigureCanvasTkAgg(master=window, figure=figure)
canvas = figure_canvas.get_tk_widget()
canvas.configure(width=WIDTH, height=HEIGHT, borderwidth=BW, background=color.GREY)
chart = Chart(figure_canvas=figure_canvas, figure=figure) 
canvas.pack()

menu_bar = tk.Menu(master=window)
window.configure(menu=menu_bar)
file_menu = tk.Menu(master=menu_bar)
menu_bar.add_cascade(label="File", menu=file_menu, underline=0)
file_menu.add_command(label="Load", command=chart.load, underline=0)
file_menu.add_command(label="Save", command=chart.save, underline=0)

graph_menu = tk.Menu(master=menu_bar)
menu_bar.add_cascade(label="Graph", menu=graph_menu, underline=0)
graph_menu.add_checkbutton(label="Mean", onvalue=True, offvalue=False, variable=chart.mean, command=chart.graph, underline=0)
chart.menus["graph"] = tk.Menu(master=graph_menu), tk.StringVar(), chart.graph
graph_menu.add_cascade(label="Variable", menu=chart.menus["graph"][0], underline=0)


histogram_menu = tk.Menu(master=menu_bar)
menu_bar.add_cascade(label="Histo", menu=histogram_menu, underline=0)

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