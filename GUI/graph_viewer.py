# --- IMPORTS --- #

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
from matplotlib.figure import Figure
import tkinter as tk
import sim_utils
import pickle

# --- CONSTANTS --- #

WIDTH = 500
HEIGHT = 500
BW = 4

# --- CLASSES --- #

class Chart:
    def __init__(self, figure_canvas: FigureCanvasTkAgg, figure: Figure) -> None:
        self.figure_canvas = figure_canvas
        self.figure = figure
        self.data = dict()
        self.mean = tk.BooleanVar()
        self.param = tk.StringVar()
        self.menu: tk.Menu | None = None

    def load(self) -> None:
        file_name = filedialog.askopenfilename(title="File Select", filetypes=[("Pickle Files", "*.pik")])
        if file_name == "":
            return
        with open(file_name, "rb") as rfile:
            metrostate: sim_utils.MetroState = pickle.load(rfile)
        self.figure.clear()
        self.figure_canvas.draw()
        self.param.set("")
        if len(self.data) != 0:
            self.menu.delete(0, len(self.data)-1)
        self.data.clear()

        for param in metrostate.param_info["names"]:
            if not metrostate.param_info["active"][param]:
                continue
            self.data[param] = dict()
            self.data[param][False] = metrostate.H.__getattribute__(param)[0].tolist()
            self.data[param][True] = metrostate.H.__getattribute__("mean_"+param)[0].tolist()
            self.menu.add_checkbutton(label=param, onvalue=param, offvalue=param, variable=self.param, command=self.graph)

    def graph(self) -> None:
        if self.param.get() == "":
            return
        self.figure.clear()
        axes = self.figure.add_subplot()
        axes.plot(self.data[self.param.get()][self.mean.get()])
        text = self.param.get()
        if self.mean.get():
            text = "mean " + text
        axes.set_title(text)
        self.figure_canvas.draw()

    def save(self) -> None:
        if self.param.get() == "":
            return
        text = self.param.get()
        if self.mean.get():
            text = "mean_" + text
        self.figure.savefig(f"Figures/{text}.jpg")

class color:
    GREY = "#606060"
    BLACK = "#000000"
    DARK_GREY = "#202020"
    LIGHT_GREY = "#808080"
    WHITE = "#FFFFFF"

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
graph_menu = tk.Menu(master=menu_bar)
menu_bar.add_cascade(label="File", menu=file_menu, underline=0)
menu_bar.add_cascade(label="Graph", menu=graph_menu, underline=0)

file_menu.add_command(label="Load", command=chart.load, underline=0)
file_menu.add_command(label="Save", command=chart.save, underline=0)

chart.menu = tk.Menu(master=graph_menu)

graph_menu.add_checkbutton(label="Mean", onvalue=True, offvalue=False, variable=chart.mean, command=chart.graph, underline=0)
graph_menu.add_cascade(label="Variables", menu=chart.menu, underline=0)

window.mainloop()