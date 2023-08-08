import tkinter as tk

import mc_plot
from popup import Popup
from gui_colors import LIGHT_GREY

WIDTH = 500
HEIGHT = 500

def rgb(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"

class QuicksimResultPopup(Popup):

    def __init__(self, window, master):
        super().__init__(window, master, WIDTH, HEIGHT)
        self.toplevel.resizable(False, False)
        self.toplevel.title("Quicksim result")
        self.qs_chart = self.window.Chart(self.toplevel, 400, 400)
        self.qs_chart.place(0, 0)
        self.qs_chart.figure.clear()
        self.qs_axes = self.qs_chart.figure.add_subplot()

    def plot(self, sim_result, colors):
        """Add a curve to the quicksim plot"""
        xlabel = "delay time [ns]"
        ylabel = "TRPL"
        scale = "log"
        color = colors[0]
        mc_plot.sim_plot(self.qs_axes, sim_result[0], sim_result[1], xlabel,
                         ylabel, scale, color)
        self.qs_chart.figure.tight_layout()
        self.qs_chart.canvas.draw()

    def clear(self):
        """Reset the quicksim plot"""
        self.qs_chart.figure.clear()

    def on_close(self):
        """Re-enable the simulate button"""
        self.toplevel.destroy()
        self.window.mini_panel.widgets["quicksim button"].configure(state=tk.NORMAL)
        self.window.mini_panel.widgets["load button"].configure(state=tk.NORMAL)
