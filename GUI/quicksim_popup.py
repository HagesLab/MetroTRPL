import tkinter as tk

import mc_plot
from popup import Popup

class QuicksimPopup(Popup):

    def __init__(self, window, master, bkg_color):
        self.window = window
        self.toplevel = tk.Toplevel(master)
        self.toplevel.configure(**{"background": bkg_color})
        width = 500
        height = 500
        x_offset = (self.window.widget.winfo_screenwidth() - width) // 2
        y_offset = (self.window.widget.winfo_screenheight() - height) // 2
        self.toplevel.geometry(f"{width}x{height}+{x_offset}+{y_offset}")
        self.toplevel.resizable(False, False)
        self.toplevel.title("Quicksim result")
        self.toplevel.protocol("WM_DELETE_WINDOW", self.on_close)
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
