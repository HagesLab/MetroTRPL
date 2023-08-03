import tkinter as tk

import mc_plot
from popup import Popup
from gui_colors import LIGHT_GREY
from gui_styles import LABEL_KWARGS

class QuicksimEntryPopup(Popup):

    def __init__(self, window, master):
        self.window = window
        self.toplevel = tk.Toplevel(master)
        self.toplevel.configure(**{"background": LIGHT_GREY})
        width = 500
        height = 500
        x_offset = (self.window.widget.winfo_screenwidth() - width) // 2
        y_offset = (self.window.widget.winfo_screenheight() - height) // 2
        self.toplevel.geometry(f"{width}x{height}+{x_offset}+{y_offset}")
        self.toplevel.resizable(False, False)
        self.toplevel.title("Quicksim Settings")
        self.toplevel.protocol("WM_DELETE_WINDOW", self.on_close)

        self.c_frame = self.window.Panel(self.toplevel, width=width,
                                              height=100, color=LIGHT_GREY)
        self.c_frame.place(0, 0)
        self.c_frame.widgets["n_sims_label"] = tk.Label(master=self.c_frame.widget,
                                                text="# Simulations", **LABEL_KWARGS)
        self.c_frame.widgets["n_sims_label"].place(x=20, y=20)
        self.c_frame.variables["n_sims"] = tk.StringVar()
        self.c_frame.widgets["n_sims_entry"] = tk.Entry(master=self.c_frame.widget, width=16, border=3,
                                                        textvariable=self.c_frame.variables["n_sims"])
        self.c_frame.widgets["n_sims_entry"].place(x=20, y=48)


    def on_close(self):
        self.toplevel.destroy()
