"""
Parent class for popups, to be expanded when
the number of different popup types becomes
less manageable
"""
import tkinter as tk
from gui_colors import LIGHT_GREY

class Popup():
    """Popup manager for tkinter Toplevels attached to a Window object"""
    def __init__(self, window, master, width, height):
        self.window = window
        self.toplevel = tk.Toplevel(master)
        self.is_open = True
        self.toplevel.configure(**{"background": LIGHT_GREY})
        self.toplevel.protocol("WM_DELETE_WINDOW", self.on_close)

        x_offset = (self.window.widget.winfo_screenwidth() - width) // 2
        y_offset = (self.window.widget.winfo_screenheight() - height) // 2
        self.toplevel.geometry(f"{width}x{height}+{x_offset}+{y_offset}")


    def on_close(self) -> None:
        self.toplevel.destroy()
