"""
When multiple MMC pickle files are loaded, toggle the visibility
of specific chains for visualization and simulation purposes.
"""
import os
import tkinter as tk

from popup import Popup
from gui_colors import LIGHT_GREY, PLOT_COLOR_CYCLE
class ActivateChainPopup(Popup):

    def __init__(self, window, master) -> None:
        """Toggle the visibility of specific MCMC chains."""
        self.window = window
        self.is_open = True
        self.toplevel = tk.Toplevel(master)
        width = 200
        height = 200
        x_offset = (self.window.widget.winfo_screenwidth() - width) // 2
        y_offset = (self.window.widget.winfo_screenheight() - height) // 2
        self.toplevel.geometry(f"{width}x{height}+{x_offset}+{y_offset}")
        self.toplevel.configure(**{"background": LIGHT_GREY})
        self.toplevel.attributes('-topmost', 'true')
        self.toplevel.protocol("WM_DELETE_WINDOW", self.on_close)
        tk.Label(self.toplevel, text="Display:", background=LIGHT_GREY).grid(row=0, column=0, columnspan=2)
        for i, file_name in enumerate(self.window.file_names):
            tk.Checkbutton(self.toplevel, text=os.path.basename(file_name),
                           variable=self.window.file_names[file_name],
                           onvalue=1, offvalue=0, background=LIGHT_GREY).grid(row=i+1, column=0)
            
            tk.Label(self.toplevel, width=4, height=2, background=PLOT_COLOR_CYCLE[i % len(PLOT_COLOR_CYCLE)]).grid(row=i+1,column=1)

    def on_close(self) -> None:
        self.is_open = False
        self.window.side_panel.widgets["chain_vis"].configure(state=tk.NORMAL) # type: ignore
        self.toplevel.destroy()