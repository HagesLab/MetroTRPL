"""
When multiple MMC pickle files are loaded, toggle the visibility
of specific chains for visualization and simulation purposes.
"""
import os
import tkinter as tk

from popup import Popup
from gui_colors import LIGHT_GREY, PLOT_COLOR_CYCLE

WIDTH_PER_COL = 180
CHAINS_PER_ROW = 8
BASE_HEIGHT = 10
HEIGHT_PER_CHAIN = 40
class ActivateChainPopup(Popup):

    def __init__(self, window, master) -> None:
        """Toggle the visibility of specific MCMC chains."""
        n_cols = 1 + len(window.file_names) // CHAINS_PER_ROW
        super().__init__(window, master,
                         n_cols * WIDTH_PER_COL,
                         BASE_HEIGHT + HEIGHT_PER_CHAIN * len(window.file_names))
        self.toplevel.attributes('-topmost', 'true')
        self.toplevel.title("Toggle MMC chains")
        tk.Label(self.toplevel, text="Display:", background=LIGHT_GREY).grid(row=0, column=0, columnspan=99)
        for i, file_name in enumerate(self.window.file_names):
            tk.Checkbutton(self.toplevel, text=os.path.basename(file_name),
                           variable=self.window.file_names[file_name],
                           onvalue=1, offvalue=0, background=LIGHT_GREY).grid(row=i%CHAINS_PER_ROW+1, column=2*(i//CHAINS_PER_ROW))
            
            tk.Label(self.toplevel, width=4, height=2, background=PLOT_COLOR_CYCLE[i % len(PLOT_COLOR_CYCLE)]).grid(row=i%CHAINS_PER_ROW+1,column=1+2*(i//CHAINS_PER_ROW), padx=(0,5))

    def on_close(self) -> None:
        self.is_open = False
        self.window.side_panel.widgets["chain_vis"].configure(state=tk.NORMAL) # type: ignore
        self.toplevel.destroy()