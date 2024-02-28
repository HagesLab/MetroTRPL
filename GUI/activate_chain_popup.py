"""
When multiple MMC pickle files are loaded, toggle the visibility
of specific chains for visualization and simulation purposes.
"""
import os
import tkinter as tk

from popup import Popup
from gui_colors import LIGHT_GREY, PLOT_COLOR_CYCLE

WIDTH_PER_COL = 160
DEFAULT_CHAINS_PER_COL = 8
BASE_HEIGHT = 60
HEIGHT_PER_CHAIN = 36
class ActivateChainPopup(Popup):

    def __init__(self, window, master) -> None:
        """Toggle the visibility of specific MCMC chains."""
        if len(window.chains) > DEFAULT_CHAINS_PER_COL:
            chains_per_col = len(window.chains) // window.n_files
        else:
            chains_per_col = DEFAULT_CHAINS_PER_COL

        n_cols = 1 + len(window.chains) // chains_per_col
        super().__init__(window, master,
                         n_cols * WIDTH_PER_COL,
                         BASE_HEIGHT + HEIGHT_PER_CHAIN * min(chains_per_col, len(window.chains)))
        self._toggle_all = tk.IntVar(value=1)
        self._toggle_all.trace("w", self.on_toggle_all)
        self.toplevel.attributes('-topmost', 'true')
        self.toplevel.title("Toggle MMC chains")
        tk.Label(self.toplevel, text="Display:", background=LIGHT_GREY).grid(row=0, column=0, columnspan=99)
        tk.Checkbutton(self.toplevel, text="Toggle All", variable=self._toggle_all, onvalue=1, offvalue=0,
                       background=LIGHT_GREY).grid(row=1,column=0,columnspan=99)
        for i, chain in enumerate(self.window.chains):
            tk.Checkbutton(self.toplevel, text=os.path.basename(chain.fname),
                           variable=chain.visible,
                           onvalue=1, offvalue=0, background=LIGHT_GREY).grid(row=i%chains_per_col+2, column=2*(i//chains_per_col))
            
            tk.Label(self.toplevel, width=4, height=2, background=PLOT_COLOR_CYCLE[i % len(PLOT_COLOR_CYCLE)]).grid(row=i%chains_per_col+2,column=1+2*(i//chains_per_col), padx=(0,5))

    def on_toggle_all(self, *args) -> None:
        """Sets visibility of all loaded chains."""
        v = self._toggle_all.get()
        for chain in self.window.chains:
            chain.visible.trace_vdelete("w", chain.visible.trace_id)
            chain.visible.set(v)
            chain.visible.trace_id = chain.visible.trace("w", self.window.on_active_chain_update)

        self.window.on_active_chain_update()

    def on_close(self) -> None:
        self.is_open = False
        self.window.side_panel.widgets["chain_vis"].configure(state=tk.NORMAL) # type: ignore
        self.toplevel.destroy()