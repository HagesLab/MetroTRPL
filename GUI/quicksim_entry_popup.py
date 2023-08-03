import tkinter as tk
from functools import partial

import mc_plot
from popup import Popup
from gui_colors import LIGHT_GREY, BLACK, WHITE, DARK_GREY
from gui_styles import LABEL_KWARGS

WIDTH = 500
HEIGHT = 500

class QuicksimEntryPopup(Popup):

    def __init__(self, window, master) -> None:
        super().__init__(window, master, WIDTH, HEIGHT)
        self.continue_ = False
        self.toplevel.resizable(False, False)
        self.toplevel.title("Quicksim Settings")
        self.toplevel.protocol("WM_DELETE_WINDOW", partial(self.on_close, False))

        self.c_frame = self.window.Panel(self.toplevel, width=500,
                                              height=100, color=DARK_GREY)
        self.c_frame.variables["n_sims"] = tk.StringVar(value="1")
        self.c_frame.variables["n_sims"].trace("w", self.count_sims)
        self.c_frame.variables["total_sims"] = tk.StringVar()
        self.draw_c_frame()
        self.c_frame.place(0, 0)

    def count_sims(self, *args):
        try:
            n_sims = int(self.c_frame.variables["n_sims"].get())
        except ValueError:
            n_sims = 0

        n_chains = 0
        for file_name in self.window.file_names:
            if self.window.file_names[file_name].get():
                n_chains += 1

        self.c_frame.variables["total_sims"].set(f"{n_sims * n_chains} total simulations")

    def on_close(self, continue_ : bool=False) -> None:
        self.continue_ = continue_
        self.is_open = False
        self.toplevel.destroy()

    def draw_c_frame(self) -> None:
        self.count_sims()
        self.c_frame.widgets["n_sims_label"] = tk.Label(master=self.c_frame.widget,
                                                text="# Simulations", **LABEL_KWARGS)
        self.c_frame.widgets["n_sims_label"].place(x=20, y=20)
        
        self.c_frame.widgets["n_sims_entry"] = tk.Spinbox(master=self.c_frame.widget, from_=1, to=12,
                                                          state="readonly", width=15, border=3,
                                                          textvariable=self.c_frame.variables["n_sims"])
        self.c_frame.widgets["n_sims_entry"].place(x=20, y=48)

        self.c_frame.widgets["total_sims_label"] = tk.Label(master=self.c_frame.widget,
                                                            textvariable=self.c_frame.variables["total_sims"],
                                                            width=16, background=LIGHT_GREY)
        self.c_frame.widgets["total_sims_label"].place(x=360, y=20)

        self.c_frame.widgets["continue"] = tk.Button(master=self.c_frame.widget, width=15, text="Continue",
                                                     background=BLACK, foreground=WHITE,
                                                     command=partial(self.on_close, True),
                                                     border=4)
        self.c_frame.widgets["continue"].place(x=360, y=40)
