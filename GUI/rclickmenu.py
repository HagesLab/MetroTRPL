"""Menu of options that should appear on right-click"""
from tkinter import Menu

CLICK_EVENTS = {#"key": {"escape": "<Escape>", "enter": "<Return>"},
                "click": {"left": "<Button-1>", "right": "<Button-3>"},}

class Clickmenu():
    """Menu of options that should appear on right-click"""
    def __init__(self, window, master, target_widget) -> None:
        self.window = window
        self.master = master
        self.target_widget = target_widget
        self.menu = Menu(self.master, tearoff=0)
        self.latest_event = (-1, -1)

    def show(self, event):
        """Display menu at click event location"""
        if event.widget != self.target_widget:
            return

        try:
            self.menu.tk_popup(event.x_root, event.y_root)
            self.latest_event = (event.x, event.y)
        finally:
            self.menu.grab_release()
