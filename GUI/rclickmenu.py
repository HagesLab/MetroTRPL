"""Menu of options that should appear on right-click"""
import platform
from io import BytesIO
from tkinter import Menu
from PIL import Image

CLICK_EVENTS = {#"key": {"escape": "<Escape>", "enter": "<Return>"},
                "click": {"left": "<Button-1>", "right": "<Button-3>"},}

REQ_CLIPBOARD_LIB = {"windows": "win32clipboard"}
OSTYPE = platform.system().lower()
if OSTYPE == "windows":
    try:
        import win32clipboard
        HAS_CLIPBOARD_LIB = True
    except ImportError:
        HAS_CLIPBOARD_LIB = False

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

class FigureClickmenu(Clickmenu):
    """Adds copy-paste functionality to clickmenu"""

    def __init__(self, window, master, chart):
        super().__init__(window, master, target_widget=chart.widget)
        self.chart = chart
        self.menu.add_command(label="Copy", command=self.copy_fig)

    def copy_fig(self):
        """
        Adapted from: addcopyfighandler by joshburnett (09/14/2023)
        https://github.com/joshburnett/addcopyfighandler
        """
        if not HAS_CLIPBOARD_LIB:
            raise ImportError(f"No copy-paste library found: {OSTYPE} systems require {REQ_CLIPBOARD_LIB[OSTYPE]}")

        if OSTYPE != "windows":
            raise NotImplementedError("Copy-paste only supported on Windows (WIP)")

        with BytesIO() as buf:
            self.chart.canvas.figure.savefig(buf, dpi=600, format="png")

            image = Image.open(buf)
            with BytesIO() as output:
                image.convert("RGB").save(output, "BMP")
                data = output.getvalue()[14:]  # The file header off-set of BMP is 14 bytes
                format_id = win32clipboard.CF_DIB  # DIB = device independent bitmap

        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(format_id, data)
        win32clipboard.CloseClipboard()
