"""Menu of options that should appear on right-click"""
import platform
from io import BytesIO
from tkinter import Menu
from PIL import Image
OSTYPE = platform.system().lower()
if OSTYPE == "windows":
    import win32clipboard

class Clickmenu():
    """Menu of options that should appear on right-click"""
    def __init__(self, window, master, chart) -> None:
        self.window = window
        self.master = master
        self.chart = chart
        self.menu = Menu(self.master, tearoff=0)
        self.menu.add_command(label="Copy", command=self.copy_fig)


    def show(self, event):
        """Display menu at click event location"""
        if event.widget != self.chart.widget:
            return

        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def copy_fig(self):
        """
        Adapted from: addcopyfighandler by joshburnett (09/14/2023)
        https://github.com/joshburnett/addcopyfighandler
        """
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
