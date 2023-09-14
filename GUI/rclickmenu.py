"""Menu of options that should appear on right-click"""
import platform
from PIL import Image
from io import BytesIO
OSTYPE = platform.system().lower()
if OSTYPE == "windows":
    import win32clipboard

class Clickmenu():

    def __init__(self, window, master) -> None:
        self.window = window
        self.master = master
        return
    
    def copy_fig(self, canvas):
        """
        Adapted from: addcopyfighandler by joshburnett (09/14/2023)
        https://github.com/joshburnett/addcopyfighandler
        """
        if OSTYPE != "windows":
            raise NotImplementedError("Copy-paste only supported on Windows (WIP)")

        with BytesIO() as buf:
            canvas.figure.savefig(buf, dpi=600, format="png")
        
            # Open the saved image using PIL
            im = Image.open(buf)
            with BytesIO() as output:
                im.convert("RGB").save(output, "BMP")
                data = output.getvalue()[14:]  # The file header off-set of BMP is 14 bytes
                format_id = win32clipboard.CF_DIB  # DIB = device independent bitmap

        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(format_id, data)
        win32clipboard.CloseClipboard()
