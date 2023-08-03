"""List of standardized colors for GUI"""

def rgb(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"

WHITE = rgb(255, 255, 255)
LIGHT_GREY = rgb(191, 191, 191)
GREY = rgb(127, 127, 127)
DARK_GREY = rgb(63, 63, 63)
BLACK = rgb(0, 0, 0)
RED = rgb(127, 0, 0)
GREEN = rgb(0, 127, 0)
