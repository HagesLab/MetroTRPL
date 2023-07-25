"""Custom matplotlib plots for MCMC data"""

from matplotlib.figure import Axes # type: ignore
import numpy as np

def traceplot1d(axes: Axes, x_list: np.ndarray, title: str, scale: str, *hline) -> None:
    """1D trace, showing history of moves for a single parameter"""
    axes.plot(x_list)
    if len(hline) == 1:
        if min(x_list) < hline and hline < max(x_list):
            axes.hlines(hline[0], 0, len(x_list), colors='k', linestyles="dashed")
    axes.set_title(title)
    axes.set_yscale(scale)
    axes.set_xlabel("n", fontstyle="italic")

def traceplot2d(axes: Axes, x_list: np.ndarray, y_list: np.ndarray,
                x_label: str, y_label: str, scale: str) -> None:
    """2D trace, showing history of moves for two parameters"""
    axes.plot(x_list, y_list)
    axes.plot(x_list[0], y_list[0], marker=".", linestyle=" ", color='g',
                label="Start", markersize=10)
    axes.plot(x_list[-1], y_list[-1], marker=".", linestyle=" ", color='r',
                label="End", markersize=10)
    axes.set_xscale(scale)
    axes.set_yscale(scale)
    axes.legend()
    axes.set_xlabel(f"Accepted {x_label}")
    axes.set_ylabel(f"Accepted {y_label}")

def histogram1d(axes: Axes, x_list: np.ndarray, title: str, scale: str, bins: int) -> None:
    """1D histogram, showing distribution of values visited by one parameter"""
    axes.hist(x_list, bins, edgecolor='k')
    axes.set_yscale(scale)
    axes.set_title(title)

def histogram2d(axes: Axes, x_list: np.ndarray, y_list: np.ndarray,
                x_label: str, y_label: str, scale: str, bins: int) -> None:
    """2D histogram, showing distribution of visited values for two parameters"""
    data = axes.hist2d(x_list, y_list, bins, cmap="Blues")[0]
    axes.set_xscale(scale)
    axes.set_yscale(scale)
    axes.set_xlabel(f"Accepted {x_label}")
    axes.set_ylabel(f"Accepted {y_label}")
    return data
