"""Custom matplotlib plots for MCMC data"""

from matplotlib.figure import Axes # type: ignore
import numpy as np

from matplotlib import rcParams
rcParams.update({"font.size":16})
# For symlog - no variable ever crosses between > 0 to < 0,
# so this allows negative log
LINTHRESH = 1e-99 
def traceplot1d(axes: Axes, x_list: np.ndarray, title: str, scale: str, xlim: tuple,
                hlines: tuple, vlines: tuple[int],
                color: str) -> None:
    """1D trace, showing history of moves for a single parameter"""
    axes.plot(x_list, color=color)
    for hline in hlines:
        if min(x_list) < hline < max(x_list):  # Draw only if within trace range
            axes.hlines(hline, 0, len(x_list), colors='k', linestyles="dashed")

    for vline in vlines:
        if 0 < vline <= len(x_list):
            axes.vlines(vline, np.amin(x_list), np.amax(x_list), colors='k')
    axes.set_title(title)
    if scale == "symlog":
        axes.set_yscale(scale, linthresh=LINTHRESH)
    else:
        axes.set_yscale(scale)
    axes.set_xlabel("n", fontstyle="italic")

    if xlim[0] is None and xlim[1] is None:
        pass
    elif xlim[0] is None:
        axes.set_xlim(right=xlim[1])
    elif xlim[1] is None:
        axes.set_xlim(left=xlim[0])
    else:
        axes.set_xlim(xlim)


def traceplot2d(axes: Axes, x_list: np.ndarray, y_list: np.ndarray,
                x_label: str, y_label: str, scale: str, color: str) -> None:
    """2D trace, showing history of moves for two parameters"""
    axes.plot(x_list, y_list, color=color)
    axes.plot(x_list[0], y_list[0], marker=".", linestyle=" ", color='b',
                label="Start", markersize=6)
    axes.plot(x_list[-1], y_list[-1], marker=".", linestyle=" ", color='k',
                label="End", markersize=6)
    if scale == "symlog":
        axes.set_xscale(scale, linthresh=LINTHRESH)
        axes.set_yscale(scale, linthresh=LINTHRESH)
    else:
        axes.set_xscale(scale)
        axes.set_yscale(scale)
    #axes.legend()
    axes.set_xlabel(f"{x_label}")
    axes.set_ylabel(f"{y_label}")

def histogram1d(axes: Axes, x_list: np.ndarray, title: str, x_label: str, scale: str, bins: int,
                color: str) -> None:
    """1D histogram, showing distribution of values visited by one parameter"""
    axes.hist(x_list, bins, edgecolor='k', facecolor=color)
    if scale == "symlog":
        axes.set_yscale(scale, linthresh=LINTHRESH)
    else:
        axes.set_yscale(scale)
    axes.set_title(title)
    axes.set_ylabel("Counts")
    axes.set_xlabel(x_label)

def histogram2d(axes: Axes, x_list: np.ndarray, y_list: np.ndarray,
                x_label: str, y_label: str, scale: str, bins: int) -> None:
    """2D histogram, showing distribution of visited values for two parameters"""
    axes.hist2d(x_list, y_list, bins, cmap="Blues")
    axes.set_xscale(scale)
    if scale == "symlog":
        axes.set_xscale(scale, linthresh=LINTHRESH)
        axes.set_yscale(scale, linthresh=LINTHRESH)
    else:
        axes.set_xscale(scale)
        axes.set_yscale(scale)
    axes.set_xlabel(f"{x_label}")
    axes.set_ylabel(f"{y_label}")

def sim_plot(axes: Axes, x_list: np.ndarray, y_list: np.ndarray,
                  x_label: str, y_label: str, scale: str, color: str) -> None:
    """Ordinary plot of TRPL, TRTS, etc... decay, e.g. for quicksim feature"""
    axes.plot(x_list, y_list, color=color)
    axes.set_yscale(scale)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
