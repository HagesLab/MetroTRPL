# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:25:44 2022

@author: cfai2
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import SymmetricalLogLocator, NullFormatter
import numpy as np
from math import floor, ceil


def make_1D_tracker(x, all_accepted, **kwargs):
    mark_value = kwargs.get("mark_value", None)
    size = kwargs.get("size", (3.5, 2.7))
    ylabel = kwargs.get("ylabel", "")
    xlabel = kwargs.get("xlabel", r"$N$")
    ylim = kwargs.get("ylim", None)
    do_log = kwargs.get("do_log", 0)
    show_legend = kwargs.get("show_legend", 0)
    colors = kwargs.get("colors", None)
    lw = kwargs.get("linewidth", 1)

    fig = kwargs.get("fig", None)
    ax = kwargs.get("ax", None)

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, dpi=200, figsize=size)

    if all_accepted.ndim == 1:
        all_accepted = np.expand_dims(all_accepted, axis=0)

    if isinstance(colors, str) or colors is None:
        colors = [colors] * len(all_accepted)

    for i in range(len(all_accepted)):
        # ax.plot(all_proposed, label="Proposed samples", color='steelblue', zorder=1)
        ax.plot(x, all_accepted[i], color=colors[i], linewidth=lw, zorder=2)
    if mark_value is not None:
        ax.axhline(mark_value, linewidth=0.6, color='r',
                   linestyle='dashed', label="Actual value", zorder=3)

    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if do_log:
        ax.set_yscale('log')

    if show_legend:
        ax.legend()


def make_1D_histo(accepted, **kwargs):
    mark_value = kwargs.get("mark_value", None)
    size = kwargs.get("size", (2.5, 2.5))
    xlabel = kwargs.get("xlabel", "")
    ylabel = kwargs.get("ylabel", "")
    do_log = kwargs.get("do_log", 0)
    bin_count = kwargs.get("bin_count", 96)
    axis_overrides = kwargs.get("axis_overrides", None)
    fig = kwargs.get("fig", None)
    ax = kwargs.get("ax", None)

    minx = min(accepted.flatten())
    maxx = max(accepted.flatten())

    if do_log:
        accepted = np.log10(accepted)
        minx = np.log10(minx)
        maxx = np.log10(maxx)
        if mark_value is not None:
            mark_value = np.log10(mark_value)

        if axis_overrides is not None:
            axis_overrides = (
                np.log10(axis_overrides[0]), np.log10(axis_overrides[1]))

    bins_x = np.arange(bin_count+1)

    bins_x = minx + (maxx-minx)*(bins_x)/bin_count
    h = np.histogram(accepted.flatten(), bins=bins_x, density=True)
    h_mass = np.cumsum(h[0] / np.sum(h[0]))

    h_05 = np.where(h_mass < 0.05)[0]
    if len(h_05) == 0:
        h_05 = 0
    else:
        h_05 = h_05[-1]

    h_95 = np.where(h_mass > 0.95)[0]
    if len(h_95) == 0:
        h_95 = -1
    else:
        h_95 = h_95[0]

    h_05 = bins_x[h_05]
    h_95 = bins_x[h_95]
    if do_log:
        h_05 = 10 ** h_05
        h_95 = 10 ** h_95

    print("90% CI: {} to {}".format(h_05, h_95))

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, dpi=200, figsize=size)
    ax.bar(bins_x[:-1], h[0], width=np.diff(bins_x), align='edge')

    if mark_value is not None:
        ax.axvline(mark_value, color='r', linewidth=0.6)

    if do_log:
        decades = np.arange(floor(minx), ceil(maxx) + 1)
        if len(decades) == 1:
            decades = np.arange(decades[0] - 1, decades[-1] + 2)
        ax.set_xticks(decades)
        ax.set_xticks(np.concatenate(
            [d + np.log10(np.linspace(1, 10, 10)) for d in decades]), minor=True)
        ax.set_xlim((decades[0], decades[-1]))
        ax.set_xticklabels([r"10$^{{{}}}$".format(d) for d in decades])

        # ax.set_xticklabels([r"$10^{{{}}}$".format(int(d)) for d in np.concatenate(
        #     [d + np.log10(np.linspace(1, 10, 70)) for d in decades])], minor=True)
        # ax.tick_params(axis='x', colors='white')

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if axis_overrides is not None:
        ax.set_xlim(axis_overrides)

    fig.tight_layout()


def make_2D_histo(x_accepted, y_accepted, **kwargs):
    assert x_accepted.shape == y_accepted.shape

    contour_info = kwargs.get("contours", None)
    if contour_info is not None:
        assert len(contour_info) == 4  # (x, y, Z(x,y), levels)
    xlabel = kwargs.get("xlabel", "")
    ylabel = kwargs.get("ylabel", "")
    logx = kwargs.get("do_logx", 0)
    logy = kwargs.get("do_logy", 0)
    markx = kwargs.get("markx", None)
    marky = kwargs.get("marky", None)
    bin_count = kwargs.get("bin_count", 96)
    axis_overrides = kwargs.get("axis_overrides", (None, None))

    minx = min(x_accepted.flatten())
    maxx = max(x_accepted.flatten())
    miny = min(y_accepted.flatten())
    maxy = max(y_accepted.flatten())
    if contour_info is not None:
        cx = contour_info[0]
        cy = contour_info[1]
        cZ = contour_info[2]
        clevels = contour_info[3]

    else:
        cx = None
        cy = None
        cZ = None
        clevels = None

    if logx:
        x_accepted = np.log10(x_accepted)
        minx = np.log10(minx)
        maxx = np.log10(maxx)
        if markx is not None:
            markx = np.log10(markx)
        if cx is not None:
            cx = np.log10(cx)
        if axis_overrides[0] is not None:
            axis_overrides[0] = (
                np.log10(axis_overrides[0][0]), np.log10(axis_overrides[0][1]))

    if logy:
        y_accepted = np.log10(y_accepted)
        miny = np.log10(miny)
        maxy = np.log10(maxy)
        if marky is not None:
            marky = np.log10(marky)
        if cy is not None:
            cy = np.log10(cy)
        if axis_overrides[1] is not None:
            axis_overrides[1] = (
                np.log10(axis_overrides[1][0]), np.log10(axis_overrides[1][1]))

    bins_x = np.arange(bin_count+1)
    bins_x = minx + (maxx-minx)*(bins_x)/bin_count
    bins_y = np.arange(bin_count+1)
    bins_y = miny + (maxy-miny)*(bins_y)/bin_count
    grid_y, grid_x = np.meshgrid(bins_x, bins_y)

    h2d = np.histogram2d(x_accepted.flatten(), y_accepted.flatten(), bins=[
                         bins_x, bins_y], density=True)

    fig, ax2d = plt.subplots(1, 1, figsize=(2, 2), dpi=120)
    ax2d.pcolormesh(grid_y, grid_x, h2d[0].T, cmap='Blues')
    if markx is not None and marky is not None:
        ax2d.scatter(markx, marky, c='r', marker='s', s=8)

    if contour_info is not None:
        cwg = ax2d.contour(cx, cy, cZ, levels=clevels,
                           colors='black', zorder=9999)
        ax2d.clabel(cwg)

    if logy:
        decades = np.arange(floor(min(y_accepted.flatten())),
                            floor(max(y_accepted.flatten())) + 1)
        if len(decades) == 1:
            decades = np.arange(decades[0] - 1, decades[-1] + 2)
        ax2d.set_yticks(decades)
        ax2d.set_yticks(np.concatenate(
            [d + np.log10(np.linspace(1, 10, 10)) for d in decades]), minor=True)
        ax2d.set_ylim((miny, maxy))
        ax2d.set_yticklabels([r"$10^{{{}}}$".format(d) for d in decades])
    if logx:
        decades = np.arange(floor(min(x_accepted.flatten())),
                            floor(max(x_accepted.flatten())) + 1)
        if len(decades) == 1:
            decades = np.arange(decades[0] - 1, decades[-1] + 2)
        ax2d.set_xticks(decades)
        ax2d.set_xticks(np.concatenate(
            [d + np.log10(np.linspace(1, 10, 10)) for d in decades]), minor=True)
        ax2d.set_xlim((minx, maxx))
        ax2d.set_xticklabels([r"$10^{{{}}}$".format(d) for d in decades])

    # ax2d.set_xlabel(xlabel)
    # ax2d.set_ylabel(ylabel)
    # ax2d.axvline(np.log10(9e-29), color='k', linestyle='dashed', linewidth=0.6)

    if axis_overrides is not None:
        ax2d.set_xlim(axis_overrides[0])
        ax2d.set_ylim(axis_overrides[1])
    fig.tight_layout()


def make_2D_tracker(x_accepted, y_accepted, **kwargs):
    assert x_accepted.shape == y_accepted.shape

    contour_info = kwargs.get("contours", None)
    if contour_info is not None:
        assert len(contour_info) == 4  # (x, y, Z(x,y), levels)
    xlabel = kwargs.get("xlabel", "")
    ylabel = kwargs.get("ylabel", "")
    colors = kwargs.get("colors", None)
    logx = kwargs.get("do_logx", 0)
    logy = kwargs.get("do_logy", 0)
    markx = kwargs.get("markx", None)
    marky = kwargs.get("marky", None)
    axis_overrides = kwargs.get("axis_overrides", None)
    lw = kwargs.get("linewidth", 1)

    fig = kwargs.get("fig", None)
    ax2d = kwargs.get("ax", None)
    size = kwargs.get("size", (2, 2))

    if fig is None or ax2d is None:
        fig, ax2d = plt.subplots(1, 1, figsize=size, dpi=200)

    if x_accepted.ndim == 1 and y_accepted.ndim == 1:
        x_accepted = np.expand_dims(x_accepted, axis=0)
        y_accepted = np.expand_dims(y_accepted, axis=0)

    if isinstance(colors, str) or colors is None:
        colors = [colors] * len(x_accepted)

    for i in range(len(x_accepted)):
        ax2d.plot(x_accepted[i], y_accepted[i], color=colors[i], linewidth=lw,
                  )
        # ax2d.scatter(x_accepted[0], y_accepted[0], c='purple', zorder=99)
        # ax2d.scatter(x_accepted[-1], y_accepted[-1], c='red', marker='s', zorder=99)

    # if first:
    #     import os
    #     def fmt(s):
    #         return r"$10^{{{:.0f}}}$".format(s)
    #     xyz = np.load(os.path.join("..", "bay_outputs", "kp0grid.npy"))
    #     Y_corr, X_corr, marP = xyz
    #     im_corr = ax2d.contour(10 ** Y_corr, 10 ** X_corr, marP, levels=np.arange(0,7,1), colors='k', zorder=0)
    #     #ax2d.clabel(im_corr, im_corr.levels, inline=False, fmt=fmt,fontsize=10, zorder=0)

    if markx is not None and marky is not None:
        ax2d.scatter(markx, marky, c='r', marker='*', zorder=99)

    if contour_info is not None:
        cx = contour_info[0]
        cy = contour_info[1]
        cZ = contour_info[2]
        clevels = contour_info[3]
        cwg = ax2d.contour(cx, cy, cZ, levels=clevels, colors='black', zorder=0)
        # ax2d.clabel(cwg)

    ax2d.set_xlabel(xlabel)
    ax2d.set_ylabel(ylabel)
    if logy:
        ax2d.set_yscale('log')

    if logx:
        ax2d.set_xscale('log')

    if axis_overrides is not None:
        ax2d.set_ylim(axis_overrides[1])
        ax2d.set_xlim(axis_overrides[0])

    # if first:
    #     cbar = fig.colorbar(ScalarMappable(), ax=ax2d, ticks=[0,1])
    #     cbar.ax.set_yticklabels(["Initial", "Final"])
    fig.tight_layout()


def plot_history(x, y, **kwargs):
    """ Plot a collection of simple 1D curves with data (x,y).
        y is 2D - one list per curve. If 1D, promotes to 2D.
        x is 1D - shared over curves
        kwargs as needed for matplotlib customization.
    """
    title = kwargs.get("title", "")
    xlabel = kwargs.get("xlabel", "")
    ylabel = kwargs.get("ylabel", "")
    xlim = kwargs.get("xlim", None)
    ylim = kwargs.get("ylim", None)
    do_log = kwargs.get("do_log", False)
    fig = kwargs.get("fig", None)
    ax = kwargs.get("ax", None)
    labels = kwargs.get("labels", "")
    marker = kwargs.get("marker", None)
    colors = kwargs.get("colors", None)
    lw = kwargs.get("linewidth", 1)
    yticks = kwargs.get("yticks", None)
    yticklabels = kwargs.get("yticklabels", None)

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=120)

    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)

    if labels:
        ax.legend(framealpha=1)

    if isinstance(labels, str):
        labels = [labels] * len(y)

    if isinstance(colors, str) or colors is None:
        colors = [colors] * len(y)

    for i, yy in enumerate(y):
        ax.plot(x, yy, marker=marker,
                label=labels[i], color=colors[i], linewidth=lw)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if yticks is None:
        if do_log:
            ax.set_yscale("symlog")
            locmaj = SymmetricalLogLocator(base=10, linthresh=1)
            ax.yaxis.set_major_locator(locmaj)
            locmin = SymmetricalLogLocator(
                base=10.0, subs=(np.arange(0.1, 1, 0.1)), linthresh=1)
            ax.yaxis.set_minor_locator(locmin)
            ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_yticks(yticks)

    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        # pass
        ylim = ax.get_ylim()

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_title(title)
