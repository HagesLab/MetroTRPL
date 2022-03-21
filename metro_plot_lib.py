# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:25:44 2022

@author: cfai2
"""
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
from math import floor, ceil

def make_1D_tracker(all_proposed, all_accepted, **kwargs):
    mark_value = kwargs.get("mark_value", None)
    size = kwargs.get("size", (3.5,2.7))
    ylabel = kwargs.get("ylabel", "")
    xlabel = kwargs.get("xlabel", "Sample #")
    do_log = kwargs.get("do_log", 0)
    show_legend = kwargs.get("show_legend", 0)
    did_multicore = kwargs.get("did_multicore", (all_proposed.ndim == 2))
    
    fig, ax = plt.subplots(1,1,dpi=200, figsize=size)
    if did_multicore:
        for i in range(len(all_proposed)):
            if i == 0:
                labelp = "Proposed"
                labela = "Accepted"
            else:
                labelp = None
                labela = None
            
            ax.plot(all_proposed[i], label=labelp, color='steelblue', zorder=1)
            ax.plot(all_accepted[i], label=labela, color='orange', zorder=2)
    
    else:
        ax.plot(all_proposed, label="Proposed samples", color='steelblue', zorder=1)
        ax.plot(all_accepted, label="Accepted samples", color='orange', zorder=2)
    if mark_value is not None:
        ax.axhline(mark_value, color='r', linestyle='dashed', label="Actual value", zorder=3)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if do_log: ax.set_yscale('log')
    
    if show_legend:
        ax.legend()

def make_1D_histo(accepted, **kwargs):
    mark_value = kwargs.get("mark_value", None)
    size = kwargs.get("size", (2.5,2.5))
    xlabel = kwargs.get("xlabel", "")
    do_log = kwargs.get("do_log", 0)
    bin_count = kwargs.get("bin_count", 96)
    did_multicore = kwargs.get("did_multicore", (accepted.ndim == 2))
    
    minx = min(accepted.flatten())
    maxx = max(accepted.flatten())
    
    if do_log:
        accepted = np.log10(accepted)
        minx = np.log10(minx)
        maxx = np.log10(maxx)
        mark_value = np.log10(mark_value)
    
    bins_x = np.arange(bin_count+1)

    bins_x   = minx + (maxx-minx)*(bins_x)/bin_count
    h = np.histogram(accepted.flatten(), bins=bins_x, density=True)
    
    fig, ax = plt.subplots(1,1,dpi=200, figsize=size)
    ax.bar(bins_x[:-1], h[0], width=np.diff(bins_x), align='edge')
    
    if mark_value is not None: 
        ax.axvline(mark_value, color='r',linewidth=1)
    
    if do_log:
        decades = np.arange(floor(minx), ceil(maxx) + 1)
        if len(decades) == 1:
            decades = np.arange(decades[0] - 1, decades[-1] + 2)
        ax.set_xticks(decades)
        ax.set_xlim((decades[0], decades[-1]))
        ax.set_xticklabels([r"$10^{{{}}}$".format(d) for d in decades])
    
    ax.set_ylabel(f"P({xlabel})")
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    
def make_2D_histo(x_accepted, y_accepted, **kwargs):
    assert x_accepted.shape == y_accepted.shape
    
    contour_info = kwargs.get("contours", None)
    if contour_info is not None:
        assert len(contour_info) == 4 # (x, y, Z(x,y), levels)
    xlabel = kwargs.get("xlabel", "")
    ylabel = kwargs.get("ylabel", "")
    logx = kwargs.get("do_logx", 0)
    logy = kwargs.get("do_logy", 0)
    markx = kwargs.get("markx", None)
    marky = kwargs.get("marky", None)
    bin_count = kwargs.get("bin_count", 96)
    did_multicore = kwargs.get("did_multicore", (x_accepted.ndim == 2))
    
    minx = min(x_accepted.flatten())
    maxx = max(x_accepted.flatten())
    miny = min(y_accepted.flatten())
    maxy = max(y_accepted.flatten())
    if contour_info is not None:
        cx = contour_info[0]
        cy = contour_info[1]
        cZ = contour_info[2]
        clevels = contour_info[3]
    
    if logx:
        x_accepted = np.log10(x_accepted)
        minx = np.log10(minx)
        maxx = np.log10(maxx)
        markx = np.log10(markx)
        cx = np.log10(cx)
        
    if logy:
        y_accepted = np.log10(y_accepted)
        miny = np.log10(miny)
        maxy = np.log10(maxy)
        marky = np.log10(marky)
        cy = np.log10(cy)
        
    bins_x = np.arange(bin_count+1)
    bins_x   = minx + (maxx-minx)*(bins_x)/bin_count
    bins_y = np.arange(bin_count+1)
    bins_y   = miny + (maxy-miny)*(bins_y)/bin_count
    grid_y, grid_x = np.meshgrid(bins_x, bins_y)
    
    h2d = np.histogram2d(x_accepted.flatten(), y_accepted.flatten(), bins=[bins_x,bins_y], density=True)
    
    
    fig, ax2d = plt.subplots(1,1,figsize=(3.5,3.5), dpi=120)
    ax2d.pcolormesh(grid_y, grid_x, h2d[0].T, cmap='Blues')
    if markx is not None and marky is not None:
        ax2d.scatter(markx, marky, c='r', marker='s')
        
    if contour_info is not None:
        cwg = ax2d.contour(cx, cy, cZ, levels=clevels, colors='black', zorder=9999)
        ax2d.clabel(cwg)        

    if logy:
        decades = np.arange(floor(min(y_accepted.flatten())), floor(max(y_accepted.flatten())) + 1)
        if len(decades) == 1:
            decades = np.arange(decades[0] - 1, decades[-1] + 2)
        ax2d.set_yticks(decades)
        ax2d.set_ylim((miny, maxy))
        ax2d.set_yticklabels([r"$10^{{{}}}$".format(d) for d in decades])
    if logx:
        decades = np.arange(floor(min(x_accepted.flatten())), floor(max(x_accepted.flatten())) + 1)
        if len(decades) == 1:
            decades = np.arange(decades[0] - 1, decades[-1] + 2)
        ax2d.set_xticks(decades)
        ax2d.set_xlim((minx, maxx))
        ax2d.set_xticklabels([r"$10^{{{}}}$".format(d) for d in decades])
        
    ax2d.set_xlabel(xlabel)
    ax2d.set_ylabel(ylabel)
    fig.tight_layout()
    
        
def make_2D_tracker(x_accepted, y_accepted, **kwargs):
    assert x_accepted.shape == y_accepted.shape
    
    contour_info = kwargs.get("contours", None)
    if contour_info is not None:
        assert len(contour_info) == 4 # (x, y, Z(x,y), levels)
    xlabel = kwargs.get("xlabel", "")
    ylabel = kwargs.get("ylabel", "")
    logx = kwargs.get("do_logx", 0)
    logy = kwargs.get("do_logy", 0)
    markx = kwargs.get("markx", None)
    marky = kwargs.get("marky", None)
    did_multicore = kwargs.get("did_multicore", (x_accepted.ndim == 2))
    
    fig, ax2d = plt.subplots(1,1,figsize=(3.5*1.1,3.5), dpi=120)
    
    
    
    if did_multicore:
        
        for i in range(len(x_accepted)):
            color_grad = np.linspace(0, 1, len(x_accepted[i]))
            ax2d.scatter(x_accepted[i], y_accepted[i], c=color_grad)
    else:
        color_grad = np.linspace(0, 1, len(x_accepted))
        ax2d.scatter(x_accepted, y_accepted, c=color_grad)
    if markx is not None and marky is not None:
        ax2d.scatter(markx, marky, c='r', marker='s')

    if contour_info is not None:
        cx = contour_info[0]
        cy = contour_info[1]
        cZ = contour_info[2]
        clevels = contour_info[3]
        cwg = ax2d.contour(cx, cy, cZ, levels=clevels, colors='black', zorder=9999)
        ax2d.clabel(cwg)        

    ax2d.set_xlabel(xlabel)
    ax2d.set_ylabel(ylabel)
    if logy:
        ax2d.set_yscale('log')
    if logx:
        ax2d.set_xscale('log')
    cbar = fig.colorbar(ScalarMappable(), ax=ax2d, ticks=[0,1])
    cbar.ax.set_yticklabels(["Initial", "Final"])
    fig.tight_layout()