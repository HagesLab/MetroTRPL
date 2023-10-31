# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 12:05:40 2022

@author: amurad2
"""

from window import Window

APPLICATION_NAME = "MCMC Visualization"

if __name__ == "__main__":
    window = Window(1000, 800, APPLICATION_NAME)
    window.mainloop()
