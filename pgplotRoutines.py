# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:09:54 2020

@author: Seo
"""


import pyqtgraph as pg
from PyQt5.QtCore import Qt

def pgPlotDeltaFuncs(fig, x, h, color='r'):
    '''
    Adds delta function dashed lines to the specified pyqtgraph plot.

    Parameters
    ----------
    fig : PyQtgraph figure.
        
    x : List of x values where the delta functions should be plotted
        
    h : Height of the delta functions to be plotted.
        

    Returns
    -------
    None.

    '''
    for i in x:
        fig.plot([i,i], [0, h], pen=pg.mkPen(color=color, style=Qt.DashLine))