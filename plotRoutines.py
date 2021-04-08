# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:09:54 2020

@author: Seo
"""


import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtCore import Qt, QRectF
import numpy as np
import matplotlib.pyplot as plt
from signalCreationRoutines import makeFreq
from matplotlib import cm
from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

def pgPlotDeltaFuncs(fig, x, h, color='r', symbol=None, name=None):
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
    
    if not hasattr(h, "__len__"):
        h = np.zeros(x.shape) + h
        
    
    for i in range(len(x)):
        if i == 0: # use a uniform legend for all the delta points otherwise will flood the legend
            legendname = name
        else:
            legendname = None
            
        if h[i] != 0:
            fig.plot([x[i],x[i]], [0, h[i]], pen=pg.mkPen(color=color, style=Qt.DashLine), name=legendname)
        if symbol is not None:
            fig.plot([x[i]], [h[i]], symbol=symbol, symbolPen=color, symbolBrush=color, name=legendname)
        
def pgPlotSurface(x, y, z, shader='normalColor', autoscale=True, title=None):
    '''
    Adds a new window with a surface plot and a grid item.
    
    Returns
    w = view widget item
    g = grid item
    p = plot item
    '''
    
    # win = pg.GraphicsWindow()
    w = gl.GLViewWidget()
    w.show()
    w.setWindowTitle(title)
    
    g = gl.GLGridItem()
    
    if autoscale == True:
        sx = np.max(np.abs(x))
        sy = np.max(np.abs(y))
        g.scale(sx,sy,1)
        
        w.setCameraPosition(distance=np.max([sx,sy]) * 2)
        
    g.setDepthValue(10)  # draw grid after surfaces since they may be translucent
    w.addItem(g)
    
    p = gl.GLSurfacePlotItem(x=x, y=y, z=z, shader=shader)
    w.addItem(p)
    
    return w, g, p

def pgPlotHeatmap(heatmap, x0, y0, width, height, window=None):
    '''
    This is a useful tool to overlay heatmaps onto normal scatter plots,
    in the mathematical x-y axis (unlike the conventional image axis which has y-axis flipped).
    
    heatmap: 2-D array containing the data.
    x0,y0 : coordinates of bottom-left most point.
    width, height: scale of the heatmap.
    '''
    if window is None:
        window = pg.plot()
    
    # create image item
    img = pg.ImageItem(heatmap)
    img.setRect(QRectF(x0,y0,width,height))
    img.setZValue(-100) # to ensure it's behind everything else
    
    # pick one to turn into an actual colormap
    # cm2use = pg.ColorMap(*zip(*Gradients["bipolar"]["ticks"])) # from pyqtgraph gradients
    cm2use = pg.colormap.getFromMatplotlib('viridis')
    img.setLookupTable(cm2use.getLookupTable())
    
    window.addItem(img)
    window.show()
    
    return window, img

def pgPlotPhasorVsTime(complexData, color=(1.0,1.0,1.0,1.0), start=0, end=200, scale='auto', view=None):
    
    if view is None:
        view = gl.GLViewWidget()
        view.show()
        ## create three grids, add each to the view
        xgrid = gl.GLGridItem()
        ygrid = gl.GLGridItem()
        zgrid = gl.GLGridItem()
        view.addItem(xgrid)
        view.addItem(ygrid)
        view.addItem(zgrid)
        
        ## create axis?
        axis = gl.GLAxisItem()
        view.addItem(axis)
        
        ## rotate x and y grids to face the correct direction
        xgrid.rotate(90, 0, 1, 0)
        ygrid.rotate(90, 1, 0, 0)
        
    plotdata = np.vstack((np.real(complexData.flatten()), np.arange(len(complexData)), np.imag(complexData.flatten()))).T
    
    if scale=='auto':
        magicFactor = 10 # this is the number of squares you want it to fill really
        scale = magicFactor / np.abs(end-start)
        plotdata[:,1] = (plotdata[:,1]-start) * scale
    else:
        plotdata[:,1] = (plotdata[:,1]-start) * scale
    
    
    lineItem = gl.GLLinePlotItem(pos = plotdata[start:end,:], color=color)
    view.addItem(lineItem)
    
    # # print helpful things
    # print("pyqtgraph uses mousewheel hold + drag to pan the camera.")
    
    return view

def plotSpectra(dataList, fs, labels=None, colors=None, windowTitle=None, title=None):    
    fig = plt.figure(windowTitle)
    ax = fig.add_subplot(111)
    
    for i in range(len(dataList)):
        spec = 20*np.log10(np.abs(np.fft.fft(dataList[i])))
        if colors is not None:
            ax.plot(makeFreq(len(spec), fs[i]), spec, colors[i])
        else:
            ax.plot(makeFreq(len(spec), fs[i]), spec)
        
    if labels is not None:
        plt.legend(labels)
    
    plt.title(title)
    
    return fig, ax