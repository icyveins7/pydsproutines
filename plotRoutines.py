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
    PYQTGRAPH surface plotter.
    Adds a new window with a surface plot and a grid item.
    
    x: 1-d array
    y: 1-d array
    z: 2-d array (usually created by function on xm, ym; xm & ym are outputs from np.meshgrid(x,y))
    
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
        sx = np.abs(np.max(x)-np.min(x))
        sy = np.abs(np.max(y)-np.min(y))
        # sx = np.max(np.abs(x)).astype(np.float64)
        # sy = np.max(np.abs(y)).astype(np.float64)
        g.scale(sx,sy,1)
        
        w.setCameraPosition(distance=np.max([sx,sy]) * 2)
        
    g.setDepthValue(10)  # draw grid after surfaces since they may be translucent
    w.addItem(g)
    
    p = gl.GLSurfacePlotItem(x=x.astype(np.float64), y=y.astype(np.float64), z=z, shader=shader)
    w.addItem(p)
    
    return w, g, p

def plotSurface(xm, ym, z, cmap='coolwarm'):
    '''
    Matplotlib Surface Plotter.
    
    Parameters
    ----------
    xm : 2-d array
        Meshgrid of x
    ym : 2-d array
        Meshgrid of y
    z : 2-d array
        Function values.
    cmap : optional
        Matplotlib colormapping. The default is 'coolwarm'.

    Returns
    -------
    Matplotlib Figure and Axes objects.
    '''
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(xm,ym,z,cmap=cmap)
    
    return fig, ax

def pgPlotHeatmap(heatmap, x0, y0, width, height, window=None, imgLvls=None):
    '''
    This is a useful tool to overlay heatmaps onto normal scatter plots,
    in the mathematical x-y axis (unlike the conventional image axis which has y-axis flipped).
    
    heatmap: 2-D array containing the data.
    x0,y0 : coordinates of bottom-left most point.
    width, height: scale of the heatmap.
    imgLvls: list_like, passed to img.setLevels(), specifies colour limits to values
    '''
    if window is None:
        window = pg.plot()
    
    # create image item
    img = pg.ImageItem(heatmap)
    img.setRect(QRectF(x0,y0,width,height))
    img.setZValue(-100) # to ensure it's behind everything else
    if imgLvls is not None:
        img.setLevels(imgLvls)
    
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

def plotSpectra(dataList, fs, labels=None, colors=None, windowTitle=None, title=None, ax=None):    
    if ax is None: # hurray for python scoping allowing this
        fig = plt.figure(windowTitle)
        ax = fig.add_subplot(111)
    else:
        fig = None
    
    for i in range(len(dataList)):
        spec = 20*np.log10(np.abs(np.fft.fft(dataList[i])))
        if colors is not None:
            ax.plot(makeFreq(len(spec), fs[i]), spec, colors[i])
        else:
            ax.plot(makeFreq(len(spec), fs[i]), spec)
        
    if labels is not None:
        ax.legend(labels)
    
    ax.set_title(title)
    
    return fig, ax


def plotTrajectory2d(r_x, r_xdot=None, r_xfmt='b.', quiver_scale=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1,1)
        
    # plot the points
    ax.plot(r_x[:,0], r_x[:,1], r_xfmt)
    
    # get some scaling based on the positions if not supplied
    if quiver_scale is None:
        quiver_scale = np.mean(np.linalg.norm(np.diff(r_x, axis=0), axis=1))
    
    # plot the velocity vectors as quivers
    r_xdot_normed = r_xdot / np.linalg.norm(r_xdot, axis=1).reshape((-1,1))
    ax.quiver(r_x[:,0], r_x[:,1], r_xdot_normed[:,0] * quiver_scale, r_xdot_normed[:,1] * quiver_scale, scale_units='xy', angles='xy', scale=1)
    
    ax.axis('equal')
    
    return ax

def plotConstellation(syms, fmt='.', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1,1)
        
    ax.plot(np.real(syms), np.imag(syms), fmt)
    ax.axis('eaual')
    
    return ax

def mplBtnToggle(p, fig):
    '''
    Binds 'a' to reset and show all plots.
    Binds 't' to toggle one plot at a time.
    
    Parameters
    ----------
    p : List of plot items.
        Example: 
            line = ax.plot(np.sin(np.arange(0,10,0.01)))
            line2 = ax.plot(2*np.sin(np.arange(0,10,0.01)))
            p = [line,line2]
    fig : Figure object.
        Returned from 
        fig = plt.figure() and similar calls.

    Returns
    -------
    None.

    '''
    # One line flattener
    pl = []
    _ = [pl.extend(b) if hasattr(b,'__len__') else pl.append(b) for b in p] # result is irrelevant, pl is extended in place
    
    def btnToggle(event):
        if event.key == 'a': # Default to turning everything on
            for i in pl:
                i.set_visible(True)
        elif event.key == 't': # Swap one at a time
            if all([i.get_visible() for i in pl]):
                for i in pl:
                    i.set_visible(False)
                pl[0].set_visible(True)
            else:
                for i in pl:
                    i.set_visible(not i.get_visible())
        fig.canvas.draw()
        
    # Connect the button
    fig.canvas.mpl_connect('key_press_event', btnToggle)
    
