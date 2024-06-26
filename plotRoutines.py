# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:09:54 2020

@author: Seo
"""


import pyqtgraph as pg
import pyqtgraph.opengl as gl


import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
from signalCreationRoutines import makeFreq
from matplotlib import cm
from pyqtgraph.graphicsItems.GradientEditorItem import Gradients
from timingRoutines import Timer

try:  # Although this is the recommendation in requirements.txt, this does not work with Spyder
    from PySide6.QtCore import Qt, QRectF
    from PySide6.QtWidgets import QApplication
except ImportError:  # This occurs when using spyder which requires pyqt5 instead
    from PyQt5.QtCore import Qt, QRectF
    from PyQt5.QtWidgets import QApplication


def closeAllFigs():
    """Helper function to close both Pyqtgraph and Matplotlib windows."""
    QApplication.closeAllWindows()
    plt.close("all")


def _getPgColourRotation() -> list:
    """
    Load a list of colours to rotate for pyqtgraph,
    like how matplotlib does it.
    """
    colourRotation = plt.rcParams["axes.prop_cycle"].by_key()[
        "color"
    ]  # This is in strings like '#1f77b4'
    colourRotation.insert(
        0, "w"
    )  # Put white first cause that's the default pyqtgraph colour
    return colourRotation


def pgRender():
    """
    On Mac OS X, this function is used to render the plot.
    Otherwise you might get a hanging window.
    """
    pg.mkQApp().exec()


def pgPlotDeltaFuncs(fig, x, h, color="r", symbol=None, name=None):
    """
    Adds delta function dashed lines to the specified pyqtgraph plot.

    Parameters
    ----------
    fig : PyQtgraph figure.

    x : List of x values where the delta functions should be plotted

    h : Height of the delta functions to be plotted.


    Returns
    -------
    None.

    """

    if not hasattr(h, "__len__"):
        h = np.zeros(x.shape) + h

    for i in range(len(x)):
        if (
            i == 0
        ):  # use a uniform legend for all the delta points otherwise will flood the legend
            legendname = name
        else:
            legendname = None

        if h[i] != 0:
            fig.plot(
                [x[i], x[i]],
                [0, h[i]],
                pen=pg.mkPen(color=color, style=Qt.DashLine),
                name=legendname,
            )
        if symbol is not None:
            fig.plot(
                [x[i]],
                [h[i]],
                symbol=symbol,
                symbolPen=color,
                symbolBrush=color,
                name=legendname,
            )


def pgPlotSurface(x, y, z, shader="normalColor", autoscale=True, title=None):
    """
    PYQTGRAPH surface plotter.
    Adds a new window with a surface plot and a grid item.

    x: 1-d array
    y: 1-d array
    z: 2-d array (usually created by function on xm, ym; xm & ym are outputs from np.meshgrid(x,y))

    Returns
    w = view widget item
    g = grid item
    p = plot item
    """

    # win = pg.GraphicsWindow()
    w = gl.GLViewWidget()
    w.show()
    w.setWindowTitle(title)

    g = gl.GLGridItem()

    if autoscale == True:
        sx = np.abs(np.max(x) - np.min(x))
        sy = np.abs(np.max(y) - np.min(y))
        # sx = np.max(np.abs(x)).astype(np.float64)
        # sy = np.max(np.abs(y)).astype(np.float64)
        g.scale(sx, sy, 1)

        w.setCameraPosition(distance=np.max([sx, sy]) * 2)

    # draw grid after surfaces since they may be translucent
    g.setDepthValue(10)
    w.addItem(g)

    p = gl.GLSurfacePlotItem(
        x=x.astype(np.float64), y=y.astype(np.float64), z=z, shader=shader
    )
    w.addItem(p)

    return w, g, p


def plotSurface(xm, ym, z, cmap="coolwarm"):
    """
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
    """
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(xm, ym, z, cmap=cmap)

    return fig, ax


def plotHeatmap(
    heatmap, x0, y0, width, height, ax=None, aspect="auto", vmin=None, vmax=None
):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None

    plot = ax.imshow(
        heatmap,
        origin="lower",
        aspect=aspect,
        vmin=vmin,
        vmax=vmax,  # this decides the color bar mapping
        extent=(x0, x0 + width, y0, y0 + height),
    )
    fig.colorbar(plot, ax=ax)

    return fig, ax


def pgPlotHeatmap(
    heatmap, x0, y0, width, height, window=None, imgLvls=None, autoBorder=False
):
    """
    This is a useful tool to overlay heatmaps onto normal scatter plots,
    in the mathematical x-y axis (unlike the conventional image axis which has y-axis flipped).

    heatmap: 2-D array containing the data.
    x0,y0 : coordinates of bottom-left most point.
    width, height: scale of the heatmap.
    imgLvls: list_like, passed to img.setLevels(), specifies colour limits to values
    autoBorder: configures whether to pad half a bin width around the image, usually the data is generated around the bin 'centres', so this is required
    """
    if window is None:
        window = pg.plot()

    if autoBorder:
        # Correct for half the bin widths
        xstep = width / heatmap.shape[0]
        ystep = height / heatmap.shape[1]
        # width = width + xstep
        # height = height + ystep
        x0 = x0 - xstep / 2
        y0 = y0 - ystep / 2

    # create image item
    img = pg.ImageItem(heatmap)
    img.setRect(QRectF(x0, y0, width, height))
    img.setZValue(-100)  # to ensure it's behind everything else
    if imgLvls is not None:
        img.setLevels(imgLvls)

    # pick one to turn into an actual colormap
    # cm2use = pg.ColorMap(*zip(*Gradients["bipolar"]["ticks"])) # from pyqtgraph gradients
    cm2use = pg.colormap.getFromMatplotlib("viridis")
    img.setLookupTable(cm2use.getLookupTable())

    window.addItem(img)
    window.show()

    return window, img


def pgPlotPhasorVsTime(
    complexData, color=(1.0, 1.0, 1.0, 1.0), start=0, end=200, scale="auto", view=None
):

    if view is None:
        view = gl.GLViewWidget()
        view.show()
        # create three grids, add each to the view
        xgrid = gl.GLGridItem()
        ygrid = gl.GLGridItem()
        zgrid = gl.GLGridItem()
        view.addItem(xgrid)
        view.addItem(ygrid)
        view.addItem(zgrid)

        # create axis?
        axis = gl.GLAxisItem()
        view.addItem(axis)

        # rotate x and y grids to face the correct direction
        xgrid.rotate(90, 0, 1, 0)
        ygrid.rotate(90, 1, 0, 0)

    plotdata = np.vstack(
        (
            np.real(complexData.flatten()),
            np.arange(len(complexData)),
            np.imag(complexData.flatten()),
        )
    ).T

    if scale == "auto":
        magicFactor = 10  # this is the number of squares you want it to fill really
        scale = magicFactor / np.abs(end - start)
        plotdata[:, 1] = (plotdata[:, 1] - start) * scale
    else:
        plotdata[:, 1] = (plotdata[:, 1] - start) * scale

    lineItem = gl.GLLinePlotItem(pos=plotdata[start:end, :], color=color)
    view.addItem(lineItem)

    # # print helpful things
    # print("pyqtgraph uses mousewheel hold + drag to pan the camera.")

    return view


def plotRealImag(
    dataList,
    fs,
    labels=None,
    colors=None,
    windowTitle=None,
    title=None,
    ax=None,
    idxbounds=None,
):
    if ax is None:  # hurray for python scoping allowing this
        fig = plt.figure(windowTitle)
        ax = fig.add_subplot(111)
    else:
        fig = None

    for i, data in enumerate(dataList):
        real = np.real(data)
        imag = np.imag(data)
        t = np.arange(data.size) / fs[i]
        if idxbounds is not None:
            t = t[idxbounds[i][0]: idxbounds[i][1]]
            real = real[idxbounds[i][0]: idxbounds[i][1]]
            imag = imag[idxbounds[i][0]: idxbounds[i][1]]
        if colors is not None:
            ax.plot(t, real, colors[i] + "-")
            ax.plot(t, imag, colors[i] + "--")
        else:
            ax.plot(t, real, "-")
            ax.plot(t, imag, "--")

    if labels is not None:
        reimlabels = [
            [labels[i] + "Real", labels[i] + "Imag"] for i in range(len(labels))
        ]
        reimlabels = [i for sublist in reimlabels for i in sublist]
        ax.legend(reimlabels)

    ax.set_title(title)

    return fig, ax


# Pyqtgraph version
def pgPlotAmpTime(
    dataList: list,
    fs: list = None,
    labels: list = None,
    colors: list = None,
    windowTitle: str = None,
    title: str = None,
    ax: pg.PlotItem = None,
):
    """
    Uses pyqtgraph to plot the amplitude vs time for a list of data arrays.

    Parameters
    ----------
    dataList : list of np.ndarray, or single np.ndarray
        The data to be plotted. np.abs() will be called for each array in the list.
    fs : int or float, or list of ints/floats, optional
        Sampling rate(s) of the data. Time vector will be generated for each data array input.
        Defaults to None, which will generate using fs=1 for each data array input.
    labels : list of str, optional
        Labels for each data array in the legend, by default None,
        which will result in no legend being displayed.
    colors : list of str, optional
        Colour strings supplied to the 'pen' argument, by default None.
        If this is None, a standard rotation of colours is used.
    windowTitle : str, optional
        Title of the window, by default None
    title : str, optional
        Title of the plot, usually used if multiple plots are in the same window. By default None
    ax : pg.PlotItem, optional
        The PlotItem generated by addPlot, usually used if the plot is added manually. By default None

    Returns
    -------
    win : pg.GraphicsLayoutWidget or None
        The window containing the plot. This is a newly generated window if the input 'ax' parameter was not supplied.
        Otherwise, it is None. You can use this to .addPlot() if you'd like.
    ax : pg.PlotItem
        The PlotItem containing the plotted data. This is the same as the input 'ax' parameter if it was supplied.
        Otherwise it is a newly (automatically) generated PlotItem.
    """
    if ax is None:
        win = pg.GraphicsLayoutWidget(title=windowTitle)
        ax = win.addPlot(title=title)
        win.show()
    else:
        win = None

    if labels is not None:
        ax.addLegend()

    # Handle case where dataList is a single item and not a list
    if not isinstance(dataList, list) and dataList.ndim == 1:
        dataList = [
            dataList
        ]  # We make it a list so the rest of the processing is identical

    # fs is often the same for everything
    if isinstance(fs, int) or isinstance(fs, float):
        fs = [fs] * len(dataList)
    elif fs is None:
        fs = [1] * len(dataList)

    # Load the default colour rotation from matplotlib
    colourRotation = _getPgColourRotation()

    for i, data in enumerate(dataList):
        amp = np.abs(data)
        t = np.arange(len(data)) / fs[i]  # Don't use .size so lists are okay
        if colors is not None:
            ax.plot(
                t, amp, pen=colors[i], name=labels[i] if labels is not None else None
            )
        else:
            ax.plot(
                t,
                amp,
                pen=colourRotation[i % len(colourRotation)],
                name=labels[i] if labels is not None else None,
            )

    return win, ax


def plotAmpTime(
    dataList,
    fs,
    labels=None,
    colors=None,
    windowTitle=None,
    title=None,
    ax=None,
    idxbounds=None,
):
    if ax is None:  # hurray for python scoping allowing this
        fig = plt.figure(windowTitle)
        ax = fig.add_subplot(111)
    else:
        fig = None

    for i, data in enumerate(dataList):
        amp = np.abs(data)
        t = np.arange(data.size) / fs[i]
        if idxbounds is not None:
            t = t[idxbounds[i][0]: idxbounds[i][1]]
            amp = amp[idxbounds[i][0]: idxbounds[i][1]]
        if colors is not None:
            ax.plot(t, amp, colors[i])
        else:
            ax.plot(t, amp)

    if labels is not None:
        ax.legend(labels)

    ax.set_title(title)

    return fig, ax


# Pyqtgraph version
def pgPlotSpectra(
    dataList: list,
    fs: list = None,
    nfft: list = None,
    labels: list = None,
    colors: list = None,
    windowTitle: str = None,
    title: list = None,
    ax: pg.PlotItem = None,
):
    """
    Uses pyqtgraph to plot the spectra (PSD) for a list of data arrays.

    Parameters
    ----------
    dataList : list of np.ndarray, or single np.ndarray
        The data to be plotted. np.abs(20*np.log10(np.fft.fft())) will be called for each array in the list.
    fs : int or float, or list of ints/floats, optional
        Sampling rate of the data. Frequency vector will be generated for each data array input.
        Defaults to None, which will generate using fs=1 for each data array input.
    nfft : list of ints/floats, optional
        The FFT size to use for each spectrum. Defaults to None, which will
        use the length of the data arrays as the FFT size.
    labels : list of str, optional
        Labels for each data array in the legend, by default None,
        which will result in no legend being displayed.
    colors : list of str, optional
        Colour strings supplied to the 'pen' argument, by default None.
        If this is None, a standard rotation of colours is used.
    windowTitle : str, optional
        Title of the window, by default None
    title : str, optional
        Title of the plot, usually used if multiple plots are in the same window. By default None
    ax : pg.PlotItem, optional
        The PlotItem generated by addPlot, usually used if the plot is added manually. By default None

    Returns
    -------
    win : pg.GraphicsLayoutWidget or None
        The window containing the plot. This is a newly generated window if the input 'ax' parameter was not supplied.
        Otherwise, it is None. You can use this to .addPlot() if you'd like.
    ax : pg.PlotItem
        The PlotItem containing the plotted data. This is the same as the input 'ax' parameter if it was supplied.
        Otherwise it is a newly (automatically) generated PlotItem.
    """

    if ax is None:
        win = pg.GraphicsLayoutWidget(title=windowTitle)
        ax = win.addPlot(title=title)
        win.show()
    else:
        win = None

    if labels is not None:
        ax.addLegend()

    # Handle case where dataList is a single item and not a list
    if not isinstance(dataList, list) and dataList.ndim == 1:
        dataList = [
            dataList
        ]  # We make it a list so the rest of the processing is identical

    # fs is often the same for everything
    if isinstance(fs, int) or isinstance(fs, float):
        fs = [fs] * len(dataList)
    elif fs is None:
        fs = [1] * len(dataList)

    colourRotation = _getPgColourRotation()

    for i, data in enumerate(dataList):
        spec = 20 * np.log10(np.abs(np.fft.fft(data, n=nfft)))
        if nfft is None:
            freqs = makeFreq(len(spec), fs[i])
        else:
            freqs = makeFreq(nfft, fs[i])

        if colors is not None:
            ax.plot(
                freqs,
                spec,
                pen=colors[i],
                name=labels[i] if labels is not None else None,
            )
        else:
            ax.plot(
                freqs,
                spec,
                pen=colourRotation[i % len(colourRotation)],
                name=labels[i] if labels is not None else None,
            )

    return win, ax


def plotSpectra(
    dataList,
    fs,
    nfft=None,
    labels=None,
    colors=None,
    windowTitle=None,
    title=None,
    ax=None,
):
    if ax is None:  # hurray for python scoping allowing this
        fig = plt.figure(windowTitle)
        ax = fig.add_subplot(111)
    else:
        fig = None

    for i in range(len(dataList)):
        spec = 20 * np.log10(np.abs(np.fft.fft(dataList[i], n=nfft)))
        if nfft is None:
            freqs = makeFreq(len(spec), fs[i])
        else:
            freqs = makeFreq(nfft, fs[i])

        if colors is not None:
            ax.plot(freqs, spec, colors[i])
        else:
            ax.plot(freqs, spec)

    if labels is not None:
        ax.legend(labels)

    ax.set_title(title)

    return fig, ax


# %%
def pgPlotAmpTimeChannels(chnls, chnl_fs, windowTitle=None, equalYScale=False):
    win = pg.GraphicsLayoutWidget(title=windowTitle)
    ax = []
    t = np.arange(chnls.shape[0]) / chnl_fs

    for i in range(chnls.shape[1]):
        p = win.addPlot(row=i, col=0)
        p.addLegend()
        p.plot(
            t, np.abs(chnls[:, -1 - i]), name="Channel %d" % (chnls.shape[1] - 1 - i)
        )
        if i > 0:
            p.setXLink(ax[0])

        ax.append(p)

    if equalYScale:
        maxamp = np.max(np.abs(chnls.flatten()))
        for p in ax:
            p.setYRange(0, maxamp)

    win.show()

    return win, ax


# %%
def plotTrajectory2d(r_x, r_xdot=None, r_xfmt="b.", quiver_scale=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # plot the points
    ax.plot(r_x[:, 0], r_x[:, 1], r_xfmt)

    # get some scaling based on the positions if not supplied
    if quiver_scale is None:
        quiver_scale = np.mean(np.linalg.norm(np.diff(r_x, axis=0), axis=1))

    # plot the velocity vectors as quivers
    r_xdot_normed = r_xdot / np.linalg.norm(r_xdot, axis=1).reshape((-1, 1))
    ax.quiver(
        r_x[:, 0],
        r_x[:, 1],
        r_xdot_normed[:, 0] * quiver_scale,
        r_xdot_normed[:, 1] * quiver_scale,
        scale_units="xy",
        angles="xy",
        scale=1,
    )

    ax.axis("equal")

    return ax


def plotConstellation(syms, fmt=".", labels=None, ax=None):
    """
    Plots a constellation for syms.
    Can be called with a single array or a list of arrays.
    Single array

    Parameters
    ----------
    syms : np.ndarray (complex64/128) or list of such arrays
        Input array(s).
    fmt : str or list of str, optional
        Format(s) to plot the constellation points. The default is '.'.
    labels : str or list of str, optional
        Label(s) for each input array. The default is None.
    ax : axes object, optional
        Axes object to plot on.
        The default is None, which generates a new figure and axes.

    Returns
    -------
    ax : Matplotlib.pyplot axes
        The axes object. Can be used to replot other things.

    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if isinstance(syms, list):
        for si, sym in enumerate(syms):
            ax.plot(np.real(sym), np.imag(sym), fmt[si], label=labels[si])
    else:
        ax.plot(np.real(syms), np.imag(syms), fmt, label=labels)

    if labels is not None:
        ax.legend()
    ax.axis("equal")

    return ax


def plotPossibleConstellations(
    syms_rs: np.ndarray, osr: int, fmt: str = ".", ax: np.ndarray = None
):
    """
    Convenience function to plot all possible resample points to find an eye-opening
    for time synchronization for one signal.
    Uses plotConstellation for each resample index.
    """
    # Create enough subplots
    rows = int(osr**0.5)
    cols = osr // rows if osr % rows == 0 else osr // rows + 1
    if ax is None:
        fig, ax = plt.subplots(rows, cols)

    for i in range(osr):
        plotConstellation(syms_rs[i::osr], fmt=fmt, ax=ax[i // cols, i % cols])
        ax[i // cols, i % cols].set_title("%d::%d" % (i, osr))
    return ax


def plotFreqz(taps: np.ndarray | list, cutoff: float = None, showPhase: bool = False):
    """
    Plots filter response for given taps array (or list of arrays).

    Parameters
    ----------
    taps : np.ndarray | list
        Taps vectors or list of it.
    cutoff : float, optional
        Cutoff for the filter taps, by default None.
        Will plot a vertical line to indicate the desired
        lowpass band if supplied.
    showPhase : bool, optional
        Plots the phase response if True. Defaults to False.

    Returns
    -------
    fig : Matplotlib.pyplot figure
        Figure that you can use for other plotting.
    ax : Matplotlib.pyplot axes
        Axes that you can use for other plotting.
    """
    if not isinstance(taps, list):
        taps = [taps]

    numPlotRows = 2 if showPhase else 1
    fig, ax = plt.subplots(
        numPlotRows, 1, num="Filter performance", sharex=True)
    aax = ax[0] if showPhase else ax
    pax = ax[1] if showPhase else None
    for i, vtaps in enumerate(taps):
        w, h = sps.freqz(vtaps, 1, vtaps.size)
        aax.plot(
            w / np.pi, 20 * np.log10(np.abs(h)), label="%d: %d taps" % (i, vtaps.size)
        )
        if showPhase:
            pax.plot(
                w / np.pi, np.unwrap(np.angle(h)), label="%d: %d taps" % (i, vtaps.size)
            )

    aax.legend()
    aax.set_xlabel("Normalised Frequency")
    aax.set_ylabel("Amplitude (dB)")
    if showPhase:
        pax.legend()
        pax.set_xlabel("Normalised Frequency")
        pax.set_ylabel("Phase (radians)")

    if cutoff is not None:
        yl = aax.get_ylim()
        aax.vlines([cutoff], yl[0], yl[1], colors="k", linestyle="dashed")
        if showPhase:
            pax.vlines([cutoff], yl[0], yl[1], colors="k", linestyle="dashed")

    return fig, ax


def plotAngles(
    angles: np.ndarray,
    colour: str = "b",
    label: str = None,
    showCircle: bool = False,
    showConnectors: bool = False,
    ax=None,
):
    x = np.cos(angles)
    y = np.sin(angles)

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None

    ax.plot(x, y, colour + "x", label=label)
    if showCircle:
        cx = np.cos(np.arange(0, 2 * np.pi, 0.001))
        cy = np.sin(np.arange(0, 2 * np.pi, 0.001))
        ax.plot(cx, cy, "k--")

    if showConnectors:
        for i in range(len(x)):
            ax.plot([0, x[i]], [0, y[i]], colour + "-")

    ax.axis([-1.1, 1.1, -1.1, 1.1])
    ax.set_aspect("equal")

    return fig, ax


def plotXcorrResults1D(
    td_scan_range: np.ndarray,
    qf2: np.ndarray,
    freqinds: np.ndarray = None,
    windowTitle: str = None,
    maxIdx: int = None,
):
    # Plot 2 rows if freqinds is specified
    if freqinds is not None:
        fig, ax = plt.subplots(2, 1, num=windowTitle, sharex=True)
        ax[0].plot(td_scan_range, qf2)
        ax[1].plot(td_scan_range, freqinds)
        ax[1].set_xlabel("TDOA (s)")
        ax[0].set_ylabel("$QF^2$")
        ax[1].set_ylabel("Max Freq. Index")

        # Get the maximum and show it for convenience
        if maxIdx is not None:
            tdest = td_scan_range[maxIdx]
            qf2est = qf2[maxIdx]
            freqIdxest = freqinds[maxIdx]
            ax[0].plot(tdest, qf2est, "rx")
            ax[1].plot(tdest, freqIdxest, "rx")
            ax[0].set_title(
                "$TD_{est} = %g, QF^2 = %g, f_i = %d$" % (
                    tdest, qf2est, freqIdxest)
            )

    # Otherwise just plot the qf2
    else:
        fig, ax = plt.subplots(2, 1, num=windowTitle, sharex=True)
        ax.plot(td_scan_range, qf2)
        ax.set_xlabel("TDOA (s)")
        ax.set_ylabel("$QF^2$")

        # Get the maximum and show it for convenience
        if maxIdx is not None:
            tdest = td_scan_range[maxIdx]
            qf2est = qf2[maxIdx]
            ax.plot(tdest, qf2est, "rx")
            ax.set_title("$TD_{est} = %g, QF^2 = %g$" % (tdest, qf2est))

    return fig, ax


def mplBtnToggle(p, fig):
    """
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

    """
    # One line flattener
    pl = []
    _ = [
        pl.extend(b) if hasattr(b, "__len__") else pl.append(b) for b in p
    ]  # result is irrelevant, pl is extended in place

    def btnToggle(event):
        if event.key == "a":  # Default to turning everything on
            for i in pl:
                i.set_visible(True)
        elif event.key == "t":  # Swap one at a time
            if all([i.get_visible() for i in pl]):
                for i in pl:
                    i.set_visible(False)
                pl[0].set_visible(True)
            else:
                for i in pl:
                    i.set_visible(not i.get_visible())
        fig.canvas.draw()

    # Connect the button
    fig.canvas.mpl_connect("key_press_event", btnToggle)


def reverseMapToPixels(x: np.ndarray, y: np.ndarray,
                       xLim: tuple[float, float],
                       yLim: tuple[float, float],
                       xPixels: int = 2000,
                       yPixels: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """
    Maps the data points to pixel coordinates.
    This allows the plotted points to be compressed when
    point density is high, reducing the total number of points
    passed to the plotter.

    Example:
        Resolution of a pixel is 1 unit * 1 unit.
        Two closely spaced data points are
        A: (0, 0)
        B: (0, 0.1)

        Assuming a pixel centre is at (0,0), we can safely plot a
        single point to represent both A and B, as it would be
        indistinguishable at the pixel level;
        (0,1) is too far away to accurately represent B.

    This function is not intended for heatmaps; each point is assigned to a
    single specific pixel coordinate, without any extra 'spreading' or
    anti-aliasing effects.

    Parameters
    ----------
    x : np.ndarray
        Input x data points.
    y : np.ndarray
        Input y data points.
    xLim : tuple[float, float]
        X-axis limits for the pixels, (min, max) inclusive.
    yLim : tuple[float, float]
        Y-axis limits for the pixels, (min, max) inclusive.
    xPixels : int
        Number of pixels for the x-axis.
    yPixels : int
        Number of pixels for the y-axis.

    Returns
    -------
    xCoords : np.ndarray
        The (potentially compressed) x coordinates.
    yCoords : np.ndarray
        The (potentially compressed) y coordinates.
    """
    timer = Timer()
    timer.start()
    xSpacing = (xLim[1] - xLim[0]) / (xPixels - 1)
    ySpacing = (yLim[1] - yLim[0]) / (yPixels - 1)
    timer.evt("spacing")

    # Then calculate the index of each point
    xInd = np.round((x - xLim[0]) / xSpacing).astype(np.int32)
    yInd = np.round((y - yLim[0]) / ySpacing).astype(np.int32)
    timer.evt("indices")

    # We make sure we only keep those that are in range
    kept = np.logical_and(
        np.logical_and(xInd >= 0, xInd < xPixels),
        np.logical_and(yInd >= 0, yInd < yPixels)
    )
    xInd = xInd[kept]
    yInd = yInd[kept]
    timer.evt("kept")

    # Now we find only the unique indices
    uInd = np.unique(np.vstack((xInd, yInd)), axis=1)
    timer.evt("unique")

    # Then extract the coordinates of the used pixels
    xCoords = uInd[0] * xSpacing + xLim[0]
    yCoords = uInd[1] * ySpacing + yLim[0]
    timer.end("coords")

    return xCoords, yCoords


# %% testing
if __name__ == "__main__":
    import unittest
    from timingRoutines import Timer

    # Also do some simple benchmarking
    # This must be before unittest.main
    timer = Timer()
    timer.start()
    x = np.random.randn(1000000)
    y = np.random.randn(1000000)
    xPixels, yPixels = (2000, 2000)
    xx, yy = reverseMapToPixels(x, y, (-1, 1), (-1, 1), xPixels, yPixels)
    timer.end("reverseMap %d pts to (%d, %d)" % (
        x.size, xPixels, yPixels))

    # Write the actual tests
    class TestPixelMapping(unittest.TestCase):
        def test_reverseMapToPixels(self):
            data = np.array([
                [0, 0],
                [0, 0.1],
                [0, 0.2],  # Mapped to (0, 0)
                [0, 0.3],
                [0, 0.4],
                [0, 0.5],
                [0, 0.6],
                [0, 0.7],  # Mapped to (0, 0.5)
                [0, 0.8],
                [0, 0.9],  # Mapped to (0, 1.0)
                [0.1, 0],
                [0.1, 0.1],
                [0.1, 0.2],  # Mapped to (0, 0) as well
                [0.1, 0.3],
                [0.1, 0.4],
                [0.1, 0.5],
                [0.1, 0.6],
                [0.1, 0.7],  # Mapped to (0, 0.5)
                [0.1, 0.8],
                [0.1, 0.9]  # Mapped to (0, 1.0)
            ])
            x = data[:, 0]
            y = data[:, 1]
            xLim = (0, 1)
            yLim = (0, 1)
            xPixels = 3  # Map to (0, 0.5, 1)
            yPixels = 3  # Same as x
            xCoords, yCoords = reverseMapToPixels(
                x, y, xLim, yLim, xPixels, yPixels)
            # Note that the ordering is not necessarily maintained.
            # In order to test it more easily we stack to 2d
            coords = np.vstack((xCoords, yCoords)).T
            # And then we convert it to list of tuples for checking
            # We do this by turning into list of lists,
            # then converting to list of tuples
            coords = list(map(tuple, coords.tolist()))
            # Change it into an unordered set for comparison
            coords = set(coords)
            # Then check our 3 points
            self.assertEqual(
                coords,
                set([(0, 0), (0, 0.5), (0, 1.0)])
            )

        def test_reverseMapToPixels_outofbounds(self):
            pass  # TODO

    unittest.main()
