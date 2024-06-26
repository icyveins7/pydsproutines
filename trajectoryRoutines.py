# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:59:47 2020

@author: Seo
"""
from __future__ import annotations

import time
import numpy as np
from enum import IntEnum
import pyqtgraph as pg
from localizationRoutines import *
from timingRoutines import Timer

from scipy.constants import speed_of_light as lightspd

import warnings

try:
    import cupy as cp

    def calcFOA(r_x, r_xdot, t_x, t_xdot, freq=30e6):
        """
        Expects individual row vectors.
        All numpy array shapes expected to match.

        Assumed that arrays are either all cupy arrays or all numpy arrays,
        operates agnostically using cupy/numpy.
        """
        xp = cp.get_array_module(r_x)

        lightspd = 299792458.0

        radial = t_x - r_x  # convention pointing towards transmitter
        radial_n = radial / xp.linalg.norm(radial, axis=1).reshape(
            (-1, 1)
        )  # don't remove this reshape, nor the axis arg

        if radial_n.ndim == 1:
            vradial = xp.dot(radial_n, r_xdot) - xp.dot(
                radial_n, t_xdot
            )  # minus or plus?
        else:
            # vradial = np.zeros(len(radial_n))
            # for i in range(len(radial_n)):
            #     vradial[i] = np.dot(radial_n[i,:],r_xdot[i,:]) - np.dot(radial_n[i,:], t_xdot[i,:])

            # make distinct numpy calls instead of the loop
            dot_radial_r = xp.sum(radial_n * r_xdot, axis=1)
            dot_radial_t = xp.sum(radial_n * t_xdot, axis=1)
            vradial = dot_radial_r - dot_radial_t

        foa = vradial / lightspd * freq

        return foa

except:
    print("Skipping trajectoryRoutines cupy imports.")


# %%
class Trajectory:
    class TauMethod(IntEnum):
        First_Order_Velocity_Approximation = 0
        Position_Vector_Chasing_Iterator = 1

    def __init__(self, x0: np.ndarray):
        """
        Initialises a trajectory object.

        Parameters
        ----------
        x0 : np.ndarray
            Initial position vector. This is the position at t=0.
        """
        if x0.ndim != 1:
            raise ValueError("x0 must be a 1D array.")

        if x0.size != 2 and x0.size != 3:
            raise TypeError("x0 must represent a 2D or 3D point.")
        self._x0 = x0

    @property
    def x0(self):
        """
        Initial position vector.
        """
        return self._x0

    def at(self, t: np.ndarray):
        """
        Method to return the position at given time(s).
        """
        raise NotImplementedError("This is only implemented for subclasses.")

    def to(
        self,
        rx: Trajectory,
        t: np.ndarray,
        method: int = TauMethod.First_Order_Velocity_Approximation,
    ):
        """
        Method to return the time taken by a photon to travel from the current trajectory
        from time(s) 't' to when it hits the 'rx' trajectory.

        In signal processing terms, 't' is the known transmit time.

        In general, this is not just the time taken between the two trajectories at a given time point,
        as that does not take into account the distance travelled by the 'rx' while the photon is in flight.

        Work in progress.
        """
        if isinstance(rx, StationaryTrajectory):
            # Simply calculate the time directly
            return np.linalg.norm(self.at(t) - rx.at(t), axis=1) / lightspd
        elif method == Trajectory.TauMethod.First_Order_Velocity_Approximation:
            tau = self._quadraticVelocityMethod(rx, t)
            tau = np.max(
                tau, axis=1
            )  # Take the larger one, which is generally the positive one

            return tau

    def frm(
        self,
        tx: Trajectory,
        t: np.ndarray,
        method: int = TauMethod.First_Order_Velocity_Approximation,
    ):
        """
        Method to return the time taken by a photon to travel to the current trajectory
        at time(s) 't' from when it begins from the 'tx' trajectory.

        In signal processing terms, 't' is the known receive time.

        In general, this is not just the time taken between the two trajectories at a given time point,
        as that does not take into account the distance travelled by the 'tx' while the photon is in flight.
        """

        if isinstance(tx, StationaryTrajectory):
            # Simply calculate the time directly
            return np.linalg.norm(self.at(t) - tx.at(t), axis=1) / lightspd
        elif method == Trajectory.TauMethod.First_Order_Velocity_Approximation:
            tau = self._quadraticVelocityMethod(tx, t)
            if np.all(tau < 0):
                raise ValueError(
                    "Not sure how to select tau; both negative valued")
            tau = -np.min(
                tau, axis=1
            )  # Take the smaller one, which is generally the negative one

            return tau

    def _scalarToArray(self, t: np.ndarray, dtype: np.dtype = np.float64):
        """
        Returns t as a numpy array if it is a single Pythonic scalar value,
        otherwise just returns t.

        Parameters
        ----------
        t : np.ndarray
            The time vector or time scalar.
        dtype : np.dtype, optional
            The data type of the returned array. This is only used if t is a scalar.
            If it is already an array, it will be returned as-is.

        Returns
        -------
        np.ndarray
            The time vector.
        """
        if isinstance(t, float) or isinstance(t, int):
            t = np.array([t], dtype=dtype)
        return t

    # Tau approximation methods
    def _quadraticVelocityMethod(self, rx: ConstantVelocityTrajectory, t: np.ndarray):
        if isinstance(rx, ConstantVelocityTrajectory):
            # Define the initial connecting vector
            D = self.at(t) - rx.at(t)

            # Define the quadratic coefficients
            a = np.linalg.norm(rx.v) ** 2 - lightspd**2
            b = -2 * D @ rx.v.reshape((-1, 1))
            c = np.linalg.norm(D) ** 2

            # Solve the quadratic equation
            disc = b**2 - 4 * a * c
            tau = np.vstack((np.sqrt(disc), -np.sqrt(disc)))
            tau = (-b + tau) / (2 * a)
            tau = tau.T

            return tau
        else:
            raise TypeError(
                "Quadratic velocity method for to() is only applicable to ConstantVelocityTrajectory."
            )


class StationaryTrajectory(Trajectory):
    def at(self, t: np.ndarray):
        """
        Returns the constant position at all time points.

        Parameters
        ----------
        t : np.ndarray
            Array of time values at which to calculate the position.
            Must be a 1D array. A single value will be converted to a numpy array automatically.
        """
        t = self._scalarToArray(t)
        return self._x0 + np.zeros_like(t).reshape((-1, 1))


class ConstantVelocityTrajectory(Trajectory):
    def __init__(self, x0: np.ndarray, v: np.ndarray):
        super().__init__(x0)
        # Ensure same dimensions
        if v.shape != self.x0.shape:
            raise np.ValueError("v must be the same shape as x0.")
        self._v = v

    @property
    def v(self):
        """
        Velocity vector.
        """
        return self._v

    def at(self, t: np.ndarray):
        """
        Calculates the position at time t by multiplying the velocity vector.

        Parameters
        ----------
        t : np.ndarray
            Array of time values at which to calculate the position.
            Must be a 1D array. A single value will be converted to a numpy array automatically.
        """
        t = self._scalarToArray(t)
        if not isinstance(t, np.ndarray):
            raise TypeError("t must be a numpy array.")
        if t.ndim != 1:
            raise np.ValueError("t must be a 1D array.")

        return self._x0 + t.reshape((-1, 1)) * self._v


class InterpolatedTrajectory(Trajectory):
    def __init__(self, xp: np.ndarray, tp: np.ndarray):
        # Transpose into 3xN so that interpolation is easier
        self._xp = xp.T
        self._tp = tp  # This is a 1D array
        # Check if t=0 is defined
        if 0.0 >= self._tp[0] and 0.0 <= self._tp[-1]:
            # Derive the value at t=0
            x0 = np.vstack(
                (
                    np.interp(0.0, self._tp, self._xp[0, :]),
                    np.interp(0.0, self._tp, self._xp[1, :]),
                    np.interp(0.0, self._tp, self._xp[2, :]),
                )
            )
            x0 = x0.reshape(-1)
        else:
            x0 = None

        super().__init__(x0)

    @property
    def xp(self):
        """
        Defined position vectors.
        """
        return self._xp

    @property
    def tp(self):
        """
        Defined time points.
        """
        return self._tp


# %%
def createLinearTrajectory(totalSamples, pos1, pos2, speed, sampleTime, start_coeff=0):
    # Define connecting vector between two anchors
    dirVec = pos2 - pos1
    anchorDist = np.linalg.norm(dirVec)
    dirVecNormed = dirVec / np.linalg.norm(dirVec)

    # Define the start position
    pos_start = pos1 + start_coeff * dirVec

    # Calculate percentage of anchor-anchor distance travelled per sample
    distPerSample = sampleTime * speed
    percentPerSample = distPerSample / anchorDist

    # Formulate in terms of multiples of anchorDist
    percentAnchorDist = start_coeff + \
        np.arange(totalSamples) * percentPerSample

    # First, correct the ones that have returned full cycles
    percentAnchorDist = np.mod(
        percentAnchorDist, 2
    )  # 2 means it is back at anchor pos1 (not pos_start necessarily!)
    # Then, correct the ones which are in reverse direction
    reverseIdx = np.argwhere(percentAnchorDist > 1.0)
    percentAnchorDist[reverseIdx] = (
        2.0 - percentAnchorDist[reverseIdx]
    )  # this will move it backwards appropriately e.g. 1.1 -> 0.9

    # These indices have the velocities flipped
    r_xdot = (
        np.zeros((totalSamples, len(pos1))) + dirVecNormed * speed
    )  # everything is identical, except for the flips which are handled next
    r_xdot[reverseIdx, :] = -r_xdot[reverseIdx, :]

    # Now compute the positions r_x
    r_x = pos1 + percentAnchorDist.reshape((-1, 1)) * dirVec

    return r_x, r_xdot


def createCircularTrajectory(
    totalSamples,
    r_a=100000.0,
    desiredSpeed=100.0,
    r_h=300.0,
    sampleTime=3.90625e-6,
    phi=0,
):
    # initialize a bunch of rx points in a circle in 3d
    dtheta_per_s = desiredSpeed / r_a  # rad/s
    arcangle = totalSamples * sampleTime * dtheta_per_s  # rad
    r_theta = np.arange(phi, phi + arcangle, dtheta_per_s *
                        sampleTime)[:totalSamples]

    r_x_x = r_a * np.cos(r_theta)
    r_x_y = r_a * np.sin(r_theta)
    r_x_z = np.zeros(len(r_theta)) + r_h
    r_x = np.vstack((r_x_x, r_x_y, r_x_z)).transpose()

    r_xdot_x = r_a * -np.sin(r_theta) * dtheta_per_s
    r_xdot_y = r_a * np.cos(r_theta) * dtheta_per_s
    r_xdot_z = np.zeros(len(r_theta))
    r_xdot = np.vstack((r_xdot_x, r_xdot_y, r_xdot_z)).transpose()

    return r_x, r_xdot, arcangle, dtheta_per_s


def createTriangularSpacedPoints(
    numPts: int, dist: float = 1.0, startPt: np.ndarray = np.array([0, 0]), make3d=False
):
    """
    Spawns locations in a set, beginning with startPt. Each location is spaced
    'dist' apart from any other location, e.g.

       2      1

    3     O      0

       4      5

    The alignment is in the shape of triangles. The order of generation is anticlockwise as shown.

    """

    if numPts < 2:
        raise Exception("Please specify at least 2 points.")

    origin = np.array([0.0, 0.0])

    ptList = [origin]

    dirVecs = (
        np.array(
            [
                [1.0, 0.0],
                [0.5, np.sqrt(3) / 2],
                [-0.5, np.sqrt(3) / 2],
                [-1.0, 0.0],
                [-0.5, -np.sqrt(3) / 2],
                [0.5, -np.sqrt(3) / 2],
                [1.0, 0.0],
            ]
        )
        * dist
    )  # cyclical to ensure indexing later on

    layer1ptr = 0
    turnLayer = 0
    i = 1
    while i < numPts:
        idx = i - 1  # we go back to 0-indexing

        # test for layer
        layer = 1
        while idx >= (layer + 1) * (layer / 2) * 6:
            layer += 1

        # print("i: %d, idx: %d, layer: %d"% (i,idx,layer)) # verbose index printing

        if layer == 1:  # then it's simple, just take the genVec and propagate
            newPt = origin + dirVecs[idx]
            ptList.append(newPt)
            i += 1
        else:
            # use the pointer at layer 1
            layerptr = origin + dirVecs[layer1ptr]

            if turnLayer == 0:  # go straight all the way
                for d in range(layer - 1):
                    layerptr = layerptr + dirVecs[layer1ptr]
                ptList.append(np.copy(layerptr))
                turnLayer = layer - 1  # now set it to turn
            else:
                for d in range(turnLayer - 1):  # go straight for some layers
                    layerptr = layerptr + dirVecs[layer1ptr]
                for d in range(layer - turnLayer):
                    layerptr = layerptr + dirVecs[layer1ptr + 1]
                ptList.append(np.copy(layerptr))
                turnLayer = turnLayer - 1  # decrement
                if (
                    turnLayer == 0
                ):  # if we have hit turnLayer 0, time to move the layer1ptr
                    layer1ptr = (layer1ptr + 1) % 6

            i += 1

    # swap to array for cleanliness
    ptList = np.array(ptList)
    ptList = ptList + startPt  # move the origin

    if make3d:
        ptList = np.pad(ptList, ((0, 0), (0, 1)))  # make into 3-d

    return ptList


# %% Containers
class Transceiver:
    def __init__(
        self,
        x: np.ndarray,
        xdot: np.ndarray,
        t: np.ndarray,
        symbol: str = "x",
        symbolBrush: str = "b",
        symbolPen: str = "b",
    ):
        self.x = x
        self.xdot = xdot
        self.t = t

        self.symbol = symbol
        self.symbolBrush = symbolBrush
        self.symbolPen = symbolPen

    @classmethod
    def asStationary(cls, x: np.ndarray, t: np.ndarray):
        return cls(x, np.zeros(x.shape), t)

    @staticmethod
    def plotFlat2d(transceivers: list, idx: np.ndarray):
        win = pg.GraphicsLayoutWidget()
        ax = win.addPlot()
        for i, transceiver in enumerate(transceivers):
            if i > 0:
                assert np.all(transceiver.t == transceivers[0].t)
            # Plot the point
            ax.plot(
                transceiver.x[idx, 0],
                transceiver.x[idx, 1],
                pen=None,
                symbol=transceiver.symbol,
                symbolBrush=transceiver.symbolBrush,
                symbolPen=transceiver.symbolPen,
            )

        win.show()
        return win, ax


##########################
class Receiver(Transceiver):
    def __init__(
        self,
        x: np.ndarray,
        xdot: np.ndarray,
        t: np.ndarray,
        symbol: str = "x",
        symbolBrush: str = "r",
        symbolPen: str = "r",
    ):
        super().__init__(x, xdot, t, symbol, symbolBrush, symbolPen)


##########################
class Transmitter(Transceiver):
    def __init__(
        self,
        x: np.ndarray,
        xdot: np.ndarray,
        t: np.ndarray,
        symbol: str = "o",
        symbolBrush: str = "b",
        symbolPen: str = "b",
    ):
        super().__init__(x, xdot, t, symbol, symbolBrush, symbolPen)

    def theoreticalRangeDiff(self, rx1: Receiver, rx2: Receiver):
        assert np.all(self.t == rx1.t)
        assert np.all(self.t == rx2.t)
        range1 = np.linalg.norm(rx1.x - self.x, axis=1)
        range2 = np.linalg.norm(rx2.x - self.x, axis=1)
        return range2 - range1

    def plotHyperbolaFlat(
        self,
        rx1: Receiver,
        rx2: Receiver,
        rangediff: float = None,
        z: float = 0,
        ax: pg.PlotItem = None,
    ):
        if rangediff is None:
            rangediff = self.theoreticalRangeDiff(rx1, rx2)

        timer = Timer()
        timer.start()
        hyperbola = generateHyperbolaXY(
            200, rangediff, rx1.x[0], rx2.x[0], orthostep=0.1
        )
        timer.end()

        if ax is None:
            fig = pg.GraphicsLayoutWidget()
            ax = fig.addPlot()

        hypItem = ax.plot(hyperbola[:, 0], hyperbola[:, 1], pen="k")
        hypItem.setSymbol

        return hyperbola, hypItem


# %%
if __name__ == "__main__":
    from timingRoutines import Timer

    timer = Timer()
    from plotRoutines import *

    closeAllFigs()
    rxHeight = 1
    rxA = Receiver.asStationary(np.array([[0, -1, rxHeight]]), np.array([0]))
    rxB = Receiver.asStationary(np.array([[0, +1, rxHeight]]), np.array([0]))
    tx = Transmitter.asStationary(np.array([[0, 0.51, 0]]), np.array([0]))

    rd = tx.theoreticalRangeDiff(rxA, rxB)
    print(rd)

    win, ax = tx.plotFlat2d([rxA, rxB, tx], np.array([0]))

    from localizationRoutines import *

    lightspd = 299792458.0
    xr = np.arange(-10, 10, 0.1)
    yr = np.arange(-12, 12, 0.1)
    costgrid = gridSearchTDOA(
        rxA.x, rxB.x, rd / lightspd, np.array([1e-9]), xr, yr, 0)

    pgPlotHeatmap(
        np.exp(-costgrid.reshape((yr.size, xr.size)).T),
        xr[0],
        yr[0],
        xr[-1] - xr[0],
        yr[-1] - yr[0],
        window=ax,
        autoBorder=True,
    )

    # Test hyperbola plots
    # timer.start()
    hyperbola, hypItem = tx.plotHyperbolaFlat(rxA, rxB, ax=ax)
    # tx.plotHyperbolaFlat(rxA, rxB, ax=ax)
    # timer.end()
    # hypItem.setSymbol('x')

    # Checking
    plt.plot(
        np.linalg.norm(hyperbola - rxB.x[0], axis=1)
        - np.linalg.norm(hyperbola - rxA.x[0], axis=1)
    )
    plt.hlines([rd], 0, hyperbola.shape[0], colors="r", linestyles="dashed")
