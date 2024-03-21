#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 23:14:33 2021

@author: seolubuntu
"""

import numpy as np
import scipy as sp
import scipy.signal as sps
from signalCreationRoutines import makeFreq


# %%
class PSKOrderDetector:

    # Define the possible m values
    m_p = [2, 4, 8]

    def __init__(self, max_m: int):
        if max_m not in [4, 8]:
            raise ValueError("Max order 'm' must be 4 or 8.")

        self.max_m = max_m

        # Store internal workspaces
        self.mi = None
        self.peaks = None
        self.ratios = None

    def estimateOrder(self, x: np.ndarray, threshold: float = 0.2):
        """
        Estimates the order of the PSK signal.
        This is done by appropriate squaring of the signal and spectral
        peak detection.

        A peak value P of |F((A x[n])^m)| is given by:
            A^2 L
            A^4 L
            A^8 L

        As such, each step up in the PSK modulation order is a scaling of
        P -> (P/L)^2 * L = (P^2 / L). In real world signals, the scaling is not exact,
        but is usually within an order of magnitude.

        This scaling is used to estimate the peak height of the next order,
        to determine (along with the peak frequency) the PSK order.

        Parameters
        ----------
        x : np.ndarray, N x L
            The N PSK signal(s) of length L.
            Each row is processed separately
            i.e. treated as an individual signal.
        threshold : float
            The threshold to distinguish a higher/lower pair of orders.
            Not advisable to be below 0.1.
            Example:
                BPSK vs QPSK, <= 0.2 will be QPSK, > 0.2 will be BPSK.

        Returns
        -------
        order : np.ndarray
            The order of the PSK signal(s).

        """

        # If 1-D, reshape to 2-D
        if x.ndim == 1:
            x = x.reshape((1, -1))

        numIter = self.m_p.index(self.max_m) + 1
        N, L = x.shape

        # Populate internal workspace
        self._computeCmMaxes(x, numIter, N)

        # Now perform predictions
        order = self._orderFromRatios(numIter, N, L, threshold)

        return order

    def _computeCmMaxes(self, x: np.ndarray, numIter: int, N: int):
        """
        Internal method to compute the internal self.mi, self.peaks matrices.
        """
        # Instantiate internal workspace (can be accessed for debugging)

        self.mi = np.zeros((numIter, N), dtype=np.uint32)
        self.peaks = np.zeros((numIter, N), dtype=np.float64)

        # Populate the workspace
        _rowIndexer = np.arange(N)
        for i in range(numIter):
            # Square as required
            x = x * x
            # Get the magnitude of spectrum for each row
            xf = np.abs(np.fft.fft(x, axis=1))
            # Take argmax and max along each row
            self.mi[i, :] = np.argmax(xf, axis=1)
            self.peaks[i, :] = xf[
                _rowIndexer, self.mi[i, :]
            ]  # cannot use : for the rows

    def _orderFromRatios(self, numIter: int, N: int, L: int, threshold: float):
        """
        Internal method to compute the order by comparing the predicted peak values
        with the actual peak values.
        """
        order = np.zeros(N, dtype=np.uint8)
        self.ratios = np.zeros((numIter - 1, N), dtype=np.float64)
        for i in range(1, numIter):
            prediction = (self.peaks[i - 1, :] / L) ** 2 * L
            self.ratios[i - 1, :] = prediction / self.peaks[i]
            # Estimate the order
            order[self.ratios[i - 1, :] > threshold] = self.m_p[i - 1]

        # At the end, the remainder are the max_m
        order[order == 0] = self.max_m

        return order


# %%
def estimateBaud(x: np.ndarray, fs: float):
    """
    Estimates baud rate of signal. (CM21)

    Parameters
    ----------
    x : np.ndarray
        Signal vector.
    fs : float
        Sample rate.

    Returns
    -------
    estBaud : float
        Estimated baudrate.
    idx1
        First index of fft vector used. The index is a peak that was found after
        applying fftshift to the fft of the signal. That is, the peak value should
        be "fftshift(fft(abs(signal)))[idx1]".
    idx2
        Second index of fft vector used. Similar to the first.
    Xf
        fftshift(fft(abs(signal))) i.e. the FFT of the abs signal, described in idx1.
    freq
        freq vector (fft shifted) to apply the indices idx1 and idx2 to directly.

    """
    Xf = sp.fft.fftshift(sp.fft.fft(np.abs(x)))
    Xfabs = np.abs(Xf)
    freq = sp.fft.fftshift(makeFreq(x.size, fs))
    # Find the peaks
    peaks, _ = sps.find_peaks(Xfabs)
    prominences = sps.peak_prominences(Xfabs, peaks)[0]
    # Sort prominences
    si = np.argsort(prominences)
    peaks = peaks[si]
    b1 = freq[peaks[-2]]  # 2nd highest, 1st highest is the centre
    b2 = freq[peaks[-3]]  # 3rd highest

    # Average the 2
    estBaud = (np.abs(b1) + np.abs(b2)) / 2

    return estBaud, peaks[-2], peaks[-3], Xf, freq


# %%
def estimateOffsetViaCM(x: np.ndarray, fs: float, order: int) -> float:
    """
    Performs the CMX0 frequency offset estimation.

    Parameters
    ----------
    x : np.ndarray
        Input array
    fs : float
        Sampling rate of the input.
    order : int
        The order of the CM. E.g. 2 for CM20, 4 for CM40.

    Returns
    -------
    offset: float
        The frequency offset. Shift by -offset to centre the signal.
    """
    # Power the signal
    xp = x**order

    # Take the FFT
    xpf = np.fft.fft(xp)

    # Find the max value
    mi = np.argmax(xpf)
    freqvec = makeFreq(x.size, fs)
    freqpeak = freqvec[mi]
    offset = freqpeak / order

    return offset


# %%
try:
    import cupy as cp
    from cupyExtensions import *

    # %%
    class PSKOrderDetectorCupy(PSKOrderDetector):
        def _computeCmMaxes(self, x: cp.ndarray, numIter: int, N: int):
            """
            Internal method to compute the internal self.mi, self.peaks matrices.
            """
            # Make sure it's on device
            requireCupyArray(x)

            # Instantiate internal workspace (can be accessed for debugging)
            self.mi = cp.zeros((numIter, N), dtype=np.uint32)
            self.peaks = cp.zeros((numIter, N), dtype=np.float32)

            # Populate the workspace
            _rowIndexer = np.arange(N)
            for i in range(numIter):
                # Square as required
                x = x * x
                # # Get the magnitude of spectrum for each row
                # xf = cp.abs(cp.fft.fft(x, axis=1))
                # Take argmax and max along each row
                # self.mi[i,:] = cp.argmax(xf, axis=1)

                # For cupy, we perform the abs and max in the kernel instead
                xf = cp.fft.fft(x, axis=1)
                cupyArgmaxAbsRows_complex64(
                    xf, self.mi[i, :], self.peaks[i, :], True, 64
                )

                # self.mi[i,:] = cupyArgmaxAbsRows_complex64(xf, 64)
                # self.peaks[i,:] = cp.abs(xf[_rowIndexer, self.mi[i,:]]) # cannot use : for the rows

        def _orderFromRatios(self, numIter: int, N: int, L: int, threshold: float):
            """
            Internal method to compute the order by comparing the predicted peak values
            with the actual peak values.
            """
            order = cp.zeros(N, dtype=np.uint8)
            self.ratios = cp.zeros((numIter - 1, N), dtype=np.float64)
            for i in range(1, numIter):
                # prediction = (self.peaks[i-1,:] / L)**2 * L
                prediction = self.peaks[i - 1, :] ** 2 / L
                self.ratios[i - 1, :] = prediction / self.peaks[i]
                # Estimate the order
                order[self.ratios[i - 1, :] > threshold] = self.m_p[i - 1]

            # At the end, the remainder are the max_m
            order[order == 0] = self.max_m

            return order

    # %%
    def cupyEstimateBaud(x: cp.ndarray, fs: float):
        Xf = cp.fft.fftshift(cp.fft.fft(cp.abs(x)))
        Xfabs = cp.abs(Xf)
        freq = cp.fft.fftshift(cp.asarray(makeFreq(x.size, fs)))
        # Find the peaks
        peaks, _ = sps.find_peaks(Xfabs.get())
        prominences = sps.peak_prominences(Xfabs.get(), peaks)[0]
        # Sort prominences
        si = np.argsort(prominences)
        peaks = peaks[si]
        b1 = freq.get()[peaks[-2]]  # 2nd highest, 1st highest is the centre
        b2 = freq.get()[peaks[-3]]  # 3rd highest

        # Average the 2
        estBaud = (np.abs(b1) + np.abs(b2)) / 2

        return estBaud, peaks[-2], peaks[-3], Xf, freq

except Exception as e:
    print("Skipping cupy-related cyclostationaryRoutines.")


# %%
if __name__ == "__main__":
    from signalCreationRoutines import randPSKsyms

    syms, bits = randPSKsyms(1000, 4)
    x = sps.resample_poly(syms, 2, 1)
    fs = 1000
    x = x * np.exp(1j * 2 * np.pi * 5 * np.arange(x.size) / fs)

    offset = estimateOffsetViaCM(x, fs, 4)
    print("Estimated offset = %f" % (offset))
