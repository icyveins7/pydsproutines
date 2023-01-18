#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 23:14:33 2021

@author: seolubuntu
"""

import numpy as np
import scipy as sp
import scipy.signal as sps
import cupy as cp
from signalCreationRoutines import makeFreq

def estimateBaud(x: np.ndarray, fs: float):
    '''
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

    '''
    Xf = sp.fft.fftshift(sp.fft.fft(np.abs(x)))
    Xfabs = np.abs(Xf)
    freq = sp.fft.fftshift(makeFreq(x.size, fs))
    # Find the peaks
    peaks, _ = sps.find_peaks(Xfabs)
    prominences = sps.peak_prominences(Xfabs, peaks)[0]
    # Sort prominences
    si = np.argsort(prominences)
    peaks = peaks[si]
    b1 = freq[peaks[-2]] # 2nd highest, 1st highest is the centre
    b2 = freq[peaks[-3]] # 3rd highest

    # Average the 2
    estBaud = (np.abs(b1) + np.abs(b2)) / 2
    
    return estBaud, peaks[-2], peaks[-3], Xf, freq

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
    b1 = freq.get()[peaks[-2]] # 2nd highest, 1st highest is the centre
    b2 = freq.get()[peaks[-3]] # 3rd highest

    # Average the 2
    estBaud = (np.abs(b1) + np.abs(b2)) / 2
    
    return estBaud, peaks[-2], peaks[-3], Xf, freq
    