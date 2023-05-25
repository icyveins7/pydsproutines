#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:11:04 2023

@author: seoxubuntu

Test the threshold at which FFT estimate of CMx0 blind frequency offset estimation
fails vs a simpler phase increment method.

TODO: complete
"""

from plotRoutines import *
from signalCreationRoutines import *

import sys
import numpy as np
import scipy.signal as sps

if __name__ == "__main__":
    closeAllFigs()
    # Command-line options
    print(sys.argv)
    
    # Display the parameters
    osr = 2
    numSyms = 1000
    m = 4
    snr_linear = 20000
    fshiftHalfWidth = 0.3
    
    #%% Run the test
    # Create signal and oversample with noise
    syms, bits = randPSKsyms(numSyms, m)
    rssyms = sps.resample_poly(syms, osr, 1)
    
    fshift = np.random.rand() * fshiftHalfWidth
    noise, rx, tone = addSigToNoise(rssyms.size, 0, rssyms, 1, osr, snr_inband_linear=snr_linear,
                              fshift = fshift)
    
    # Plot signal
    win, ax = plotSpectra([rx], [1], windowTitle="FFT")
    twin, tax = plotAmpTime([rx], [1], windowTitle="Time-amplitude")
    