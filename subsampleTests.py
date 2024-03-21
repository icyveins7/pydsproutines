# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:44:59 2020

@author: Seo
"""


import numpy as np
import scipy as sp
import scipy.signal as sps
from signalCreationRoutines import *
import pyqtgraph as pg

# create test signal
fs = 100
T = 1 / fs
sig, bits = randPSKsyms(fs, 4)
noise, rx = addSigToNoise(fs + 10, 0, sig)
fig1 = pg.plot(title="padded")
fig1.plot(np.abs(rx), symbol="x", symbolPen="r")

# propagate by subsample
rx2, tone = propagateSignal(rx, T * 0.5, fs, freq=10)
fig2 = pg.plot(title="half sample propagated")
fig2.plot(np.abs(rx2[0]), symbol="x", symbolPen="r")

# propagate by subsample for 1 sample
rx3, tone = propagateSignal(rx, T * 1, fs, freq=48)
fig3 = pg.plot(title="full sample propagated")
fig3.plot(np.abs(rx3[0]), symbol="x", symbolPen="r")

# check the tone is doing what it says
wrongslice = rx3.squeeze()[1 : 1 + len(sig)]
wrongqf2 = (
    np.abs(np.vdot(wrongslice, sig)) ** 2
    / np.linalg.norm(wrongslice) ** 2
    / np.linalg.norm(sig) ** 2
)
print(wrongqf2)

rxbackshift = rx3 * tone.conj()
rxbackshift = rxbackshift.squeeze()
rxslice = rxbackshift[1 : 1 + len(sig)]
qf2 = (
    np.abs(np.vdot(rxslice, sig)) ** 2
    / np.linalg.norm(rxslice) ** 2
    / np.linalg.norm(sig) ** 2
)
print(qf2)
