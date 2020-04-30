# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:04:13 2020

@author: Seo
"""


# add the outside code routines
import sys
addedpaths = ["F:\\PycharmProjects\\pydsproutines"]
for path in addedpaths:
    if path not in sys.path:
        sys.path.append(path)

# imports
import numpy as np
import scipy as sp
import scipy.signal as sps
import pyqtgraph as pg
from signalCreationRoutines import *
from xcorrRoutines import *
from filterCreationRoutines import *
from pgplotRoutines import *
from PyQt5.QtWidgets import QApplication
import time
# end of imports

#%% ============== generate a sample signal to be received at each sensor timestamp
# parameters
numBitsPerBurst = 48
baud = 16000
numBursts = 20
numBitsTotal = numBitsPerBurst * numBursts
m = 2 # m-ary
h = 1.0/m
up = 16
print('Duration of burst = %fs' % (numBitsTotal/baud))

# create bits
bits = randBits(numBitsTotal, m)

# create cpfsk signal
gflat = np.ones(up)/(2*up)

# create SRC4 CPFSK symbols
gSRC4 = makeSRC4(np.arange(4 * up)/up,1)
gSRC4 = makeScaledSRC4(up, 1.0)/up
syms0, fs, data = makePulsedCPFSKsyms(bits, baud, g = gSRC4, up = up) # new method of creation
T = 1/fs
print('\nWorking at fs = %fHz, sample period T = %gs' % (fs, T))

rxLen = 3 * len(syms0)
#=========================================== end of template signal generation

#%% now add it to 2 separate rx versions
numTxCopies = 3
rx1_startIdx = np.random.randint(-60, 60, size=numTxCopies) + len(syms0)
rx2_startIdx = np.random.randint(-60, 60, size=numTxCopies) + len(syms0)
snr_inband_linear1 = np.random.rand(len(rx1_startIdx)) * 36 + 4
snr_inband_linear2 = np.random.rand(len(rx1_startIdx)) * 36 + 4 # also expected to be the same..
print('\nSNR_inband_dB = ')
print(10*np.log10(snr_inband_linear1))
print(10*np.log10(snr_inband_linear2))

fshifts1 = []
fshifts2 = [] # for now let's just not bother...
signalList = [syms0 for i in range(len(rx1_startIdx))]
noise1, rx1 = addManySigToNoise(rxLen, rx1_startIdx, signalList, baud, fs, snr_inband_linear1, fshifts = None)
noise2, rx2 = addManySigToNoise(rxLen, rx2_startIdx, signalList, baud, fs, snr_inband_linear2, fshifts = None)
print('\nActual start indices are:')
print(rx1_startIdx)
print(rx2_startIdx)
print('\nActual TD (samples):')
actual_td_samples = rx2_startIdx - rx1_startIdx
print(actual_td_samples)

# get theoretical peak possibilities
theo_peaks = theoreticalMultiPeak(rx1_startIdx, rx2_startIdx)
print('\nTheoretical peaks: ')
print(theo_peaks)

# slice one part of rx1
cutoutStartIdx = len(syms0)
rx1_cutout = rx1[len(syms0):len(syms0) + 4096]
cutoutFreq = makeFreq(len(rx1_cutout), fs)

# use it to xcorr
shifts = np.arange(-25*up,25*up+1) + cutoutStartIdx

qf2, flist = fastXcorr(rx1_cutout, rx2, freqsearch=True, outputCAF=False, shifts=shifts)
effSNR = convertQF2toEffSNR(qf2)

fig1 = pg.GraphicsWindow(title='effSNR')
fig1_1 = fig1.addPlot(0,0)
fig1_1.plot(shifts - cutoutStartIdx,effSNR)
pgPlotDeltaFuncs(fig1_1, theo_peaks, np.max(effSNR), 'r')
pgPlotDeltaFuncs(fig1_1, actual_td_samples, np.max(effSNR), 'b')
fig1_2 = fig1.addPlot(1,0)
fig1_2.plot(shifts - cutoutStartIdx, cutoutFreq[flist])
fig1_2.setXLink(fig1_1)

#%% create a gaussian window and attempt to overlay it
window = sps.gaussian(len(effSNR),std=1/(2*baud)/T)
fig1_1.plot(np.arange(theo_peaks[0] - 400, theo_peaks[0] + 401), window * effSNR[np.argwhere(shifts-cutoutStartIdx == theo_peaks[0]).flatten()], symbol='x')

# seems correct, how about we try deconvolving with this
G = np.fft.fft(window)
# G_fixed = G + 0.0001 # prevent divides by 0, doesn't seem to work
H = np.fft.fft(effSNR)
H_fixed = H
H_fixed[50:750] = 0
F = H / G
# F_fixed = H / G_fixed
F_fixed2 = H_fixed / G
deconvolved = np.fft.ifft(F)
# dec_fixed = np.fft.ifft(F_fixed) # doesn't seem to work?

dec_fixed2 = np.fft.ifft(F_fixed2)
fig2 = pg.GraphicsWindow(title='effSNR before and after deconvolution')
fig2_1 = fig2.addPlot(0,0)
fig2_1.plot(shifts - cutoutStartIdx,effSNR)
fig2_2 = fig2.addPlot(1,0)
fig2_2.plot(shifts - cutoutStartIdx,np.abs(np.fft.ifftshift(dec_fixed2))) 
pgPlotDeltaFuncs(fig2_2, theo_peaks, np.max(np.abs(dec_fixed2)), 'r')
pgPlotDeltaFuncs(fig2_2, actual_td_samples, np.max(np.abs(dec_fixed2)), 'b')

# # ???
# testdelta = np.zeros(window.shape)
# testdelta[0] = 1
# testc = np.convolve(window, testdelta) # standard way

# c_f = np.fft.fft(testdelta)
# c_h = c_f * G
# h = np.fft.ifft(c_h)
# pg.plot(testc)
# pg.plot(np.real(h))



