# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:57:46 2020

@author: Seo
"""


import numpy as np
import scipy as sp
import scipy.signal as sps
from signalCreationRoutines import *
from xcorrRoutines import *
import pyqtgraph as pg

# create a template signal
chnBW = 1000
numSyms = 2000
m = 4
syms0, bits = randPSKsyms(numSyms, m, dtype=np.complex128)

# resample it
up = 2
signal = sps.resample_poly(syms0,up,1)
fs = up * chnBW
fig = pg.plot(title='noiseless spectrum plot')
fig.plot(makeFreq(len(signal),fs), 20*np.log10(np.abs(np.fft.fft(signal))))

# add it 3 times completely separated to check
snr_inband_linearList = [10.0, 20.0, 5.0]
noiseLen = (len(snr_inband_linearList) + 1) * len(signal)
# sigStartIdxList = np.arange(100, noiseLen, len(signal) + 100)[:len(snr_inband_linearList)] # fully separated
sigStartIdxList = np.array([100,110,120])

signalList = np.vstack((signal,signal,signal))
fshifts = [100.0, 200.0, 300.0]

noise, rxfull, tones = addManySigToNoise(noiseLen, sigStartIdxList, signalList, chnBW, fs, snr_inband_linearList, fshifts = fshifts)

# # check time plot
# fig1 = pg.plot(title='time-power plot')
# fig1.plot(np.abs(rxfull)**2.0)

# # check spectrum plot
# fig2 = pg.plot(title='spectrum plot')
# fig2.plot(makeFreq(len(rxfull),fs), 20*np.log10(np.abs(np.fft.fft(rxfull))))
    
   
# just xcorr using the functions..
result, freqlist = fastXcorr(signal, rxfull, freqsearch=True, outputCAF=False)
fig3 = pg.plot(title='xcorr func td flattened')
fig3.plot(result)



# # check the known xcorr values, using the tones generated
# sigNormSq = np.linalg.norm(signal)**2.0
# for i in range(tones.shape[0]):
#     tone = tones[i]
#     sigStartIdx = sigStartIdxList[i]
    
#     # shift the rx back
#     rxShift = rxfull * tone.conj()
#     rxShiftSlice = rxShift[sigStartIdx:sigStartIdx + len(signal)]
#     pdt = np.vdot(signal, rxShiftSlice)
#     qf2 = np.abs(pdt) ** 2.0 / sigNormSq / np.linalg.norm(rxShiftSlice)**2.0
    
#     print('qf2 = %f, snr_linear = %f' % (qf2, convertQF2toSNR(qf2)))
#     print('note that these values should be half the snr input, since the snr is calculated IN-BAND')