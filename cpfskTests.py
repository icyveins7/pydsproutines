# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:34:54 2020

@author: Seo
"""

from xcorrRoutines import *
from signalCreationRoutines import *
import time
import pyqtgraph as pg
from scipy.fftpack import fftshift

# parameters
numBitsPerBurst = 64
baud = 16000
numBursts = 10
numBitsTotal = numBitsPerBurst * numBursts
m = 2 # m-ary
h = 1.0/m

# create bits
bits = randBits(numBitsTotal, m)

# create cpfsk signal
syms0, fs, theta, data = makeCPFSKsyms(bits, baud, up = 4) # syms0 will be the template

# shift signal?
fshift_idx = 123
fshift = fshift_idx * fs / len(syms0) # in order to clip to an index of the fft search, for now
print('fshift index = %i, corresponding to frequency %fHz' % (fshift_idx, fshift))
tone = np.exp(1j*2*np.pi*fshift * np.arange(len(syms0)) / fs)
syms = syms0 * tone # after shifting

# check spectrum
freq = makeFreq(len(syms), fs)
fftsyms = np.fft.fft(syms)

# plot spectrum
pw = pg.plot(title='cpfsk spectrum')
pw.addLegend()
pw.plot(freq, 20*np.log10(np.abs(fftsyms)), name='fft')

# test rabbit ears
cm20 = syms**2.0
fftcm20 = np.fft.fft(cm20)
pw.plot(freq, 20*np.log10(np.abs(fftcm20)), pen='r', name='cm20')

# add noise
startIdx = len(syms)
noise, rx = addSigToNoise(3 * len(syms), startIdx, syms, baud, fs, 10, sigPwr = 1.0)

# plot new spectrum and rabbit ears?
freqrx = makeFreq(len(rx), fs)
fftrx = np.fft.fft(rx)

pw2 = pg.plot(title='cpfsk with noise')
pw2.addLegend()
pw2.plot(freqrx, 20*np.log10(np.abs(fftrx)), name='fft')

rxcm20 = rx**2.0
fftrxcm20 = np.fft.fft(rxcm20)
pw2.plot(freqrx, 20*np.log10(np.abs(fftrxcm20)), pen='r', name='cm20')

# perform xcorr with original
shift_lim = 20
shifts=np.arange(startIdx - shift_lim, startIdx + shift_lim + 1)

## xcorr with no freq scan
#xc = fastXcorr(syms, rx, shifts = shifts)
#xc_mi = np.argmax(xc)
#print('Xcorr max at %i, val = %f' % (xc_mi, xc[xc_mi]))
#pw3 = pg.plot(title='fast xcorr (no freq shifts)')
#pw3.plot(shifts, xc)

## xcorr with freq scan
#xc, freqlist = fastXcorr(syms0, rx, freqsearch=True, shifts=shifts)
#xc_mi = np.argmax(xc)
#print('Xcorr max at %i, val = %f, freqidx = %i' % (xc_mi, xc[xc_mi], freqlist[xc_mi]))
#pw3 = pg.plot(title='fast xcorr (no freq shifts)')
#pw3.plot(shifts, xc)

# xcorr with CAF output
xc = fastXcorr(syms0, rx, freqsearch=True, outputCAF=True, shifts=shifts)
#xc_mi = np.argmax(xc)
#print('Xcorr max at %i, val = %f, freqidx = %i' % (xc_mi, xc[xc_mi], freqlist[xc_mi]))
pw3 = pg.image(title='xcorr CAF plot')
pw3.setImage(fftshift(xc,1))
pw3.setPredefinedGradient('thermal')
# interestingly, there are extra peaks AWAY from the real timeshift
pw4 = pg.plot(makeFreq(len(syms0), fs),xc[shift_lim], title='freq search from CAF, at correct time value')
pw5 = pg.GraphicsWindow(title='freqsearch from CAF, at +/-4 from correct time value')
pw5_1 = pw5.addPlot(row=0, col=0)
pw5_1.plot(makeFreq(len(syms0), fs),xc[shift_lim - 4])
pw5_2 = pw5.addPlot(row=1, col=0)
pw5_2.plot(makeFreq(len(syms0), fs),xc[shift_lim + 4])

