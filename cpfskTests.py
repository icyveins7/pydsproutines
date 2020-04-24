# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:34:54 2020

@author: Seo
"""

from xcorrRoutines import *
from signalCreationRoutines import *
from filterCreationRoutines import *
import time
import pyqtgraph as pg
from scipy.fftpack import fftshift


# parameters
numBitsPerBurst = 48
baud = 16000
numBursts = 20
numBitsTotal = numBitsPerBurst * numBursts
m = 2 # m-ary
h = 1.0/m
up = 8

# create bits
bits = randBits(numBitsTotal, m)

# create cpfsk signal
# syms0, fs, data = makeCPFSKsyms(bits, baud, up = up) # syms0 will be the template

gflat = np.ones(up)/(2*up)
# syms0, fs, data = makePulsedCPFSKsyms(bits, baud, g = gflat, up = up) # new method of creation

# # check on pulsed vs non-pulsed
# pwp = pg.GraphicsWindow(title='check pulsed vs non-pulsed function')
# pwp_1 = pwp.addPlot(row=0, col=0)
# pwp_1.plot(np.real(syms0), pen = 'b')
# pwp_1.plot(np.real(syms0p), pen=None, symbolPen = 'r', symbol='x')
# pwp_2 = pwp.addPlot(row=1, col=0)
# pwp_2.plot(np.imag(syms0), pen = 'b')
# pwp_2.plot(np.imag(syms0p), pen=None, symbolPen = 'r', symbol='x')

# create SRC4 CPFSK symbols
gSRC4 = makeSRC4(np.arange(4 * up)/up,1)
gSRC4 = makeScaledSRC4(up, 1.0)/up
pwg = pg.plot(title='SRC4')
pwg.plot(np.arange(4*up)/up, gSRC4, symbol='x', symbolPen='r')
syms0, fs, data = makePulsedCPFSKsyms(bits, baud, g = gSRC4, up = up) # new method of creation


# shift signal?
fshift_idx = 123
fshift = fshift_idx * fs / len(syms0) # in order to clip to an index of the fft search, for now
print('fshift index = %i, corresponding to frequency %fHz' % (fshift_idx, fshift))

# check spectrum
freq = makeFreq(len(syms0), fs)
fftsyms = np.fft.fft(syms0)

# plot spectrum
pw = pg.GraphicsWindow(title='cpfsk spectrum (fft vs cm20)')
pw_1 = pw.addPlot(0,0)
pw_1.plot(freq, 20*np.log10(np.abs(fftsyms)), name='fft')

# test rabbit ears
cm20 = syms0**2.0
fftcm20 = np.fft.fft(cm20)
pw_2 = pw.addPlot(1,0)
pw_2.plot(freq, 20*np.log10(np.abs(fftcm20)), pen='r', name='cm20')

# add noise
startIdx = len(syms0)
noise, rx, tone = addSigToNoise(3 * len(syms0), startIdx, syms0, baud, fs, 1000, sigPwr = 1.0, fshift = fshift)

# plot new spectrum and rabbit ears?
freqrx = makeFreq(len(rx), fs)
fftrx = np.fft.fft(rx)

pw2 = pg.GraphicsWindow(title='cpfsk with noise (fft vs cm20)')
pw2_1 = pw2.addPlot(0,0)
pw2_1.plot(freqrx, 20*np.log10(np.abs(fftrx)), name='fft')

rxcm20 = rx**2.0
fftrxcm20 = np.fft.fft(rxcm20)
pw2_2 = pw2.addPlot(1,0)
pw2_2.plot(freqrx, 20*np.log10(np.abs(fftrxcm20)), pen='r', name='cm20')

# perform xcorr with original
shift_lim = 80
shifts=np.arange(startIdx - shift_lim, startIdx + shift_lim + 1)
freqcutout = makeFreq(len(syms0), fs)

## xcorr with no freq scan
#xc = fastXcorr(syms, rx, shifts = shifts)
#xc_mi = np.argmax(xc)
#print('Xcorr max at %i, val = %f' % (xc_mi, xc[xc_mi]))
#pw3 = pg.plot(title='fast xcorr (no freq shifts)')
#pw3.plot(shifts, xc)

# # xcorr with freq scan
# xc, freqlist = fastXcorr(syms0, rx, freqsearch=True, shifts=shifts)
# xc_mi = np.argmax(xc)
# print('Xcorr max at %i, val = %f, freqidx = %i' % (xc_mi, xc[xc_mi], freqlist[xc_mi]))
# pw3 = pg.plot(title='fast xcorr (no freq shifts)')
# pw3.plot((shifts-startIdx)/fs, xc)

# # xcorr with CAF output
# xc = fastXcorr(syms0, rx, freqsearch=True, outputCAF=True, shifts=shifts)
# #xc_mi = np.argmax(xc)
# #print('Xcorr max at %i, val = %f, freqidx = %i' % (xc_mi, xc[xc_mi], freqlist[xc_mi]))
# pw3 = pg.image(title='xcorr CAF plot')
# pw3.setImage(fftshift(xc,1))
# pw3.setPredefinedGradient('thermal')
# # interestingly, there are extra peaks AWAY from the real timeshift
# pw4 = pg.plot(freqcutout,xc[shift_lim], title='freq search from CAF, at correct time value')
# pw5 = pg.GraphicsWindow(title='freqsearch from CAF, at +/-4 from correct time value')
# pw5_1 = pw5.addPlot(row=0, col=0)
# pw5_1.plot(freqcutout,xc[shift_lim - 4])
# pw5_2 = pw5.addPlot(row=1, col=0)
# pw5_2.plot(freqcutout,xc[shift_lim + 4])

# # look at the TD slope for the correct freqshift
# fdsolvedidx = np.argmax(xc[shift_lim])
# fdsolved = freqcutout[fdsolvedidx]
# xc_tdflatten = xc[:,fdsolvedidx]
# pw6 = pg.plot(title='tdoa flattened')
# pw6.plot((shifts-startIdx)/fs, xc_tdflatten)


## add a second signal
pw7p = []
si = np.arange(12,18)
pw7 = pg.GraphicsWindow(title = 'tdoa flattened, two peaks')

for i in range(len(si)):
    
    noise2, rx2, _ = addSigToNoise(3 * len(syms0), startIdx + si[i], syms0, baud, fs, 100000, sigPwr = 1.0, fshift = fshift)
    rx_c = rx2 + rx
    
    # # correct it with reversed freq tone
    # rx_c = rx_c * tone.conj()
    
    # xcorr again with new CAF
    xc_c = fastXcorr(syms0, rx_c, freqsearch=True, outputCAF=True, shifts=shifts)
    fdsolvedidx = np.argmax(xc_c[shift_lim])
    fdsolved = freqcutout[fdsolvedidx]
    print('fdsolved = ' + str(fdsolved))
    xc_c_tdflatten = xc_c[:,fdsolvedidx]
    
    # # xcorr again without CAF, direct to td only
    # xc_c_tdflatten, xc_c_flist = fastXcorr(syms0, rx_c, freqsearch=True, outputCAF=False, shifts=shifts)
    
    # add to plot
    pw7p.append(pw7.addPlot(row=int(i)/4, col=int(i)%4))
    pw7p[i].plot((shifts-startIdx)/fs, xc_c_tdflatten)

## add second and third signals
pw8p = []
pw8 = pg.GraphicsWindow(title = 'tdoa flattened, 3 peaks')
ssi = np.vstack((np.arange(12,18), np.arange(12,18)))
for i in range(ssi.shape[1]):
    for j in range(ssi.shape[1]):
        noise2, rx2, _ = addSigToNoise(3 * len(syms0), startIdx + ssi[0][i], syms0, baud, fs, 1000000, sigPwr=1.0, fshift = fshift)
        noise3, rx3, _ = addSigToNoise(3 * len(syms0), startIdx + ssi[0][i] + ssi[1][j], syms0, baud, fs, 1000000, sigPwr=1.0, fshift = fshift)
        rx_cc = rx3 + rx2 + rx

        # xcorr again with new CAF
        xc_cc = fastXcorr(syms0, rx_cc, freqsearch=True, outputCAF=True, shifts=shifts)
        fdsolvedidx = np.argmax(xc_cc[shift_lim])
        fdsolved = freqcutout[fdsolvedidx]
        xc_cc_tdflatten = xc_cc[:, fdsolvedidx]

        # add to plot
        pw8p.append(pw8.addPlot(row=int(i), col=int(j)))
        pw8p[-1].plot(shifts - startIdx, xc_cc_tdflatten)

