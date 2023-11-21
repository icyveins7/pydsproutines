# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:26:20 2020

@author: Seo
"""

import scipy.signal as sps
import scipy as sp
from scipy.fftpack import fftshift, fft, ifft
import numpy as np
import os
import matplotlib.pyplot as plt
from signalCreationRoutines import makeFreq
import time
plt.close("all")

# generate a single long filter
chnBW = 1e2
fs = 1e6
up = 10000
print("Fraction is " + str(chnBW/fs))

longftap = sps.firwin(250000, chnBW/fs)
w, h = sps.freqz(longftap, worN = len(longftap))
plt.figure(1)
plt.plot(w / np.pi, 20 * np.log10(abs(h)), 'b')
plt.title('Direct vs cascaded filter comparison')

# try make shorter taps
shortup = 10
shortiter = 4
sftaplist = [sps.firwin(2500, 1/shortup),
             sps.firwin(1250, 1/shortup),
             sps.firwin(750, 1/shortup),
             sps.firwin(320, 1/shortup)]
for i in range(len(sftaplist)):
    ws, hs = sps.freqz(sftaplist[i], worN = len(sftaplist[i]))
    plt.figure(2)
    plt.plot(ws / np.pi, 20 * np.log10(abs(hs)))
    plt.title('Short filter')

# create fake sig using flat spectrum, at chnBW, 1 second ( going to upsample from here )
sigF = chnBW * np.exp(1j*np.random.randn(int(chnBW))) # scale by some number to make it look nice later on (less clipping due to precision)
sig = sp.ifft(sigF)
plt.figure(3)
plt.plot(makeFreq(len(sig), fs), 20*np.log10(np.abs(fft(sig))))
plt.title('Spectrum of test signal')

# upsample the old way
t1 = time.time()
sigres = sps.resample_poly(sig, up, 1, window=longftap)
tt = time.time() - t1
print('Took %fs for old upsample.' % tt)

# upsample/interpolate through cascaded smaller filters?
# from https://www.dsprelated.com/showarticle/903.php
# the first filter is the longest, and the last filter is the shortest
step = sig

t2 = time.time()
for i in range(shortiter):
    step = sps.resample_poly(step,10,1,window=sftaplist[i])

sigres_c = step
tt2 = time.time() - t2
print('Took %fs for new upsample.' % tt2)

plt.figure(4)
plt.plot(makeFreq(len(sigres), fs), 20*np.log10(np.abs(fft(sigres))))
plt.title('Old resample method vs new method')
plt.plot(makeFreq(len(sigres_c), fs), 20*np.log10(np.abs(fft(sigres_c))))