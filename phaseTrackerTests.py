# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 21:10:55 2020

@author: Seo
"""

import numpy as np
import scipy as sp
import scipy.signal as sps
import matplotlib.pyplot as plt

plt.close('all')

#%% parameters
numBits = 1000
fs = 1000
f = 200
upsqrt = 2
up = upsqrt**2
down = 1


#%% initialization
bits = np.random.randint(0,4,numBits)
qpskSyms = np.exp(1j*bits*2*np.pi/4)
rx = sps.resample_poly(qpskSyms, up, down)

#%% checks
plt.figure('Spectrum')
plt.plot(20*np.log10(sp.fft.fftshift(np.abs(sp.fft.fft(rx)))))

plt.figure('Symbols, Original')
plt.plot(np.real(qpskSyms), np.imag(qpskSyms), 'r.')

plt.figure('Symbols, Received')
plt.plot(np.real(rx), np.imag(rx), 'r.')

plt.figure('Symbols, Absolute Rx')
plt.plot(np.abs(rx))


#%% plot eye openings over the resampled
plt.figure('Eye Openings')
for i in range(up):
    plt.subplot(upsqrt,upsqrt,i+1)
    plt.plot(np.real(rx[i::up]), np.imag(rx[i::up]), 'r.')

#%% attempt differential filter
diffTaps = np.array([-1,0,1])
diffFilt = sps.lfilter(diffTaps,1,np.abs(rx))[1:]

plt.figure('Differential Filter Output')
plt.plot(np.abs(diffFilt))
