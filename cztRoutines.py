# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:26:26 2021

@author: Lken
"""

import numpy as np
import sympy

#%%
def czt(x, f1, f2, binWidth, fs):
    '''
    n = (f2-f1)/binWidth + 1
    w = - i 2 pi (f2-f1+binWidth)/(n fs)
    a = i 2 pi (f1/fs)
    
    cztoptprep(len(x), n, w, a, nfft) # nfft custom to >len(x)+n-1
    
    there seems to be a misalignment issue. 1 element off when compared with dft output
    '''
    
    k = int((f2-f1)/binWidth + 1)
    m = len(x)
    nfft = m + k
    foundGoodPrimes = False
    while not foundGoodPrimes:
        nfft = nfft + 1
        if np.max(sympy.primefactors(nfft)) <= 7: # change depending on highest tolerable radix
            foundGoodPrimes = True
    
    kk = np.arange(-m+1,np.max([k-1,m-1])+1)
    kk2 = kk**2.0 / 2.0
    ww = np.exp(-1j * 2 * np.pi * (f2-f1+binWidth)/(k*fs) * kk2)
    chirpfilter = 1 / ww[:k-1+m]
    fv = np.fft.fft( chirpfilter, nfft )
    
    nn = np.arange(m)
    # print(ww[m+nn-1].shape)
    aa = np.exp(1j * 2 * np.pi * f1/fs * -nn) * ww[m+nn-1]
    
    y = x * aa
    fy = np.fft.fft(y, nfft)
    fy = fy * fv
    g = np.fft.ifft(fy)
    
    g = g[m:m+k] * ww[m:m+k]
    
    return g


def dft(x, freqs):
    '''

    Parameters
    ----------
    x : array
        Input data.
    freqs : array
        Array of bin frequency values to evaluate at.

    Returns
    -------
    Array of DFT bin values for input frequencies.

    '''
    tones = np.exp(-1j*2*np.pi*freqs.reshape((-1,1))*np.arange(len(x))/fs)
    output = tones @ x
    
    return output