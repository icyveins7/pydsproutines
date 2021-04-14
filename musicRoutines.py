# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:08:45 2021

@author: Seo
"""

import numpy as np
import matplotlib.pyplot as plt
from signalCreationRoutines import *

def musicAlg(x, freqlist, rows, p):
    '''
    p is the dimensionality of the signal subspace.
    x is expected as a 1-dim array (flattened).
    '''
    x = x.reshape((-1,1)) # vectorize
    cols = int(np.floor(len(x)/rows))
    xslen = rows * cols
    xs = x[:xslen] # we cut off the ending bits
    xs = xs.reshape((cols, rows)).T
    
    Rx = (1/cols) * xs @ xs.conj().T
    
    u, s, vh = np.linalg.svd(Rx)
    
    f = np.zeros(len(freqlist))
    for i in range(len(freqlist)):
        freq = freqlist[i]
        
        e = np.exp(1j*2*np.pi*freq*np.arange(rows)).reshape((1,-1)) # row, 2-d vector directly
        eh = e.conj()
        
        d = eh @ u[:,p:]
        denom = np.sum(np.abs(d)**2)
        
        f[i] = 1/denom
   
    
    return f, u, s, vh


if __name__ == '__main__':
    fs = 1e4
    length = 1*fs
    x = np.exp(1j*2*np.pi*1000*np.arange(length)/fs) + np.exp(1j*2*np.pi*1001*np.arange(length)/fs)
    # x = x + (np.random.randn(x.size) + np.random.randn(x.size)*1j) * 0.5
    xfft = np.fft.fft(x)
    
    freqlist = np.arange(999,1003,0.1)
    f,u,s,vh = musicAlg(x, freqlist/fs, 5, 2) # normalize freq
    
    plt.figure()
    plt.plot(makeFreq(len(x),fs), np.abs(xfft)/np.max(np.abs(xfft)))
    plt.plot(freqlist, f/np.max(f))
    plt.xlim([990,1010])