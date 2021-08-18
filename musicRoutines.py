# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:08:45 2021

@author: Seo
"""

import numpy as np
import matplotlib.pyplot as plt
from signalCreationRoutines import *
from spectralRoutines import *

#%%
def musicAlg(x, freqlist, rows, p, snapshotJump=None):
    '''
    p is the dimensionality of the signal subspace.
    x is expected as a 1-dim array (flattened).
    snapshotJump is the index jump per column vector. By default this jump is equal to rows,
    i.e. each column vector is unique.
    '''
    
    if snapshotJump is None: # then we snapshot via slices
        # 1 2 3 4 5 6 ->
        # 1 3 5
        # 2 4 6  for example
        x = x.reshape((-1,1)) # vectorize
        cols = int(np.floor(len(x)/rows))
        xslen = rows * cols
        xs = x[:xslen] # we cut off the ending bits
        xs = xs.reshape((cols, rows)).T
    else: # we use the rate to populate our columns
        # e.g. for snapshot jump = 1,
        # 1 2 3 4 5 6 ->
        # 1 2 3 4 5
        # 2 3 4 5 6
        if snapshotJump <= 0:
            raise ValueError("snapshotJump must be at least 1.")
            
        x = x.flatten() # in case it's not flat
        cols = (x.size - rows) / snapshotJump # calculate thee required columns
        xs = np.zeros((rows, int(cols+1)), x.dtype)
        print("Matrix dim is %d, %d" % (xs.shape[0], xs.shape[1]))
        for i in range(xs.shape[1]): # fill the columns
            xs[:,i] = x[i * snapshotJump : i * snapshotJump + rows]
            
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

#%%
if __name__ == '__main__':
    #%%
    plt.close("all")
    fs = 1e4
    length = 1*fs
    fdiff = 0.5
    f0 = 1000
    padding = 0
    # x = np.exp(1j*2*np.pi*f0*np.arange(length)/fs) + np.exp(1j*2*np.pi*(f0+fdiff)*np.arange(length)/fs)
    x = np.pad(np.exp(1j*2*np.pi*f0*np.arange(length)/fs), (padding,0))
    
    # noisyAmp = (np.random.randn(int(length+100))*0.000001+1.0)
    # noisyAmp = np.abs(noisyAmp)
    noisyAmp = 1.0
    
    # How does it react to a non-constant amplitude tone?
    plt.figure("Amplitude for second tone")
    plt.plot(noisyAmp)
    
    x = x + noisyAmp * np.pad(np.exp(1j*2*np.pi*(f0+fdiff)*np.arange(length)/fs), (0,padding))
    x = x + (np.random.randn(x.size) + np.random.randn(x.size)*1j) * 1e-2
    # xfft = np.fft.fft(x)    
    
    fineFreqStep = 0.01
    fineFreqRange = 3 # peg to the freqoffset
    fineFreqVec = np.arange((f0+fdiff/2)-fineFreqRange,(f0+fdiff/2)+fineFreqRange + 0.1*fineFreqStep, fineFreqStep)
    xczt = czt(x, (f0+fdiff/2)-fineFreqRange,(f0+fdiff/2)+fineFreqRange, fineFreqStep, fs)
    
    freqlist = np.arange(999,1003,0.1)
    plt.figure("Comparison")
    # plt.plot(makeFreq(len(x),fs), np.abs(xfft)/np.max(np.abs(xfft)))
    plt.plot(fineFreqVec, np.abs(xczt)/ np.max(np.abs(xczt)), label='CZT')
    plt.vlines([f0,f0+fdiff],0,1,colors='r', linestyles='dashed',label='Actual')
    for i in range(1,5):
        # f,u,s,vh = musicAlg(x, freqlist/fs, 100, i) # normalize freq
        f,u,s,vh = musicAlg(x, freqlist/fs, 1000, i, snapshotJump=1) # use every possible slide, performs better!
        
        plt.figure("Comparison")
        plt.plot(freqlist, f/np.max(f), label='MUSIC, p='+str(i))
        plt.legend()
        plt.xlim([fineFreqVec[0],fineFreqVec[-1]])
        
    plt.figure("Eigenvalues")
    plt.plot(np.log10(s),'x-')
    
    ## At 1e-4 noise:
    # Note that at rows=100, unable to resolve. But at rows = 1000, able to resolve (dependency on 'block size' despite total length being equal)
    # p=3 is now required, instead of p=2
    ## At 1e-3 noise:
    # Similarly, at rows = 1000, able to resolve, but clarity is diminished.
    # Still p=3
    ## At 1e-2 noise:
    # No longer able to resolve with rows=1000.
    
    ## Now with snapshotJump = 1 (sliding window for every possible snapshot rather than unique snapshot)
    # 1e-2 noise:
    # Now possible to resolve with rows = 1000 !!!
