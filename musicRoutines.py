# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:08:45 2021

@author: Seo
"""

import numpy as np

def musicAlg(x, freqlist, rows, p):
    '''
    p is the dimensionality of the signal subspace.
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
        
        e = np.exp(1j*2*np.pi*freq*np.arange(rows)) # row vector directly
        eh = e.conj()
        
        d = eh @ u[:,p:]
        denom = np.sum(np.abs(d)**2)
        
        f[i] = 1/denom
    
    
    return f, u, s, vh