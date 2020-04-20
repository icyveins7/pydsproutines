# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:35:01 2020

@author: Seo
"""

import numpy as np
import scipy.signal as sps
import scipy as sp

def makeSRC4(t,T):
    X = 2 * t/T - 4.0
    g = np.sinc(X)/(1.0-X**2)
    
    # this might not happen if t was not generated exactly at the 0 (due to precision)
    infidx = np.argwhere(g==np.inf).flatten()
    
    print('Correcting inf indices..')
    g[infidx] = 0.5
        
    return g

def makeSRC4_clipped(t,T,k=1.0):
    '''SRC4 pulse shape clipped to only the middle 2 symbols.'''
    X = 2 * t/T - 2.0
    g = k * np.sinc(X)/(1.0-X**2)
    
    infidx = np.argwhere(g==np.inf).flatten()
    
    if (len(infidx)>0):
        g[infidx] = k * 0.5
        
    zeroidx = np.argwhere(t<0).flatten()
    zero2idx = np.argwhere(t>2*T).flatten()
    
    if (len(zeroidx) > 0):
        g[zeroidx] = 0
    if (len(zero2idx) > 0):
        g[zero2idx] = 0
    
    return g