# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:58:31 2020

@author: Seo
"""


import numpy as np
import scipy as sp
import cupy as cp

def cp_lfilter(ftap, x):
    '''
    Note: try to make sure the inputs are of the same type to maintain consistency.
    Use ftap.astype(np.complex128) and x.astype(np.complex128) for example.
    '''
    
    padlen = len(ftap) + len(x) - 1
    
    h_ftap_pad = np.pad(ftap,(0,padlen-len(ftap)))
    h_x_pad = np.pad(x,(0,padlen-len(x)))
    
    d_ftap_pad = cp.asarray(h_ftap_pad)
    d_x_pad = cp.asarray(h_x_pad)
    
    d_ftap_pad = cp.fft.fft(d_ftap_pad)
    d_x_pad = cp.fft.fft(d_x_pad)
    d_prod = d_ftap_pad * d_x_pad
    d_result = cp.fft.ifft(d_prod)
    
    h_result = cp.asnumpy(d_result)
    
    h_result_clip = h_result[:len(x)]
    
    return h_result_clip

def wola(f_tap, x, Dec, N=None, dtype=np.complex64):
    '''
    Parameters
    ----------
    f_tap : array
        Filter taps. Length must be integer multiple of N.
    x : array
        Input.
    Dec : scalar
        Downsample rate per channel.
    N : scalar
        Number of channels. Defaults to Dec (corresponds to no overlapping channels).

    Returns
    -------
    Channelised output.
    '''
    
    if N == None:
        N = Dec
        print('Defaulting to ' + str(N))
    elif N/Dec != 2:
        raise Exception("Only supporting up to N/Dec = 2.")
    
    if len(f_tap) % N != 0:
        raise Exception("Length must be integer multiple of N.")
    
    
    print('N = %i, Dec = %i' % (N, Dec))
    
    L = len(f_tap)
    nprimePts = int(np.floor(len(x) / Dec))
    
    out = np.zeros((nprimePts, N), dtype=dtype)
    
    for nprime in range(nprimePts):
        n = nprime*Dec
        
        dft_in = np.zeros(N, dtype=dtype)
        
        for a in range(N):
            for b in range(int(L/N)):             
                if (n - (b*N+a) >= 0):
                    dft_in[a] = dft_in[a] + x[n - (b*N+a)] * f_tap[b*N+a]
                    
        out[nprime] = np.fft.ifft(dft_in) * N # python's version auto scales it by 1/N, which we don't want
        
        if (Dec*2 == N) and (nprime%2 != 0):
            idx2flip = np.arange(1, N, 2)
            out[nprime][idx2flip] = -out[nprime][idx2flip]
            
    return out
                
    
        