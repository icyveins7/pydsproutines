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
