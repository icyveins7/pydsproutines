# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 17:03:53 2020

@author: Seo
"""

import numpy as np
import scipy as sp

def fastXcorr(cutout, rx, shifts=None, normalize=True):
    """ No frequency scanning xcorr."""
    if shifts is None:
        shifts = np.arange(len(rx)-len(cutout)+1)
    
    result = np.zeros(len(shifts),dtype=np.float64)
    cutoutNormSq = np.linalg.norm(cutout)**2.0
    for i in range(len(shifts)):
        s = shifts[i]
        result[i] = sp.absolute(np.vdot(rx[s:s+len(cutout)], cutout))**2.0
        if normalize:
            rxNormPartSq = np.linalg.norm(rx[s:s+len(cutout)])**2.0
            result[i] = result[i]/cutoutNormSq/rxNormPartSq
            
    return result

def convertQF2toSNR(qf2):
    """For xcorr against pure signal."""
    return qf2/(1.0-qf2)

def convertQF2toEffSNR(qf2):
    """For xcorr of two noisy signals."""
    return 2.0*qf2/(1.0-qf2)

def expectedEffSNR(snr1, snr2=None):
    """For calculating expected SNR of two noisy signals."""
    if snr2 is None:
        snr2 = snr1
    
    y = 1.0/(0.5 * (1/snr1 + 1/snr2 + 1/snr1/snr2))
    return y