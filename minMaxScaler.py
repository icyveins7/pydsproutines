# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:15:49 2019

@author: Lken
"""

import numpy as np
import ctypes as ct

def cpu_threaded_multichannel_minMaxScaler_32fc(y):
    '''
    Input the numChans and chanLen as output in the multi-channel WOLA function.
    The chanLen is just the product of the inner channels and the time points.
    '''

    _libmc = np.ctypeslib.load_library('multiChannel_minMaxScaler_32fc','.')
    array_1d_complex_channels = np.ctypeslib.ndpointer(dtype=np.complex64, ndim = 3, flags = 'CONTIGUOUS') # this is ndim 2 since its multichannels
    array_1d_single = np.ctypeslib.ndpointer(dtype=np.float32, ndim = 1, flags = 'CONTIGUOUS')
    _libmc.multiChan_minMaxScaler_32fc.restype = ct.c_int32
    _libmc.multiChan_minMaxScaler_32fc.argtypes = [array_1d_complex_channels, ct.c_int32, ct.c_int32, array_1d_single]
    	
    numChans = y.shape[0]
    chanLen = y.shape[1] * y.shape[2]
    out = np.empty(int(numChans * chanLen), dtype = np.float32) # make the output
    	
    retcode = _libmc.multiChan_minMaxScaler_32fc(y, int(numChans), int(chanLen), out) # run the dll function
    	
    out = out.reshape((numChans,y.shape[1],y.shape[2])) # reshape to channels in columns
    	
    return out, retcode