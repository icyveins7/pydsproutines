# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:58:31 2020

@author: Seo
"""


import numpy as np
import scipy as sp
import scipy.signal as sps
import cupy as cp
import cupyx.scipy.signal as cpsps
import cpuWola as cpw

def cp_lfilter(ftap: cp.ndarray, x: cp.ndarray):
    '''
    Note: convert inputs into GPU arrays before passing them in.
    '''
    
    c = cpsps.convolve(ftap, x)[:x.size]
    
    return c

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

#%%
class Channeliser:
    """
    Wrapper for WOLA output, with internal memory to account for filter delay;
    similar to lfilter's 'zi' argument which specifies delay.
    
    Instead, we pad the input vectors at the front and copy the ending
    samples after every filter invocation.
    
    Internally uses the cpuWola dll.
    """
    def __init__(self, numTaps, numChannels, Dec, NUM_THREADS=4):
        self.f_tap = sps.firwin(numTaps, 1.0/Dec).astype(np.float32)
        self.numChannels = int(numChannels)
        self.Dec = int(Dec)
        self.NUM_THREADS = int(NUM_THREADS)
        
        self.reset()
        self.jump = int(self.f_tap.size / self.Dec)
        
    def reset(self):
        self.delay = np.zeros(self.f_tap.size, dtype=np.complex64)
        
    def channelise(self, x):
        y = np.hstack((self.delay, x))
        channels, _ = cpw.cpu_threaded_wola(y, self.f_tap, self.numChannels, self.Dec, NUM_THREADS=self.NUM_THREADS)
        self.delay[:] = x[-self.delay.size:] # copy the ending samples into delay
        
        return channels[self.jump:,:] # only return the valid parts ie skip the delay/Dec samples
        
    

#%%
def energyDetection(ampSq, medfiltlen, snrReqLinear=4.0, noiseIndices=None, splitSignalIndices=True):
    '''
    Parameters
    ----------
    ampSq : array
        Array of energy samples (amplitude squared values).
    medfiltlen : scalar
        Length of median filter (must be odd)
    snrReqLinear : scalar, optional
        SNR requirement for detection. The default is 4.0.
    noiseIndices : array, optional
        Specified indices to use to estimate noise power. The default is np.arange(100000).
    splitSignalIndices : bool, optional
        Boolean to specify whether to return signal indices as a list of arrays, split at every discontinuous index (difference more than 1 sample).

    Returns
    -------
    noiseIndices : array
        Array of indices used to calculate the meanNoise power.
    meanNoise : scalar
        Mean noise power.
    reqPower : scalar
        Mean noise power * the input snr requirement.
    medfiltered : array
        The median filtered output.
    signalIndices : array
        Array of indices which are greater than the reqPower.
    '''
    if noiseIndices is None:
        noiseIndices = np.arange(100000)
        print("Noise indices defaulting to [%d, %d]" % (noiseIndices[0],noiseIndices[-1]))
    
    # Medfilt in gpu as it's usually 1000x faster (not exaggeration)
    d_ampSq = cp.array(ampSq) # move to gpu
    d_medfiltered = cpsps.medfilt(d_ampSq, medfiltlen)
    medfiltered = cp.asnumpy(d_medfiltered) # move back
    
    # Detect the energy requirements
    sampleNoise = medfiltered[noiseIndices]
    meanNoise = np.mean(sampleNoise)
    reqPower = meanNoise * snrReqLinear
    signalIndices = np.argwhere(medfiltered > reqPower)
    if splitSignalIndices:
        splitIndices = np.argwhere(np.diff(signalIndices)>1).flatten() + 1 # the + 1 is necessary
        signalIndices = np.split(signalIndices, splitIndices)
    
    return noiseIndices, meanNoise, reqPower, medfiltered, signalIndices
    