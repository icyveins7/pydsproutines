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

def cp_lfilter(ftap: cp.ndarray, x: cp.ndarray, chunksize: int=None):
    '''
    Note: convert inputs into GPU arrays before passing them in.
    '''
    if chunksize is None:
        c = cpsps.convolve(ftap, x)[:x.size]
        
        return c

    else: # May not be worth to do this..
        ptr = 0
        c = cp.zeros_like(x)
        while ptr < x.size:
            block = min(x.size - ptr, chunksize)
            if ptr == 0:
                # Perform full convolution in order to zero-pad
                c[ptr:ptr+block] = cpsps.convolve(ftap, x[:block], mode='full', method='direct')[:block]
            else:
                # Perform in valid range, but cut necessary previous samples for filter, no need to clip
                c[ptr:ptr+block] = cpsps.convolve(ftap, x[ptr-ftap.size+1:ptr+block], mode='valid', method='direct')

            ptr = ptr + block

        return c
    
    
# Raw kernel for tone creation
upFirdnKernel = cp.RawKernel(r'''
#include <cupy/complex.cuh>
extern "C" __global__
void upfirdn(
    const complex<float> *d_x, const int len,
    const float *d_taps, const int tapslen,
    const int up,
    const int down,
    complex<float> *d_out, int outlen)
{
    // allocate shared memory
    extern __shared__ double s[];
    
    float *s_taps = (float*)s; // (tapslen) floats
    /* Tally:  */

    // load shared memory
    for (int t = threadIdx.x; t < tapslen; t = t + blockDim.x){
        s_taps[t] = d_taps[t];
    }

    __syncthreads();
    
    // Define the indices to write to for this block
    int outStart = blockIdx.x * blockDim.x;
    int outEnd = min((blockIdx.x + 1) * blockDim.x, outlen); // The last block must stop at the length
    
    int i0, j;
    complex<float> z = 0; // Stack-variable for each thread
    
    // Loop over the output for this block
    for (int k = outStart + threadIdx.x; k < outEnd; k += blockDim.x) // technically this loop is pointless, since each thread performs one computation only
    {
        i0 = (down*k + tapslen/2) % up;
        
        for (int i = i0; i < tapslen; i += up){
            j = (down * k + tapslen/2 - i) / up;
            
            if (j < len && j >= 0)
            {
                z = z + s_taps[i] * d_x[j];
            }
        }
        
        // Write z to global memory
        d_out[k] = z;
    }
 
}
''', '''upfirdn''')

def cupyUpfirdn(x: cp.ndarray, taps: cp.ndarray, up: int, down: int):
    # if x.dtype != cp.complex64:
    #     raise TypeError("x is expected to be type complex64.")
    # if taps.dtype != cp.float32:
    #     raise TypeError("taps is expected to be type float32.")
        
    # Allocate output
    out = cp.zeros(x.size * up // down, dtype=cp.complex64)
        
    # Define just enough blocks to cover the output
    THREADS_PER_BLOCK = 256
    NUM_BLOCKS = out.size // THREADS_PER_BLOCK
    if NUM_BLOCKS * THREADS_PER_BLOCK < out.size:
        NUM_BLOCKS += 1
        
    # Define the shared memory requirement
    if taps.size * 4 > 48e3:
        raise MemoryError("Taps length too large for shared memory.")
    smReq = taps.size * 4
    
    # Call the kernel
    upFirdnKernel((NUM_BLOCKS,),(THREADS_PER_BLOCK,), 
                  (x, x.size,
                   taps, taps.size,
                   up, down,
                   out, out.size),
                  shared_mem=smReq)
    
    return out

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
    def __init__(self, numTaps, numChannels, Dec, NUM_THREADS=4, f_tap=None):
        if f_tap is None:
            self.f_tap = sps.firwin(numTaps, 1.0/Dec).astype(np.float32)
        else:
            self.f_tap = f_tap.astype(np.float32)
            
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
class BurstDetector:
    def __init__(self, medfiltlen: int, snrReqLinear: float):
        self.medfiltlen = medfiltlen
        self.snrReqLinear = snrReqLinear
        
        # Placeholders for later results
        self.d_absx = None
        self.d_ampSq = None
        self.d_medfiltered = None
        
    def medfilt(self, x):
        d_x = cp.asarray(x) # Push to gpu if not currently in it
        self.d_absx = cp.abs(d_x)
        self.d_ampSq = self.d_absx * self.d_absx
        self.d_medfiltered = cpsps.medfilt(self.d_ampSq, self.medfiltlen)
        
    def detectViaThreshold(self, threshold: float):
        signalIndices = cp.argwhere(self.d_medfiltered > threshold).flatten()
        splitIndices = cp.argwhere(cp.diff(signalIndices)>1).flatten() + 1 # the + 1 is necessary
        signalIndices = cp.split(signalIndices, splitIndices.get()) # For cupy, need to pull the split indices to host
        
        return signalIndices

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
    d_ampSq = cp.asarray(ampSq) # move to gpu
    d_medfiltered = cpsps.medfilt(d_ampSq, medfiltlen)
    medfiltered = cp.asnumpy(d_medfiltered) # move back
    
    # Detect the energy requirements
    sampleNoise = medfiltered[noiseIndices]
    meanNoise = np.mean(sampleNoise)
    reqPower = meanNoise * snrReqLinear
    signalIndices = np.argwhere(medfiltered > reqPower).flatten() # Patched to include .flatten, how was it working before?
    if splitSignalIndices:
        splitIndices = np.argwhere(np.diff(signalIndices)>1).flatten() + 1 # the + 1 is necessary
        signalIndices = np.split(signalIndices, splitIndices)
    
    return noiseIndices, meanNoise, reqPower, medfiltered, signalIndices
    