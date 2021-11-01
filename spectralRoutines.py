# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:26:26 2021

@author: Lken
"""

import numpy as np
import sympy
from numba import jit
import cupy as cp
import time

#%%
def czt(x, f1, f2, binWidth, fs):
    '''
    n = (f2-f1)/binWidth + 1
    w = - i 2 pi (f2-f1+binWidth)/(n fs)
    a = i 2 pi (f1/fs)
    
    cztoptprep(len(x), n, w, a, nfft) # nfft custom to >len(x)+n-1
    '''
    
    k = int((f2-f1)/binWidth + 1)
    m = len(x)
    nfft = m + k
    foundGoodPrimes = False
    while not foundGoodPrimes:
        nfft = nfft + 1
        if np.max(sympy.primefactors(nfft)) <= 7: # change depending on highest tolerable radix
            foundGoodPrimes = True
    
    kk = np.arange(-m+1,np.max([k-1,m-1])+1)
    kk2 = kk**2.0 / 2.0
    ww = np.exp(-1j * 2 * np.pi * (f2-f1+binWidth)/(k*fs) * kk2)
    chirpfilter = 1 / ww[:k-1+m]
    fv = np.fft.fft( chirpfilter, nfft )
    
    nn = np.arange(m)
    # print(ww[m+nn-1].shape)
    aa = np.exp(1j * 2 * np.pi * f1/fs * -nn) * ww[m+nn-1]
    
    y = x * aa
    fy = np.fft.fft(y, nfft)
    fy = fy * fv
    g = np.fft.ifft(fy)
    
    g = g[m-1:m+k-1] * ww[m-1:m+k-1]
    
    return g

# Simple class to keep the FFT of the chirp filter to alleviate computations
class CZTCached:
    def __init__(self, xlength, f1, f2, binWidth, fs):
        self.k = int((f2-f1)/binWidth + 1)
        self.m = xlength
        self.nfft = self.m + self.k
        foundGoodPrimes = False
        while not foundGoodPrimes:
            self.nfft = self.nfft + 1
            if np.max(sympy.primefactors(self.nfft)) <= 7: # change depending on highest tolerable radix
                foundGoodPrimes = True
        
        kk = np.arange(-self.m+1,np.max([self.k-1,self.m-1])+1)
        kk2 = kk**2.0 / 2.0
        self.ww = np.exp(-1j * 2 * np.pi * (f2-f1+binWidth)/(self.k*fs) * kk2)
        chirpfilter = 1 / self.ww[:self.k-1+self.m]
        self.fv = np.fft.fft( chirpfilter, self.nfft )
        
        nn = np.arange(self.m)
        self.aa = np.exp(1j * 2 * np.pi * f1/fs * -nn) * self.ww[self.m+nn-1]
        
    def run(self, x):
        y = x * self.aa
        fy = np.fft.fft(y, self.nfft)
        fy = fy * self.fv
        g = np.fft.ifft(fy)
        
        g = g[self.m-1:self.m+self.k-1] * self.ww[self.m-1:self.m+self.k-1]
        
        return g


#%%
def dft(x, freqs, fs):
    '''

    Parameters
    ----------
    x : array
        Input data.
    freqs : array
        Array of bin frequency values to evaluate at.

    Returns
    -------
    Array of DFT bin values for input frequencies.

    '''
    
    output = np.zeros(len(freqs),dtype=np.complex128)
    for i in np.arange(len(freqs)):
        freq = freqs[i]
        tone = np.exp(-1j*2*np.pi*freq*np.arange(len(x))/fs)
        output[i] = np.dot(tone, x)
    
    return output

@jit(nopython=True)
def toneSpectrum(f0, freqs, fs, N, phi=0, A=1.0):
    '''
    Returns a spectrum corresponding to applying DFT to a tone with frequency f0 and phase phi,
    at values specified by the 'freqs' array.
    
    See the tone reproduction notebook for details.
    '''
    
    vals = -1j * A * (1 - np.exp(-1j*2*np.pi*(freqs-f0)*N/fs))/(2*np.pi*(freqs-f0)/fs) * np.exp(1j*phi)
    
    return vals

toneSpecKernel = cp.RawKernel(r'''
#include <cupy/complex.cuh>
#include <math_constants.h>
extern "C" __global__
void toneSpec_kernel(const double phi, const double f0, const double *freqs, int freqslen,
                     const double fs, const int N, const double A,
                     complex<double> *out)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    const double pi = 3.141592653589793;
    double phase;
    double common;
    double sinpart, cospart;
    complex<double> coeff;
    complex<double> phaseTerm;
    complex<double> numerator;
    
    // calculate the external phase first
    phase = phi - pi/2.0;
    sincos(phase, &sinpart, &cospart);
    phaseTerm = complex<double>(cospart, sinpart);

    for (int i = tid; i < freqslen; i += gridDim.x * blockDim.x)
    {
        common = 2.0 * (freqs[i] - f0) / fs;
        coeff = A/(common * pi);
        

        // calculate the numerator
        phase = -common * N;
        sincospi(phase, &sinpart, &cospart);
        numerator = 1.0 - complex<double>(cospart, sinpart);
        
        // final output
        out[i] = coeff * phaseTerm * numerator;

    }
}
''', 'toneSpec_kernel')  


def toneSpectrum_gpu(f0, d_freqs, fs, N, phi=0.0, A=1.0):
    '''
    Note that using this kernel is way faster than expressing it plainly in pythonic cupy (est ~7-8 times slower).
    '''
    out = cp.zeros((d_freqs.size), dtype=cp.complex128)
    
    THREADS_PER_BLOCK = 256
    NUM_BLOCKS = int(np.ceil(out.size/THREADS_PER_BLOCK) + 1)
    toneSpecKernel((NUM_BLOCKS,),(THREADS_PER_BLOCK,), (phi, f0, d_freqs, d_freqs.size, np.float64(fs), int(N), A, out))
                
    return out

toneSpecMultiKernel = cp.RawKernel(r'''
#include <cupy/complex.cuh>
#include <math_constants.h>
extern "C" __global__
void toneSpecMulti_kernel(const double *phi, const double *f0, const double *freqs, int freqslen,
                     const double fs, const int N, const double *A,
                     int m, complex<double> *out)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int blocksPerRow = freqslen/blockDim.x + ((freqslen % blockDim.x) > 0); // add 1 so that block exceeds the row length
    int blockLoadIdx = blockIdx.x % blocksPerRow;

    int row = tid/(blocksPerRow*blockDim.x);
    int col = tid%(blocksPerRow*blockDim.x);
    
    // Each block will process m rows * blockDim cols
    // That means each block need only read (blockDim) global memory values of freqs,
    // (m) values of phi, f0 and A.
    // It should be enforced that m < blockDim.
    
    // Load these into shared memory
    extern __shared__ double s[];
    double *s_phi = s; // (m) doubles
    double *s_f0 = (double*)&s_phi[m]; // (m) doubles
    double *s_A = (double*)&s_f0[m]; // (m) doubles
    double *s_freqs = (double*)&s_A[m]; // (blockDim.x) doubles
    
    // load shared memory
    for (int t = threadIdx.x; t < m; t = t + blockDim.x){
        s_phi[t] = phi[t];
    }
    for (int t = threadIdx.x; t < m; t = t + blockDim.x){
        s_f0[t] = f0[t];
    }
    for (int t = threadIdx.x; t < m; t = t + blockDim.x){
        s_A[t] = A[t];
    }
    for (int t = threadIdx.x; t < min(blockDim.x, freqslen); t = t + blockDim.x){ // either load up to the blockDim, or if freqs is shorter then to the freqslen
        s_freqs[t] = freqs[blockLoadIdx * blockDim.x + t];
    }
    
    
    __syncthreads();

    const double pi = 3.141592653589793;
    double phase;
    double common;
    double sinpart, cospart;
    complex<double> coeff;
    complex<double> phaseTerm;
    complex<double> numerator;
    
    double u_phi, u_f0, u_A;
    
    for (int mi = 0; mi < m; mi++){
        // broadcast used constants from shared mem for this iteration
        u_phi = s_phi[mi];
        u_f0 = s_f0[mi];
        u_A = s_A[mi];
            
        for (int i = threadIdx.x; i < blockDim.x; i += gridDim.x * blockDim.x)
        {
            common = 2.0 * (s_freqs[i] - u_f0) / fs; // each thread will read one value from shared mem
            coeff = u_A/(common * pi);
            
            // calculate the external phase first
            phase = u_phi - pi/2.0;
            sincos(phase, &sinpart, &cospart);
            phaseTerm = complex<double>(cospart, sinpart);
    
            // calculate the numerator
            phase = -common * N;
            sincospi(phase, &sinpart, &cospart);
            numerator = 1.0 - complex<double>(cospart, sinpart);
            
            // final output to global mem directly if it doesn't overshoot
            if (col < freqslen){
                out[mi * freqslen + col] = coeff * phaseTerm * numerator;
            }
        }
    }
}
''', 'toneSpecMulti_kernel')  


def toneSpectrumMulti_gpu(d_f0List, d_freqs, fs, N, d_phiList, d_AList):
    out = cp.zeros((d_f0List.size, d_freqs.size), dtype=cp.complex128)
    
    THREADS_PER_BLOCK = 256
    NUM_BLOCKS = int(out.size/THREADS_PER_BLOCK) + int((out.size%THREADS_PER_BLOCK)>0)
    
    if d_f0List.size > THREADS_PER_BLOCK:
        raise Exception('Reduce parameter size. %d > %d' % (d_f0List.size > THREADS_PER_BLOCK))
    
    toneSpecMultiKernel((NUM_BLOCKS,),(THREADS_PER_BLOCK,), (d_phiList, d_f0List, d_freqs, d_freqs.size, np.float64(fs), int(N), d_AList, d_f0List.size, out),
                        shared_mem=(THREADS_PER_BLOCK+3*d_f0List.size)*8)
                
    return out


if __name__=='__main__':
    # Unit testing for GPU code
    fs = 192000
    
    fineFreqRange = 5.0
    fineFreqStep = 0.001
    freqs = np.arange(-fineFreqRange,fineFreqRange+fineFreqStep/2, fineFreqStep)
    N = 145152
    
    phiSearch = np.arange(0,2*np.pi, 0.001)[:5000]
    
    f0 = 0.312
    
    t1 = time.time()
    
    for i in range(len(phiSearch)):
        ts = toneSpectrum(f0, freqs, fs, N, phi=phiSearch[i])
        
    t2 = time.time()
    print("CPU: %f s." % (t2-t1))
    
    
    
    d_freqs = cp.array(freqs)
    t1 = time.time()
    
    for i in range(len(phiSearch)):
        tsg = toneSpectrum_gpu(f0, d_freqs, fs, N, phi=phiSearch[i])
        
    t2 = time.time()
    print("Naive GPU Kernel: %f s." % (t2-t1))
    print("Check:")
    mi = np.argmax(np.abs(cp.asnumpy(tsg)-ts))
    print(tsg[mi])
    print(ts[mi])
    
    
    #%%
    d_f0List = cp.zeros(50) + f0
    d_AList = cp.zeros(50) + 1
    d_phiSearch = cp.array(phiSearch)
    
    t1 = time.time()
    
    
    for i in range(int(len(phiSearch)/50)):
        d_phiList = d_phiSearch[i*50:(i+1)*50]
        tsgm = toneSpectrumMulti_gpu(d_f0List, d_freqs, fs, N, d_phiList, d_AList)
        
    
    t2 = time.time()
    print("Batch-wise GPU Kernel: %f s." % (t2-t1))
    
    tsgcheck = cp.zeros((50,d_freqs.size), cp.complex128)
    for k in range(50):
        tsgcheck[k] = toneSpectrum_gpu(float(d_f0List[k]), d_freqs, fs, N, float(d_phiList[k]), float(d_AList[k]))
    
    print("Check:")
    mi = cp.argmax(cp.abs(tsgcheck.flatten()-tsgm.flatten()))
    print(tsgcheck.flatten()[mi])
    print(tsgm.flatten()[mi])

