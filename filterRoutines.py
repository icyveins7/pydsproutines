# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:58:31 2020

@author: Seo
"""

import os
import numpy as np
import scipy as sp
import scipy.signal as sps
import cupy as cp
import cupyx.scipy.signal as cpsps
import cpuWola as cpw
from plotRoutines import *
import scipy.cluster.vq as spc

#%% 
# Note that this function, which uses the cupy convolve, which in turn performs ffts,
# tends to be inaccurate at the start of the array (during the rampup i.e. when taps
# are being convolved with 0s)
# Try to use the new classes that use kernels below
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
    
class CupyFilter:
    def __init__(self, taps: cp.ndarray, force32f: bool=True):
        self.force32f = force32f
        if not force32f and taps.dtype != np.float32:
            raise TypeError("Taps dtype is incorrect, must be float32.")
        
        # Move to device if not yet
        self.taps = cp.asarray(taps).astype(cp.float32) # If already on device, does nothing
        
        # Interrim products
        self.delay = cp.zeros(taps.size, dtype=cp.complex64)
        
    def lfilter(self, x: cp.ndarray):
        if not self.force32f and x.dtype != cp.complex64:
            raise TypeError("x dtype is incorrect, must be complex64.")
        x = cp.asarray(x).astype(cp.complex64)
        
        # Pad the front
        xp = cp.hstack((self.delay, x))
        
        # Filter with the delay
        c = cpsps.convolve(self.taps, xp)
        
        # Set the new delay
        self.delay[:] = c[self.taps.size + x.size : ]
        
        # Return the filtered values
        cf = c[self.taps.size : self.taps.size + x.size]
        
        return cf
    
    def reset(self):
        self.delay[:] = 0
        
#%% Raw kernel with shared mem for filtering
class CupyKernelFilter:
    def __init__(self, memory: int=None, memory_dtype: type=cp.complex64):
        with open(os.path.join(os.path.dirname(__file__), "custom_kernels/filter.cu"), "r") as fid:
            sourcecode = fid.read()
        self.module = cp.RawModule(code=sourcecode)
        self.filter_smtaps_kernel = self.module.get_function("filter_smtaps")
        self.filter_smtaps_sminput_kernel = self.module.get_function("filter_smtaps_sminput")
        
        with open(os.path.join(os.path.dirname(__file__), "custom_kernels/upfirdn.cu"), "r") as fid:
            sourcecode = fid.read()
        self.module = cp.RawModule(code=sourcecode)
        self.upfirdn_naive_kernel = self.module.get_function("upfirdn_naive")

        if memory is not None:
            self.delay = cp.zeros(memory, dtype=memory_dtype)
        else:
            self.delay = None

    def resetDelay(self):
        if self.delay is not None:
            self.delay[:] = 0

    @staticmethod
    def getUpfirdnSize(originalSize: int, tapsSize: int, up: int, down: int):
        '''This should match size returned by sps.upfirdn.'''
        return int(np.ceil((originalSize * up - (up-1) + tapsSize-1) / down))
        
    def upfirdn_naive(
        self, d_x: cp.ndarray, d_taps: cp.ndarray, up: int, down: int,
        THREADS_PER_BLOCK: int=256, alsoReturnAbs: bool=False,
        d_out: cp.ndarray=None, d_outabs: cp.ndarray=None):
        '''
        Runs upfirdn, identical to scipy.signal.upfirdn.
        This calls a kernel which stores the taps in every block, so the length of the taps array
        is bounded by the shared memory maximum.

        Parameters
        ----------
        d_x : cp.ndarray, cp.complex64
            Input array.
        d_taps : cp.ndarray, cp.float32
            Filter taps array.
        up : int
            Upsampling factor.
        down : int
            Downsampling factor.
        THREADS_PER_BLOCK : int, optional
            Number of threads to use per block. The default is 256.
        alsoReturnAbs : bool, optional
            Specifies whether to concurrently return the abs of the output as a separate array.
            The default is False.

        Raises
        ------
        TypeError
            When array types are incorrect.

        Returns
        -------
        d_out : cp.ndarray, cp.complex64
            Output array.
        d_outabs : cp.ndarray, cp.float32
            Optional abs of the output array, which can be specified to be concurrently calculated.
        '''
        if d_x.dtype != cp.complex64:
            raise TypeError("d_x must be complex64.")
            
        if d_taps.dtype != cp.float32:
            raise TypeError("d_taps must be float32.")
            
        smReq = d_taps.nbytes
        
        # Allocate output length (this is designed to match sps.upfirdn)
        outlen = self.getUpfirdnSize(d_x.size, d_taps.size, up, down)
        if d_out is None:
            d_out = cp.zeros(outlen, cp.complex64)
        else:
            # Ensure correct type and required length
            if d_out.dtype != cp.complex64:
                raise TypeError("d_out must be complex64.")
            if d_out.size < outlen:
                raise ValueError("d_out must be at least length %d" % (outlen))
        
        # Run number of blocks to cover the output
        NUM_BLOCKS = outlen // THREADS_PER_BLOCK
        NUM_BLOCKS = NUM_BLOCKS + 1 if NUM_BLOCKS * THREADS_PER_BLOCK < outlen else NUM_BLOCKS
        
        # Optionally return absolute output as well
        if alsoReturnAbs:
            if d_outabs is None:
                d_outabs = cp.zeros(outlen, cp.float32)
            else:
                # Ensure correct type and required length
                if d_outabs.dtype != cp.float32:
                    raise TypeError("d_outabs must be float32.")
                if d_outabs.size < outlen:
                    raise ValueError("d_outabs must be at least length %d" % (outlen))
        
            # Run kernel
            self.upfirdn_naive_kernel(
                (NUM_BLOCKS,),(THREADS_PER_BLOCK,),
                (d_x, d_x.size,
                d_taps, d_taps.size,
                up, down,
                d_out, d_out.size, d_outabs),
                shared_mem=smReq
            )
            
            return d_out, d_outabs

        else:
            # Run kernel
            self.upfirdn_naive_kernel(
                (NUM_BLOCKS,),(THREADS_PER_BLOCK,),
                (d_x, d_x.size,
                d_taps, d_taps.size,
                up, down,
                d_out, d_out.size, 0), # Setting NULL to the pointer means it won't be written (see the kernel implementation)
                shared_mem=smReq
            )
            
            return d_out

    def run_upfirdn(self, d_x: cp.ndarray, d_taps: cp.ndarray, up: int, down: int, THREADS_PER_BLOCK: int=256):
        '''
        This method wraps the kernel call method with extra delay handling
        i.e. memory must be pre-allocated which will allow consequent calls to
        properly include the ending samples of the previous call.

        See upfirdn_naive.
        '''
        if self.delay is None:
            raise TypeError("Delay has not been allocated. Re-initialize with memory argument.")

        # Copy the data into a larger array
        d_xext = cp.hstack((self.delay, d_x))

        # Perform the filter on the extended array
        d_out = self.upfirdn_naive(d_xext, d_taps, up, down, THREADS_PER_BLOCK)

        # Copy the new delay into the holding array
        self.delay[:] = d_x[-self.delay.size:]

        # Skip the delay part, and only return the necessary length
        length2return = int(d_x.size * up // down)
        skip = int(self.delay.size * up // down)

        return d_out[skip:skip+length2return]


        
        
    def filter_smtaps(self,
        d_x: cp.ndarray,
        d_taps: cp.ndarray, 
        THREADS_PER_BLOCK: int=128, 
        OUTPUT_PER_BLK: int=128,
        useInternalDelay: bool=False
    ):
        # Type checking
        assert(d_taps.dtype == cp.float32)
        assert(d_x.dtype == cp.complex64)
        # Dimension checking
        assert(d_x.ndim == 1 and d_taps.ndim == 1)
        # Maximum length checking
        assert(d_taps.nbytes <= 48000)
        
        # Allocate output
        d_out = cp.zeros(d_x.size, dtype=cp.complex64)
        
        # Calculate shared memory requirement
        smReq = d_taps.nbytes
        
        # Calculate number of blocks required and the output size per block
        NUM_BLOCKS = d_x.size // OUTPUT_PER_BLK
        NUM_BLOCKS = NUM_BLOCKS + 1 if d_x.size % OUTPUT_PER_BLK != 0 else NUM_BLOCKS # +1 if remnants
        
        # Use delay if specified
        if useInternalDelay:
            delay = self.delay
            delaylen = self.delay.size
        else:
            delay = 0
            delaylen = 0

        # Run kernel
        self.filter_smtaps_kernel(
            (NUM_BLOCKS,),(THREADS_PER_BLOCK,), 
            (d_x, d_x.size,
             d_taps, d_taps.size,
             OUTPUT_PER_BLK,
             d_out, d_out.size,
             delay, delaylen), # These are optional parameters for the delay
            shared_mem=smReq
        )
        
        return d_out

    def run_filter_smtaps(self, d_x: cp.ndarray, d_taps: cp.ndarray, THREADS_PER_BLOCK: int=128, OUTPUT_PER_BLK: int=128):
        if self.delay is None:
            raise TypeError("Delay has not been allocated. Re-initialize with memory argument.")

        # Run the filter (with the delay)
        d_out = self.filter_smtaps(d_x, d_taps, THREADS_PER_BLOCK, OUTPUT_PER_BLK, True)

        # Extract the delay from this invocation
        self.delay[:] = d_x[-self.delay.size:]

        return d_out
 
    
    def filter_smtaps_sminput(self, d_x: cp.ndarray, d_taps: cp.ndarray, THREADS_PER_BLOCK: int=128):
        # Type checking
        assert(d_taps.dtype == cp.float32)
        assert(d_x.dtype == cp.complex64)
        # Dimension checking
        assert(d_x.ndim == 1 and d_taps.ndim == 1)
        # Maximum length checking
        assert(d_taps.size < 2400)
        # Note that this length is fixed as we draw a line at a minimum of 2 * tapslength for the workspace.
        # In theory, it can still work with anything more than 1 * tapslength, but this is the line we draw.
        
        # Allocate output
        d_out = cp.zeros(d_x.size, dtype=cp.complex64)
        
        # Calculate the workspace available (we move in multiples of THREADS_PER_BLOCK)
        workspaceFactor = ((48000 - d_taps.nbytes) - (d_taps.size-1) * 8) // 8 // THREADS_PER_BLOCK
        # print(workspaceFactor)
        workspaceSize = workspaceFactor * THREADS_PER_BLOCK + d_taps.size - 1
        OUTPUT_PER_BLK = workspaceFactor * THREADS_PER_BLOCK
        
        # Calculate shared memory requirement
        smReq = d_taps.nbytes + workspaceSize * 8
        
        # Calculate number of blocks required and the output size per block
        NUM_BLOCKS = d_x.size // OUTPUT_PER_BLK
        NUM_BLOCKS = NUM_BLOCKS + 1 if d_x.size % OUTPUT_PER_BLK != 0 else NUM_BLOCKS # +1 if remnants
        
        # Run kernel
        self.filter_smtaps_sminput_kernel(
            (NUM_BLOCKS,),(THREADS_PER_BLOCK,), 
            (d_x, d_x.size,
             d_taps, d_taps.size,
             OUTPUT_PER_BLK,
             workspaceSize,
             d_out, d_out.size),
            shared_mem=smReq
        )
        
        return d_out
        

#%% Raw kernel for upfirdn
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

# def cupyUpfirdn(x: cp.ndarray, taps: cp.ndarray, up: int, down: int):
#     # if x.dtype != cp.complex64:
#     #     raise TypeError("x is expected to be type complex64.")
#     # if taps.dtype != cp.float32:
#     #     raise TypeError("taps is expected to be type float32.")
        
#     # Allocate output
#     out = cp.zeros(x.size * up // down, dtype=cp.complex64)
        
#     # Define just enough blocks to cover the output
#     THREADS_PER_BLOCK = 256
#     NUM_BLOCKS = out.size // THREADS_PER_BLOCK
#     if NUM_BLOCKS * THREADS_PER_BLOCK < out.size:
#         NUM_BLOCKS += 1
        
#     # Define the shared memory requirement
#     if taps.size * 4 > 48e3:
#         raise MemoryError("Taps length too large for shared memory.")
#     smReq = taps.size * 4
    
#     # Call the kernel
#     upFirdnKernel((NUM_BLOCKS,),(THREADS_PER_BLOCK,), 
#                   (x, x.size,
#                    taps, taps.size,
#                    up, down,
#                    out, out.size),
#                   shared_mem=smReq)
    
#     return out


upFirdn_smKernel = cp.RawKernel('''
#include <cupy/complex.cuh>
extern "C" __global__
void upfirdn_sm(
    const complex<float> *d_x, const int len,
    const float *d_taps, const int tapslen,
    const int up,
    const int down,
    const int shm_x_size,
    complex<float> *d_out, int outlen, float *d_outabs)
{
    // allocate shared memory
    extern __shared__ double s[];
    
    float *s_taps = (float*)s; // (tapslen) floats
    complex<float> *s_x = (complex<float>*)&s_taps[tapslen]; // (shm_x_size) complex floats
    /* Tally:  */

    // load shared memory
    for (int t = threadIdx.x; t < tapslen; t = t + blockDim.x){
        s_taps[t] = d_taps[t];
    }
    
    // Define the indices to write to for this block
    int outStart = blockIdx.x * blockDim.x + tapslen / 2;
    int outEnd = min((blockIdx.x + 1) * blockDim.x + tapslen / 2, outlen + tapslen/2);
    
    // calculate the offset for this block
    int blockReadOffset = (outStart * down - tapslen) / up; // TODO: define this
    // note that shm_x_size must this extra front buffer as well
    for (int t = threadIdx.x; t < shm_x_size; t = t + blockDim.x)
    {
        if (t + blockReadOffset >= 0 && t + blockReadOffset < len){ // only read if in range
            s_x[t] = d_x[t + blockReadOffset];
        }
        else{
            s_x[t] = 0;
        }
    }
    __syncthreads();
    
    // Begin computations
    int i0, j;
    complex<float> z = 0; // Stack-variable for each thread
    
    // Make it simple, every thread writes 1 output
    int k = threadIdx.x + outStart;
    if (k < outEnd)
    {
        for (int i = 0; i < tapslen; i++)
        {
            i0 = down * k - i;
            // don't bother reading if its non-zero
            if (i0 % up == 0){
                j = i0 / up; // this is the access into the 'x' array
                j = j - blockReadOffset; // we only copied a section into shared memory, so change the index
                
                if (j < shm_x_size && j >= 0) // cannot read out of bounds
                {
                    z = z + s_taps[i] * s_x[j];
                }
            }
            
        }
        
        // write to global memory, coalesced, and offset half the filter automatically
        d_out[k - tapslen / 2] = z;
        d_outabs[k - tapslen / 2] = abs(z);
    }

 
}
''', '''upfirdn_sm''')


# def cupyUpfirdn_sm(x: cp.ndarray, taps: cp.ndarray, up: int, down: int,
#                    out: cp.ndarray=None, outabs: cp.ndarray=None):
#     '''
#     Runs a custom, shared-memory optimised kernel to perform the upfirdn
#     onboard the GPU. Note that if outputs are incorrect, it is likely that the
#     arrays' dtypes are incorrect.

#     Parameters
#     ----------
#     x : cp.ndarray
#         Input, complex64.
#     taps : cp.ndarray
#         Filter taps, float32.
#     up : int
#         Upsampling factor.
#     down : int
#         Downsampling factor.
#     out : cp.ndarray, optional
#         Output array, if already allocated, complex64. The default is None.
#     outabs : cp.ndarray, optional
#         Abs(output array), if already allocated, float32. The default is None.

#     Raises
#     ------
#     MemoryError
#         Raised when the length of taps is too large.

#     Returns
#     -------
#     out : cp.ndarray
#         Output array.
#     outabs : cp.ndarray
#         Abs(output array).

#     '''
#     # Allocate output
#     if out is None:
#         out = cp.zeros(x.size * up // down, dtype=cp.complex64)
#     if outabs is None:
#         outabs = cp.zeros(x.size * up // down, dtype=cp.float32)
    
#     THREADS_PER_BLOCK = 256
#     NUM_BLOCKS = out.size // THREADS_PER_BLOCK
#     NUM_BLOCKS = NUM_BLOCKS + 1 if out.size % THREADS_PER_BLOCK > 0 else NUM_BLOCKS
    
#     # Define the shared memory requirement
#     shm_x_size = ((THREADS_PER_BLOCK * down + taps.size) // up) + 1
#     # print(shm_x_size)
#     if taps.size * 4 + shm_x_size * 8 > 48e3:
#         raise MemoryError("Shared memory requested exceeds 48kB.")
#     smReq = taps.size * 4 + shm_x_size * 8
    
#     # Call the kernel
#     upFirdn_smKernel((NUM_BLOCKS,),(THREADS_PER_BLOCK,), 
#                   (x, x.size,
#                    taps, taps.size,
#                    up, down, shm_x_size,
#                    out, out.size, outabs),
#                   shared_mem=smReq)
    
#     return out, outabs

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

    def channelFreqs(self, fs: float=1.0):
        '''Returns the centre frequency for each channel.'''
        return makeFreq(self.numChannels, fs)

    def channelFs(self, fs: float=1.0):
        '''Returns the new sampling rate for each channel.'''
        return fs / self.Dec
        
    

#%%
class BurstDetector:
    def __init__(self, medfiltlen: int):
        self.medfiltlen = medfiltlen
        
        # Placeholders for later results
        self.d_absx = None
        self.d_ampSq = None
        self.d_medfiltered = None
        self.threshold = None
        self.codebook = None
        
    def medfilt(self, x: cp.ndarray):
        '''
        Runs the median filter on input complex data.
        No need to run abs() on your input data first!

        Parameters
        ----------
        x : cp.ndarray
            Input array.
        '''
        d_x = cp.asarray(x) # Push to gpu if not currently in it
        self.d_absx = cp.abs(d_x)
        self.d_ampSq = self.d_absx * self.d_absx
        self.d_medfiltered = cpsps.medfilt(self.d_ampSq, self.medfiltlen)

    @staticmethod
    def imposeSignalLengthLimits(signalIndices: list, minLength: int=0, maxLength: int=None):
        '''
        Use this after signalIndices are returned from the detection methods
        in order to weed out the nonsense ones.
        '''
        if maxLength is None:
            maxLength = 4294967295 # arbitrarily gonna set uint32 4294967295 as the max
        return [i for i in signalIndices if i.size >= minLength and i.size <= maxLength]

    @staticmethod
    def getStartAndEndIdx(signalIdx: np.ndarray):
        return signalIdx[0], signalIdx[-1]
        
    def detectViaThreshold(self, threshold: float):
        self.threshold = threshold # Kept for plotting
        signalIndices = cp.argwhere(self.d_medfiltered > threshold).flatten()
        splitIndices = cp.argwhere(cp.diff(signalIndices)>1).flatten() + 1 # the + 1 is necessary
        signalIndices = cp.split(signalIndices, splitIndices.get()) # For cupy, need to pull the split indices to host
        
        return signalIndices

    def autoDetectThreshold(self, noiseLevels: np.ndarray, multiplier: float=1.0):
        '''
        Attempts to detect a suitable threshold by estimating the noise level.
        The noise level is assumed to be the first 'plateau' of sample values from the median filtered power array.

        To estimate this, a histogram is performed on the median filtered array,
        but this is limited to the noiseLevels input specified; there is no need to calculate bins
        up to the maximum value in the array.
        The threshold is then defined as the first bin which is lower than its two adjacent bins.
        It is clear that doing it this way is probably an over-estimate of the 'mean noise level',
        so an optional multiplier may be supplied.

        Parameters
        ----------
        noiseLevels : np.ndarray
            Array specifying the bin edges for the histogram.
            The following advice will be useful:
                1) First value should be 0 i.e. generate with np.arange(0, ..., ...).
                2) The step size should be constant, and should be 'fairly large'.
                   This is to prevent the bin counts from being too sparse and prone to local minima.
                3) The size of this array will affect computation time, so don't use too many values.

        multiplier : float, optional
            Constant multiplier that is applied to the detected threshold. The default is 1.0.
            Since the algorithm essentially slightly overestimates the noise level, it is likely
            that using a multiplier less than 1.0 will work, especially if combined with the other constraints
            later on, like the signal length limits.
        '''

        counts, edges = cp.histogram(self.d_medfiltered, noiseLevels)
        counts = counts.get()
        for i in range(counts.size - 1):
            if counts[i] < counts[i-1] and counts[i] < counts[i+1]:
                detectedThreshold = noiseLevels[i]
                return detectedThreshold * multiplier

        return None # Otherwise return None for failure
    
    def detectSingleEmitter(self, ratio: float):
        # To seed the kmeans (which speeds it up ~30x), we find the max, and a 
        # sample which is greater than the ratio difference
        x = self.d_medfiltered.get()
        bigClusterSeed = np.max(x)
        smallClusterSeed = x[x < (bigClusterSeed/ratio)][0]
        codebook, distortion = spc.kmeans(x,np.array([smallClusterSeed, bigClusterSeed]))
        self.codebook = np.sort(codebook)
        self.threshold = np.mean(codebook)
        # Codify the samples
        codes, dists = spc.vq(x, self.codebook)
        # Match to the big cluster
        signalIndices = np.argwhere(codes == 1).reshape(-1)
        # Split as usual
        splitIndices = np.argwhere(np.diff(signalIndices)>1).reshape(-1) + 1
        signalIndices = np.split(signalIndices, splitIndices)
        
        return signalIndices
        
    def detectRegularSections(self, sectionSizeRange: np.ndarray):
        '''
        Uses the samples' power to estimate the length of a period for bursty signals.
        
        Assume a signal consists of a burst duration followed by a guard duration, which together constitutes the period.
        By sectioning into test periods and taking the mean power over the periods,
        then performing a simple kmeans estimate to cluster, we should see that the correct period length will have
        the widest spacing in the clusters. This is used to find the period length.
        
        Depending on the resolution required, it is likely that a rough estimate should be used to find the coarse period,
        then a finer, second search can be applied.
        
        Example:
            detectRegularSections(np.arange(..., ..., 1000) # Skip 1000 samples at a time
            detectRegularSections(np.arange(coarse-1000, coarse+1000, 1) # Search around the coarse estimate at sample level
        '''
        metric = np.zeros((sectionSizeRange.size, 2))
        codebooks = np.zeros((sectionSizeRange.size, 2))
        for i, partitionSize in enumerate(sectionSizeRange):
            partitioned = cp.abs(self.d_medfiltered)
            if self.d_medfiltered.size % partitionSize > 0:    
                partitioned = cp.abs(self.d_medfiltered)[:-(self.d_medfiltered.size % partitionSize)] # Slice off the ends if needed to be a multiple
            partitioned = partitioned.reshape((-1, partitionSize))
            # Take mean down the columns
            partitionMeans = cp.mean(partitioned, axis=0)
            
            ratio = 1.5
            x = partitionMeans.get()
            bigClusterSeed = np.max(x)
            try:
                smallClusterSeed = x[x < (bigClusterSeed/ratio)][0]
            except:
                smallClusterSeed = np.min(x)
            codebook, distortion = spc.kmeans(x,np.array([smallClusterSeed, bigClusterSeed]))
            codebook = np.sort(codebook)
            codebooks[i,:] = codebook

            # Codify the samples
            codes, dists = spc.vq(x, codebook)
            
            print("partitionSize = %d, codebook clustering = %f, distortion = %f" % (partitionSize, np.diff(codebook)[0], distortion))
            print("num0s = %d, num1s = %d" % (np.argwhere(codes==0).size, np.argwhere(codes==1).size))
            metric[i, 0] = np.diff(codebook)[0]
            metric[i, 1] = distortion
            
        return metric, codebooks
    
    def pgplot(self, ax=None, fs=1, start=0, end=-1):
        if self.d_ampSq is None:
            raise ValueError("Run medfilt() first.")
        
        
        rwin, rax = pgPlotAmpTime([self.d_ampSq.get()[start:end], self.d_medfiltered.get()[start:end]],
                                  [fs, fs],
                                  labels=["Power", "Medfilt"],
                                  colors=["r", "b"],
                                  ax=ax)
        if self.threshold is not None:
            rax.addItem(pg.InfiniteLine(self.threshold, angle=0, movable=False, label='Threshold'))
        if self.codebook is not None:
            rax.addItem(pg.InfiniteLine(self.codebook[0], angle=0, movable=False, label='Noise Cluster'))
            rax.addItem(pg.InfiniteLine(self.codebook[1], angle=0, movable=False, label='Signal Cluster'))
        return rwin, rax

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

#%%
def resampleFactorWizard(fs: int, rsfs: int):
    '''
    Convenience function to get up and down integer factors from a starting
    sample rate to a target sample rate.

    Parameters
    ----------
    fs : int
        Initial sample rate.
    rsfs : int
        Target (resampled) sample rate.

    Returns
    -------
    up : int
        Upsampling factor.
    down : int
        Downsampling factor.

    '''
    # Enforce types
    fs = int(fs)
    rsfs = int(rsfs)
    # Interrim sampling rate
    l = np.lcm(fs, rsfs)
    up = l // fs
    down = l // rsfs
    return int(up), int(down) # Enforce again in case
    

