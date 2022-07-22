# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:21:35 2020

@author: Seo
"""

import numpy as np
from timingRoutines import Timer
from numba import njit, jit
from xcorrRoutines import *
import warnings

try:
    import cupy as cp
    
    def cupyDemodulateCP2FSK(syms: cp.ndarray, h: float, up: int):
        m = cp.array([[-1],
                      [+1]])
        
        tones = cp.exp(1j*np.pi*h*cp.arange(up)/up*m)
        
        numSyms = int(np.floor(len(syms) / up))
        bitCost = cp.zeros((2,numSyms))
        demodBits = cp.zeros(numSyms, dtype = np.uint8)
        
        for i in range(numSyms):
            symbol = syms[i*up : (i+1)*up]
            
            for k in range(2):
                bitCost[k,i] = cp.abs(cp.vdot(symbol, tones[k]))
            
            demodBits[i] = cp.argmax(bitCost[:,i])
            
        return demodBits, bitCost, tones
except:
    print("Cupy not found.")

#%% Generic simple demodulators
class SimpleDemodulatorPSK:
    '''
    Generic demodulator implementation for BPSK/QPSK/8PSK.
    This uses a dot product method to detect which symbol in the constellation is present.
    
    The default constellation-bit mappings are provided as gray-mapped bits,
    but this can be changed.
    '''
    
    # These default psk constellations are provided only for the basic class.
    # The specialised classes use their own constellations which are optimised for demodulation.
    pskdicts = { # This is a monotonically increasing index for each increase in angle
        2: np.array([1.0, -1.0], dtype=np.complex128),
        4: np.array([1.0, 1.0j, -1.0, -1.0j], dtype=np.complex128),
        8: np.array([1.0,
                     np.sqrt(2)/2 * (1+1j),
                     1.0j,
                     np.sqrt(2)/2 * (-1+1j),
                     -1.0,
                     np.sqrt(2)/2 * (-1-1j),
                     -1.0j,
                     np.sqrt(2)/2 * (1-1j)], dtype=np.complex128)
    }
    # This is a specific bit mapping for each corresponding index i.e. each angle, in increasing order
    # E.g. QPSK/8PSK are gray mapped.
    pskbitmaps = { 
        2: np.array([0b1, 0b0], dtype=np.uint8),
        4: np.array([0b11, 0b01, 0b00, 0b10], dtype=np.uint8),
        8: np.array([0b000,
                     0b001,
                     0b011,
                     0b010,
                     0b110,
                     0b111,
                     0b101,
                     0b100], dtype=np.uint8)
    }
    
    def __init__(self, m: int, bitmap: np.ndarray=None, cluster_threshold: float=0.1):
        self.m = m
        self.const = self.pskdicts[self.m]
        self.normVecs = self.const.view(np.float64).reshape((-1,2))
        self.bitmap = self.pskbitmaps[self.m] if bitmap is None else bitmap
        self.cluster_threshold = cluster_threshold
        
        # Interrim output
        self.xeo = None # Selected eye-opening resample points
        self.xeo_i = None # Index of eye-opening
        self.eo_metric = None # Metrics of eye-opening
        self.reimc = None # Phase-locked to constellation (complex array)
        self.svd_metric = None # SVD metric for phase lock
        self.angleCorrection = None # Angle correction used in phase lock
        self.syms = None # Output mapping to each symbol (0 to M-1)
        self.matches = None # Output from amble rotation search
        
        
    def getEyeOpening(self, x: np.ndarray, osr: int, abs_x: np.ndarray=None):
        if abs_x is None:
            abs_x = np.abs(x) # Provide option for pre-computed (often used elsewhere anyway)
        x_rs_abs = abs_x.reshape((-1, osr))
        self.eo_metric = np.mean(x_rs_abs, axis=0)
        i = np.argmax(self.eo_metric)
        x_rs = x.reshape((-1, osr))
        return x_rs[:,i], i
        
    def mapSyms(self, reimc: np.ndarray):
        '''
        Maps symbols to values from 0 to m-1. Note that this may not correspond to the 
        bit values desired e.g. gray mapping. In such scenarios, the bitmap should be amended.

        This method does not need to be called directly; it is called as part of demod().
        
        See symsToBits() for actual bit mapping.

        Parameters
        ----------
        reimc : np.ndarray
            Correct eye-opening, frequency corrected and phase-locked complex-valued input.

        Returns
        -------
        syms : np.ndarray
            Output array corresponding to the symbol values 0 to m-1.

        '''
        reimcr = reimc.view(np.float32).reshape((-1,2)).T
        constmetric = self.normVecs @ reimcr
        # Pick the arg max for each column
        syms = np.argmax(constmetric, axis=0).astype(np.uint8)
        
        return syms
    
    def lockPhase(self, reim: np.ndarray):
        # Power into BPSK
        powerup = self.m // 2
        reimp = reim**powerup
        
        # Form the square product
        reimpr = reimp.view(np.float32).reshape((-1,2)).T
        reimsq = reimpr @ reimpr.T
        
        # SVD
        u, s, vh = np.linalg.svd(reimsq) # Don't need vh technically
        # Check the svd metrics
        svd_metric = s[-1] / s[:-1] # Deal with this later when there is residual frequency
        if np.any(svd_metric > self.cluster_threshold):
            warnings.warn("Constellation not well clustered. There may be residual frequency shifts.")
        # Angle correction
        angleCorrection = np.arctan2(u[1,0], u[0,0])
        reimc = self.correctPhase(reim, -angleCorrection/powerup)
        
        return reimc, svd_metric, angleCorrection
    
    def correctPhase(self, reim: np.ndarray, phase: float):
        return reim * np.exp(1j * phase)
        
    
    def demod(self, x: np.ndarray, osr: int, abs_x: np.ndarray=None, verb: bool=True):
        if x.dtype != np.complex64:
            raise TypeError("Input array must be complex64.")
        
        timer = Timer()
        timer.start()
        
        # Get eye-opening first
        xeo, xeo_i = self.getEyeOpening(x, osr, abs_x)
        timer.evt("Eye-opening")
        
        # Correct the global phase first
        reim = np.ascontiguousarray(xeo)
        self.reimc, self.svd_metric, self.angleCorrection = self.lockPhase(reim)
        timer.evt("lockPhase")
        
        # Generic method: dot product with the normalised vectors
        self.syms = self.mapSyms(self.reimc)
        timer.evt("mapSyms")
        
        if verb:
            timer.rpt()
        
        return self.syms
    
    def ambleRotate(self, amble: np.ndarray, search: np.ndarray=None, syms: np.ndarray=None):
        if syms is None:
            syms = self.syms
        
        if search is None:
            search = np.arange(syms.size - amble.size + 1)
            
        # Naive loop
        length = amble.size
        m_amble = amble + self.m # Scale it up in order to do uint8 math
        
        # # Pythonic loop
        # self.matches = np.zeros((search.size, self.m), dtype=np.uint32)
        # for i, mi in enumerate(search):
        #     diff = (m_amble - syms[mi:mi+length]) % self.m
        #     for k in range(self.m):
        #         self.matches[i, k] = np.sum(diff == k)
        
        # Numba loop
        self.matches = self._ambleSearch(m_amble, search, self.m, syms, length)
        
        # # Numba loop v2
        # self.matches = self._ambleSearch(amble, search, self.m, syms, length)
                
        s, rotation = argmax2d(self.matches)
        sample = search[s] # Remember to reference the searched indices
        self.syms = (syms + rotation) % self.m
        
        return self.syms, sample, rotation
    
    @staticmethod
    @njit('uint32[:,:](uint8[:], int32[:], intc, uint8[:], intc)', cache=True, nogil=True)
    def _ambleSearch(m_amble, search, m, syms, length):
        matches = np.zeros((search.size, m), dtype=np.uint32)
        for i in np.arange(search.size): # Use np.arange instead of range
            mi = search[i]
            diff = np.mod((m_amble - syms[mi:mi+length]), m)
            
            # One-pass loop
            for k in np.arange(diff.size):
                matches[i, diff[k]] += 1
        
        return matches
    
    # @staticmethod
    # @njit(cache=True, nogil=True) # not well tested yet
    # def _ambleSearchv2(m_amble, search, m, syms, length):
    #     matches = np.zeros((search.size, m), dtype=np.uint32)
    #     for i in np.arange(search.size): # Use np.arange instead of range
    #         mi = search[i]
    #         diff = np.bitwise_xor(amble, syms[mi:mi+length])
    #         # One-pass loop
    #         for k in np.arange(diff.size):
    #             matches[i, -1-diff[k]] += 1
        
    #     return matches
    
    def symsToBits(self, syms: np.ndarray=None):
        '''
        Maps each symbol (integer array denoting the angle) to its own bit sequence,
        as specified by the bitmap.

        Parameters
        ----------
        syms : np.ndarray, uint8, optional
            Input symbol sequence. The default is None, which will use the last internally saved
            syms array output.

        Returns
        -------
        bits : np.ndarray
            Bit sequence stored as individual bytes i.e. length of this array = length of syms.

        '''
        if syms is None:
            syms = self.syms
        
        return self.bitmap[syms]
        
    
    def unpackToBinaryBytes(self, packed: np.ndarray):
        '''
        Turns an integer valued output from mapSyms()/demod()/symsToBits() into
        a binary-valued array with each row corresponding to the binary value
        of the integer.
        
        Specifically, that means that each bit now occupies one byte in memory,
        hence the name of the method. Contrast this with the packBinaryBytesToBits()
        method which tends to follow.
        
        Example:
            m = 4.
            Input array [0,1,2,3].
            Output is [[0,0],
                       [0,1],
                       [1,0],
                       [1,1]].

        Parameters
        ----------
        packed : np.ndarray
            Integer valued array.

        Returns
        -------
        unpacked : np.ndarray
            Matrix of N x k binary values, where N is the original length of 'packed',
            and k is the number of bits used to represent each value of 'packed',
            given by log2(m).

        '''

        bitsPerVal = int(np.log2(self.m))
        # Unpack as usual
        unpacked = np.unpackbits(packed).reshape((-1,8))
        # Slice the ending bits (default is big-endian)
        unpacked = unpacked[:,-bitsPerVal:]
        
        return unpacked
    
    def packBinaryBytesToBits(self, unpacked: np.ndarray):
        '''
        This is a simple wrapper around numpy's packbits().
        In this context, it takes the unpacked matrix from unpackToBinaryBytes()
        and then compresses it to occupy the minimum requirement of bytes storage.
        
        Example:
            Input (QPSK) array [[0,0],
                                [0,1],
                                [1,0],
                                [1,1]].
            This is compressed to a single byte corresponding to 
            [0,0,0,1,1,0,1,1], which is then returned as array([27]).

        Parameters
        ----------
        unpacked : np.ndarray
            Input unpacked bits, usually from unpackToBinaryBytes().

        Returns
        -------
        packed : np.ndarray
            Packed bits storage of the input.
        '''
        return np.packbits(unpacked.reshape(-1))
        
    
###############
class SimpleDemodulatorBPSK(SimpleDemodulatorPSK):
    '''
    Faster demodulator implementation specifically for BPSK.
    '''
    def __init__(self, bitmap: np.ndarray=None, cluster_threshold: float=0.1):
        super().__init__(2, bitmap, cluster_threshold)
        
    def mapSyms(self, reimc: np.ndarray):
        # Simply get the real
        re = np.real(reimc)
        
        # And check sign
        syms = (re < 0).astype(np.uint8)
        
        return syms
        
    
    
###############
class SimpleDemodulatorQPSK(SimpleDemodulatorPSK):
    '''
    Faster demodulator implementation specifically for QPSK.
    '''
    def __init__(self, bitmap: np.ndarray=None, cluster_threshold: float=0.1):
        super().__init__(4, bitmap, cluster_threshold)
    
        self.gray4 = np.zeros((2,2), dtype=np.uint8)
        self.gray4[1,1] = 0
        self.gray4[0,1] = 1
        self.gray4[0,0] = 2
        self.gray4[1,0] = 3
        # This is X,Y > 0 gray encoded
        
    def mapSyms(self, reimc: np.ndarray):
        # Reshape
        reimd = reimc.view(np.float32).reshape((-1,2))
        
        # # Compute comparators
        # xp = (reimd[:,0] > 0).astype(np.uint8)
        # yp = (reimd[:,1] > 0).astype(np.uint8)
        
        # # Now map
        # idx = np.vstack((xp,yp))
        # # Convert to constellation integers
        # syms = self.gray4[tuple(idx)]
        
        # New one-liner, prevents multiple comparator calls hence faster?
        syms = self.gray4[tuple((reimd > 0).T.astype(np.uint8))]
        
        return syms
    
    def correctPhase(self, reim: np.ndarray, phase: float):
        # For gray-coding comparators, we move to the box
        return reim * np.exp(1j*(phase + np.pi/4)) 

################
class SimpleDemodulator8PSK(SimpleDemodulatorPSK):
    '''
    Faster demodulator implementation specifically for 8PSK.
    '''
    def __init__(self, bitmap: np.ndarray=None, cluster_threshold: float=0.1):
        super().__init__(8, bitmap, cluster_threshold)
        
        # For the custom constellation, we don't map to a number but rather to the N-D index,
        # mirroring the actual bits.
        self.map8 = np.zeros((2,2,2), dtype=np.uint8)
        self.map8[1,1,1] = 0
        self.map8[0,1,1] = 1
        self.map8[1,0,1] = 2
        self.map8[0,0,1] = 3
        self.map8[1,1,0] = 4
        self.map8[0,0,0] = 5
        self.map8[1,0,0] = 6
        self.map8[0,1,0] = 7
        
    def mapSyms(self, reimc: np.ndarray):
        # 8PSK specific, add dimensions
        reimd = reimc.view(np.float32).reshape((-1,2))
        scaling = np.max(self.eo_metric) # Assumes eye-opening has been done
        reim_thresh = np.abs(np.abs(np.cos(np.pi/8)*scaling) - np.abs(np.sin(np.pi/8)*scaling))
        # Compute |X| - |Y|
        xmy = np.abs(reimd[:,0]) - np.abs(reimd[:,1])
        # And then | |X| - |Y| | + c, this transforms into QPSK box below XY plane
        # with the new QPSK diamond above XY plane
        z = np.abs(xmy) - reim_thresh # Do not stack into single array, no difference anyway
        
        # C1: Check Z > 0; if + check even (diamond), if - check odd (QPSK, box)
        c1z = z > 0
        
        # C2: Z+ check XY and end, Z- check |X|-|Y| and C3
        cx2 = reimd[:,0] > 0
        cy2 = reimd[:,1] > 0
        cxmy2 = xmy > 0

        # C3: + check X, - check Y
        cx3 = np.logical_and(cxmy2, cx2)
        cy3 = np.logical_and(np.logical_not(cxmy2), cy2)
        
        # Build backwards
        idx1 = cxmy2
        idx2 = np.logical_or(cx3, cy3)
        
        idx1 = np.logical_or(np.logical_and(c1z, idx1), np.logical_and(np.logical_not(c1z), cx2))
        idx2 = np.logical_or(np.logical_and(c1z, idx2), np.logical_and(np.logical_not(c1z), cy2))
        
        idx0 = c1z
        
        # Now map
        idx = np.vstack((idx0.astype(np.uint8),idx1.astype(np.uint8),idx2.astype(np.uint8)))
        # Converts to the default demodulator constellation integers
        syms = self.map8[tuple(idx)] # Needs to be vstack, and need the tuple(); need each value to be a column of indices
        
        return syms
        
        


#%%
@jit(nopython=True)
def demodulateCP2FSK(syms, h, up):
    m = np.array([[-1],
                  [+1]]) # these map to [0, 1] bits
    
    # create the two tones
    tones = np.exp(1j*np.pi*h*np.arange(up)/up * m)
    
    # loop over each upsampled section of a symbol
    numSyms = int(np.floor(len(syms) / up))
    bitCost = np.zeros((2,numSyms))
    demodBits = np.zeros(numSyms, dtype = np.uint8)
    
    for i in range(numSyms):
        symbol = syms[i*up : (i+1)*up]
        
        for k in range(2):
            bitCost[k,i] = np.abs(np.vdot(symbol, tones[k]))
        
        demodBits[i] = np.argmax(bitCost[:,i])
        
    return demodBits, bitCost, tones



##
class BurstyDemodulator:
    '''
    This class and all derived versions attempt to perform an aligned, synchronous
    demodulation of all bursts at the same time, amalgamating the cost functions into one.
    This prevents the misalignment (usually by one symbol) of the bursts, which when used
    in remodulation of the observed signal, can cause grave reconstruction errors in the 
    differential modulation modes (DQPSK, CPFSK etc.).
    '''
    def __init__(self, burstLen: int, guardLen: int, up: int=1):
        self.burstLen = burstLen
        self.guardLen = guardLen
        self.period = self.burstLen + self.guardLen
        self.up = up
        
    def demod(self, x: np.ndarray, numBursts: int, searchIdx: np.ndarray=None):
        raise NotImplementedError("Only invoke with derived classes.")
        
##
class BurstyDemodulatorCP2FSK(BurstyDemodulator):
    def __init__(self, burstLen: int, guardLen: int, up: int=1, h: float=0.5):
        super().__init__(burstLen, guardLen, up) # Refer to parent class
        # Extra params
        self.h = h
        
        # Configurations
        self.burstIdxs = None
        
        # Outputs
        self.d_costs = None
        
    def demod(self, x: np.ndarray, numBursts: int=None, searchIdx: np.ndarray=None):
        # t1 = time.perf_counter()
        if self.burstIdxs is None: # Priority is to use the pre-set burstIdxs
            if numBursts is None: # Otherwise we generate it here using a simple np.arange
                raise ValueError("Please call setBurstIdxs() before demodulating or set the numBursts argument.")
            else:
                self.setBurstIdxs(np.arange(numBursts))
        
        # Construct the tones
        gtone = np.exp(1j*np.pi*self.h*np.arange(self.up)/self.up)
        tones = np.vstack((gtone.conj(),gtone))
        # Perform a one-pass correlation over the entire array
        xc = np.vstack([np.correlate(x, tone) for tone in tones])
        xc_abs = np.abs(xc)
        # Also perform the one-pass max over it
        xc_abs_argmax = np.argmax(xc_abs, axis=0) 
        xc_abs_max = np.max(xc_abs, axis=0) # Note: using the argmax to regenerate using list comprehension is extremely slow..
        
        # Construct the starting indices for each burst
        burstStarts = self.burstIdxs * self.period * self.up

        # Construct the symbol spacing for each individual burst
        symbolSpacing = np.arange(0, self.burstLen*self.up, self.up)
        # Construct zero-origin indices for every symbol, in every burst
        genIdx = np.array([start + symbolSpacing for start in burstStarts]).flatten()
        
        # Construct a search index if not specified
        if searchIdx is None:
            # extent = genIdx[-1] + self.up
            searchIdx = np.arange(xc_abs_max.size - genIdx[-1])
            print("Auto-generated search indices from %d to %d" % (searchIdx[0], searchIdx[-1]))
        
        # Loop over the search range
        self.d_costs = np.zeros(searchIdx.size)
        for i, s in enumerate(searchIdx):
            # Construct the shifted indices for every symbol based on current search index
            idx = s + genIdx
            # Sum the costs for these indices
            self.d_costs[i] = np.sum(xc_abs_max[idx])
            
        # Now find the best cost and extract the corresponding bits
        mi = searchIdx[np.argmax(self.d_costs)]
        dbits = xc_abs_argmax[mi+genIdx].reshape((-1,self.burstLen))
        
        # t2 = time.perf_counter()
        # print("Took %f s." % (t2-t1))
        
        return dbits, mi

        
    def setBurstIdxs(self, burstIdxs: np.ndarray=None):
        '''
        Parameters
        ----------
        burstIdxs : np.ndarray, optional
            Integer array of burst indices that should be demodulated.
            This can be used to ignore certain bursts, or if there are missing bursts
            within the window passed to the demod() call.
            The default is None, which will then fit as many bursts as possible during 
            the demod() method call.

        Returns
        -------
        None.

        '''
        self.burstIdxs = burstIdxs

#%%
def convertIntToBase4Combination(l, i):
    base_4_repr = np.array(list(np.base_repr(i,base=4)),dtype=np.uint8)
    base_4_repr = np.pad(base_4_repr, (l - len(base_4_repr),0)) # pad it to the numSyms
    
    return base_4_repr

def ML_demod_QPSK(y, h, up, numSyms):
    '''
    Brute force search over all combinations, don't use this if you have a long string of symbols.
    '''
    
    totalCombi = 4**numSyms
    cost = np.zeros(totalCombi)
    
    for i in range(totalCombi):
        base_4_repr = convertIntToBase4Combination(numSyms, i)

        qpsk_syms = np.exp(1j*base_4_repr*np.pi/2)
        
        qpsk_syms_up = np.zeros(numSyms * up, dtype=np.complex128)
        qpsk_syms_up[::up] = qpsk_syms
        
        # convolve this test symbol set with the channel
        test = np.convolve(h, qpsk_syms_up)
        
        # # find the normalised dot product
        # test = test[up : up + len(y)]
        # test_cost = np.vdot(y, test) / np.linalg.norm(y) / np.linalg.norm(test)
        # cost[i] = np.abs(test_cost)**2.0
        
        # find the smallest norm difference
        test = test[up : up + len(y)]
        test_cost = np.linalg.norm(test - y)
        cost[i] = -test_cost # minus to maintain the maximization criterion
    
        
        # print(base_4_repr)
        # print(qpsk_syms_up)
        # print(test)
        
        # if np.all(base_4_repr == np.array([0,1,2,1,2,3])):
        #     print(base_4_repr)
        #     print(test)
        #     print(qpsk_syms_up)
        #     break
        
    ii = np.argmax(cost)
    mm = convertIntToBase4Combination(numSyms, ii) 
    
    
    return mm, ii, cost
            

# def ML_demod_CPM_laurent(y, h, up, numSyms, outputCost=False):
#     '''
#     Brute force search over all combinations, don't use this if you have a long string of symbols.
#     '''
    
#     totalCombi = 2**numSyms
#     cost = np.zeros(totalCombi)
#     rowsToKeep = int(np.ceil(numSyms)/8)
    
#     for i in range(totalCombi):
#         i_arr = np.array([i], np.uint32) # convert to array
#         i_arr = i_arr.view(np.uint8).reshape((-1,1)) # convert to byte level
#         i_bits = np.unpackbits(i_arr, axis=1, bitorder='little')
        
#         i_bits = i_bits[:rowsToKeep+1].flatten()
#         i_bits = i_bits[0:numSyms]
        
#         print(i_bits)
        
#         for b in range(len(i_bits)):
    
if __name__ == "__main__":
    from signalCreationRoutines import *
    from plotRoutines import *
    import matplotlib.pyplot as plt
    from timingRoutines import Timer
    
    closeAllFigs()
    
    OSR = 8
    numBits = 10000
    m = 4
    syms, bits = randPSKsyms(numBits, m)
    syms_rs = sps.resample_poly(syms,OSR,1)
    _, rx = addSigToNoise(numBits * OSR, 0, syms_rs, chnBW=OSR, snr_inband_linear=1000) # Inf SNR simulates perfect filtering
    randomPhase = np.random.rand() * (2*np.pi/m) # Induce random phase
    rx = rx * np.exp(1j* randomPhase)
    ofig, oax = plt.subplots(2,1,num="Original")
    plotConstellation(syms_rs, ax=oax[0])
    plotConstellation(rx, ax=oax[0])
    plotSpectra([rx],[1],ax=oax[1])
    
    # Divide into resampled portions
    rxrs = rx.reshape((-1,OSR))
    fig, ax = plt.subplots(OSR//2, OSR//2)
    for i in range(OSR):
        plotConstellation(rxrs[:,i], ax=ax[i//2, i%2])
        
    
    # Projection to log2(m)+1 dimensions
    demodulator = SimpleDemodulatorPSK(m)

    # Most generic demodulator shouldn't use angle to avoid if/else costs
    # Use the normalised dot product from constellation, then find max across rows
    timer = Timer()
    timer.start()
    genericSyms = demodulator.demod(rx.astype(np.complex64), OSR)
    timer.end("Default demodulator")
    rotationalInvariance = (genericSyms - bits) % m
    rax = plotConstellation(demodulator.reimc)
    assert(np.all(rotationalInvariance == rotationalInvariance[0]))
    
    unpackedSyms = demodulator.unpackToBinaryBytes(genericSyms)
    demodbits = demodulator.symsToBits()
    
    # Test preamble detection
    preamble = bits[400:400+32]
    timer.start()
    rotatedSyms, sample, rotation = demodulator.ambleRotate(preamble, np.arange(400,400+256))
    timer.end("Preamble search")
    plt.figure("Preamble matching")
    plt.plot(demodulator.matches)
    
    
    # Unit test on pure QPSK
    if m == 4:
        pure4 = demodulator.pskdicts[4]
        
        rxScaled = rx.astype(np.complex64) * 3
        rxScaledAbs = np.abs(rxScaled) # Providing the amplitude pre-computed reduces computation significantly
        
        timer.reset()
        demod4 = SimpleDemodulatorQPSK() # Custom symbol mapper is present, about 3-4x
        timer.start()
        # demod4.mapSyms(pure4)
        # demod4.mapSyms(demodulator.reimc)
        genericSyms4 = demod4.demod(rxScaled, OSR, abs_x=rxScaledAbs)
        timer.end("QPSK demodulator")
        rotationalInvariance4 = (genericSyms4 - bits) % m
        plotConstellation(demod4.reimc, ax=rax)
        assert(np.all(rotationalInvariance4 == rotationalInvariance4[0]))
        
    
    # Unit test on pure 8PSK
    if m == 8:
        pure8 = demodulator.pskdicts[8]
        
        timer.reset()
        demod8 = SimpleDemodulator8PSK() # Custom symbol mapper is present, about 2-4x
        timer.start()
        # demod8.mapSyms(pure8)
        # demod8.mapSyms(demodulator.reimc)
        genericSyms8 = demod8.demod(rx.astype(np.complex64) * 3, OSR)
        timer.end("8PSK demodulator")
        rotationalInvariance8 = (genericSyms8 - bits) % m
        assert(np.all(rotationalInvariance8 == rotationalInvariance8[0]))
        
    
    
    
    
    
    
    
    #%%
    # baud = 10000
    # up = 10
    # fs = baud * up
    
    # numBursts = 99
    # burstBits = 90
    # period = 100
    # guardBits = period - burstBits
    
    # bits = randBits(burstBits * numBursts, 2).reshape((numBursts,burstBits))
    # sigs = []
    # for i in range(numBursts):
    #     sig, _, _, _ = makePulsedCPFSKsyms(bits[i,:], baud, up=up)
    #     sigs.append(sig)
        
    # sigStart = int(0.25*sigs[0].size)
    # snr = 10
    # _, rx = addManySigToNoise(int((numBursts+1)*period*up),
    #                           np.arange(0,numBursts*period*up,period*up)+sigStart, sigs, bw_signal = baud, chnBW = fs,
    #                    snr_inband_linearList = snr+np.zeros(numBursts))
    
    # fig, ax = plt.subplots(2,1)
    # ax[0].plot(np.abs(rx))
    # plotSpectra([rx],[fs],ax=ax[1])
    
    # # Filter
    # taps = sps.firwin(201, baud/fs*1.5)
    # rxfilt = sps.lfilter(taps,1,rx)
    # ax[0].plot(np.abs(rxfilt))
    # plotSpectra([rxfilt],[fs],ax=ax[1])
    
    # # Attempt bursty demod
    # searchRange = np.arange(int(0.5*sigs[0].size))
    
    # bd = BurstyDemodulatorCP2FSK(burstBits, (period-burstBits), up)
    # dbits, dalign = bd.demod(rxfilt, numBursts, searchRange)
    # print("Demodulation index at %d" % dalign)
    # plt.figure("Bursty demod cost")
    # plt.plot(searchRange, bd.dcosts)
    
    # # # This is the exact actual value
    # # dalign = 327
    # # dbits = bd.demodAtIdx(rxfilt, dalign, numBursts, numBursts * period * up - guardBits * up) # Hard coded correct alignment
    
    # if np.all(dbits==bits):
    #     print("Full demodulation of %d bursts * %d symbols is correct." % (numBursts, burstBits))
        
    # else:
    #     print("Full demodulation failed for these bursts:")
    #     failedBursts = np.argwhere(np.any(bits != dbits, axis=1)).flatten()
    #     for i in range(failedBursts.size):
    #         fb = failedBursts[i]
    #         rm, _, _, _ = makePulsedCPFSKsyms(dbits[fb,:], baud, up=up)
    #         rm = rm[:burstBits * up] # cut it
    #         balign = dalign + fb * period * up
    #         metric = np.abs(np.vdot(rm, rxfilt[balign:balign+rm.size]))**2 / np.linalg.norm(rm)**2 / np.linalg.norm(rxfilt[balign:balign+rm.size])**2
    #         print("Burst %d with QF2 %f, bits %d/%d" % (fb, metric, np.argwhere(bits[fb,:] == dbits[fb,:]).size, burstBits))      
    
    #     print("Full demodulation successful for these bursts:")
    #     successBursts = np.argwhere(np.all(bits == dbits, axis=1)).flatten()
    #     for i in range(successBursts.size):
    #         sb = successBursts[i]
    #         rm, _, _, _ = makePulsedCPFSKsyms(dbits[sb,:], baud, up=up)
    #         rm = rm[:burstBits * up] # cut it
    #         balign = dalign + sb * period * up
    #         metric = np.abs(np.vdot(rm, rxfilt[balign:balign+rm.size]))**2 / np.linalg.norm(rm)**2 / np.linalg.norm(rxfilt[balign:balign+rm.size])**2
    #         print("Burst %d with QF2 %f, bits %d/%d" % (sb, metric, np.argwhere(bits[sb,:] == dbits[sb,:]).size, burstBits))      
    