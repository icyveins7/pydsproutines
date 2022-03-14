# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:21:35 2020

@author: Seo
"""

import numpy as np
from numba import jit

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
    def __init__(self, burstLen: int, guardLen: int):
        self.burstLen = burstLen
        self.guardLen = guardLen
        self.period = self.burstLen + self.guardLen
        
    def demod(self, x: np.ndarray, numBursts: int, searchIdx: np.ndarray=None):
        raise NotImplementedError("Only invoke with derived classes.")
        
##
class BurstyDemodulatorCP2FSK(BurstyDemodulator):
    def __init__(self, burstLen: int, guardLen: int, up: int, h: float=0.5):
        super().__init__(burstLen, guardLen) # Refer to parent class
        # Extra params
        self.up = up
        self.h = h
        
        # Configurations
        self.burstIdxs = None
        
        # Outputs
        self.dcosts = None
        
        
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
    
    def demod(self, x: np.ndarray, numBursts: int, searchIdx: np.ndarray=None):
        duration = numBursts * self.burstLen + (numBursts-1) * self.guardLen
        
        # Set a default searchIdx over entire duration
        if searchIdx is None:
            searchIdx = np.arange(x.size - duration + 1)
            
        # Check for pre-configured burst indices
        if self.burstIdxs is None:
            burstShifts = np.arange(0,duration, self.period)
        else:
            burstShifts = self.burstIdxs * self.period
            
        # Main loop
        self.dcosts = np.zeros(searchIdx.size)
        for i in range(searchIdx.size):
            s = searchIdx[i]
            bursts = np.array([x[sb:sb+self.burstLen] for sb in 
                               (burstShifts + s)]) # Carve out the aligned bursts

            for b in np.arange(numBursts):
                # print("Search %d/%d, burst %d/%d" % (i,searchIdx.size,b,numBursts))
                
                xs = bursts[b,:]
                demodBits, cost, _ = demodulateCP2FSK(xs, self.h, self.up)
                # print(cost)
                
                self.dcosts[i] += np.sum(np.max(cost, axis=0))
        
        mi = np.argmax(self.dcosts)
        
        dbits = self.demodAtIdx(x, mi, numBursts, duration)
        
        return dbits, searchIdx[mi]
    
    def demodAtIdx(self, x, idx, numBursts, duration):
        bbursts = np.array([x[sb:sb+self.burstLen] for sb in np.arange(idx, idx+duration, self.period)]) # Carve out the best bursts
        dbits = np.zeros((numBursts, int(self.burstLen/self.up)), dtype=np.uint8)
        
        for b in np.arange(numBursts):
            xs = bbursts[b,:]
            dbits[b,:], cost, _ = demodulateCP2FSK(xs, self.h, self.up)
            
        return dbits


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
    
    plt.close('all')
    
    baud = 10000
    up = 10
    fs = baud * up
    
    numBursts = 20
    burstBits = 90
    period = 100
    guardBits = period - burstBits
    
    bits = randBits(burstBits * numBursts, 2).reshape((numBursts,burstBits))
    sigs = []
    for i in range(numBursts):
        sig, _, _, _ = makePulsedCPFSKsyms(bits[i,:], baud, up=up)
        sigs.append(sig)
        
    sigStart = int(0.25*sigs[0].size)
    snr = 10
    _, rx = addManySigToNoise(int((numBursts+1)*period*up),
                              np.arange(0,numBursts*period*up,period*up)+sigStart, sigs, bw_signal = baud, chnBW = fs,
                       snr_inband_linearList = snr+np.zeros(numBursts))
    
    fig, ax = plt.subplots(2,1)
    ax[0].plot(np.abs(rx))
    plotSpectra([rx],[fs],ax=ax[1])
    
    # Filter
    taps = sps.firwin(201, baud/fs*1.5)
    rxfilt = sps.lfilter(taps,1,rx)
    ax[0].plot(np.abs(rxfilt))
    plotSpectra([rxfilt],[fs],ax=ax[1])
    
    # Attempt bursty demod
    searchRange = np.arange(int(0.5*sigs[0].size))
    
    bd = BurstyDemodulatorCP2FSK(burstBits*up, (period-burstBits)*up, up)
    dbits, dalign = bd.demod(rxfilt, numBursts, searchRange)
    print("Demodulation index at %d" % dalign)
    plt.figure("Bursty demod cost")
    plt.plot(searchRange, bd.dcosts)
    
    # # This is the exact actual value
    # dalign = 327
    # dbits = bd.demodAtIdx(rxfilt, dalign, numBursts, numBursts * period * up - guardBits * up) # Hard coded correct alignment
    
    if np.all(dbits==bits):
        print("Full demodulation of %d bursts * %d symbols is correct." % (numBursts, burstBits))
        
    else:
        print("Full demodulation failed for these bursts:")
        failedBursts = np.argwhere(np.any(bits != dbits, axis=1)).flatten()
        for i in range(failedBursts.size):
            fb = failedBursts[i]
            rm, _, _, _ = makePulsedCPFSKsyms(dbits[fb,:], baud, up=up)
            rm = rm[:burstBits * up] # cut it
            balign = dalign + fb * period * up
            metric = np.abs(np.vdot(rm, rxfilt[balign:balign+rm.size]))**2 / np.linalg.norm(rm)**2 / np.linalg.norm(rxfilt[balign:balign+rm.size])**2
            print("Burst %d with QF2 %f, bits %d/%d" % (fb, metric, np.argwhere(bits[fb,:] == dbits[fb,:]).size, burstBits))      
    
        print("Full demodulation successful for these bursts:")
        successBursts = np.argwhere(np.all(bits == dbits, axis=1)).flatten()
        for i in range(successBursts.size):
            sb = successBursts[i]
            rm, _, _, _ = makePulsedCPFSKsyms(dbits[sb,:], baud, up=up)
            rm = rm[:burstBits * up] # cut it
            balign = dalign + sb * period * up
            metric = np.abs(np.vdot(rm, rxfilt[balign:balign+rm.size]))**2 / np.linalg.norm(rm)**2 / np.linalg.norm(rxfilt[balign:balign+rm.size])**2
            print("Burst %d with QF2 %f, bits %d/%d" % (sb, metric, np.argwhere(bits[sb,:] == dbits[sb,:]).size, burstBits))      
    
    
    # demodBits = np.zeros((searchRange.size, burstBits))
    # costs = np.zeros(searchRange.size)
    # for i in range(searchRange.size):
    #     s = searchRange[i]
    #     demodBits[i,:], cost, _ = demodulateCP2FSK(rxfilt[s:s+burstBits*up], 0.5, up, 0)
    #     costs[i] = np.sum(np.max(cost,axis=0))
        
    # dfig, dax = plt.subplots(1,1)
    # dax.plot(costs)
    
    # bc = np.argmax(costs)
    # bbits = demodBits[bc,:]
    # print("Bits %d/%d" % (len(np.argwhere(bbits==bits)), bits.size))
    # print("Demodded at index %d" % bc)    
    
    # plt.figure("Bits")
    # plt.plot(bits)
    # plt.plot(bbits)
    
    # if len(np.argwhere(bbits==bits)) != bits.size:
    #     # Attempt re-demod at the exact place to demonstrate
    #     print("Bits %d/%d at known alignment." % (len(np.argwhere(demodBits[351,:]==bits)), bits.size))
    