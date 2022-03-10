# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:21:35 2020

@author: Seo
"""

import numpy as np
from numba import jit

@jit(nopython=True)
def demodulateCP2FSK(syms, h, up, sIdx):
    m = np.array([[-1],
                  [+1]]) # these map to [0, 1] bits
    
    # create the two tones
    tones = np.exp(1j*np.pi*h*np.arange(up)/up * m)
    
    # loop over each upsampled section of a symbol
    numSyms = int(np.floor((len(syms)-sIdx) / up))
    bitCost = np.zeros((2,numSyms))
    demodBits = np.zeros(numSyms, dtype = np.uint8)
    
    for i in range(numSyms):
        symbol = syms[sIdx + i*up : sIdx + (i+1)*up]
        
        for k in range(2):
            bitCost[k,i] = np.abs(np.vdot(symbol, tones[k]))
        
        demodBits[i] = np.argmax(bitCost[:,i])
        
    return demodBits, bitCost, tones


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
    
    bits = randBits(100, 2)
    sig, _, data, theta = makePulsedCPFSKsyms(bits, baud, up=up)
    
    _, rx = addSigToNoise(int(sig.size*1.5), int(0.25*sig.size), sig, bw_signal = baud, chnBW = fs,
                       snr_inband_linear = 10)
    
    fig, ax = plt.subplots(2,1)
    ax[0].plot(np.abs(rx))
    plotSpectra([rx],[fs],ax=ax[1])
    
    # Filter
    taps = sps.firwin(200, 0.1)
    rxfilt = sps.lfilter(taps,1,rx)
    ax[0].plot(np.abs(rxfilt))
    plotSpectra([rxfilt],[fs],ax=ax[1])
    
    # Attempt demod
    searchRange = np.arange(int(0.5*sig.size))
    demodBits = np.zeros((searchRange.size, 100))
    costs = np.zeros(searchRange.size)
    for i in range(searchRange.size):
        s = searchRange[i]
        demodBits[i,:], cost, _ = demodulateCP2FSK(rxfilt[s:s+100*up], 0.5, up, 0)
        costs[i] = np.sum(np.max(cost,axis=0))
        
    dfig, dax = plt.subplots(1,1)
    dax.plot(costs)
    
    bc = np.argmax(costs)
    bbits = demodBits[bc,:]
    print("Bits %d/%d" % (len(np.argwhere(bbits==bits)), len(bits)))
    print("Demodded at index %d" % bc)    
    
    plt.figure("Bits")
    plt.plot(bits)
    plt.plot(bbits)
    