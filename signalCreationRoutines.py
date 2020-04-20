# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:18:01 2020

@author: Seo
"""

import numpy as np

from numba import jit
# jit not used if not supported like in randint, or just slower..
# usually requires loops to be present for some benefit to be seen

def randBits(length, m):
    return np.random.randint(0,m,length,dtype=np.uint8)

def randPSKsyms(length, m, dtype=np.complex128):
    bits = randPSKbits(length, m).astype(dtype)
    return np.exp(1j*2*np.pi*bits/m), bits

def randnoise(length, bw_signal, chnBW, snr_inband_linear, sigPwr = 1.0):
    basicnoise = (np.random.randn(length) + 1j*np.random.randn(length))/np.sqrt(2) * np.sqrt(sigPwr)
    noise = basicnoise * np.sqrt(1.0/snr_inband_linear) * np.sqrt(chnBW/bw_signal) # pretty sure this is correct now..
    return noise

def addSigToNoise(noiseLen, sigStartIdx, signal, bw_signal, chnBW, snr_inband_linear, sigPwr = 1.0):
    '''Add signal into noisy background at particular index'''
    noise = randnoise(noiseLen, bw_signal, chnBW, snr_inband_linear, sigPwr)
    aveNoisePwr = np.linalg.norm(noise)**2.0/len(noise)
    print('Ave noise power = ' + str(aveNoisePwr))
    aveSigPwr = np.linalg.norm(signal)**2.0/len(signal)
    print('Ave sig power = '+str(aveSigPwr))
    expectedNoisePwr = (1.0/snr_inband_linear) * chnBW/bw_signal
    print('Expected noise power = '+str(expectedNoisePwr))
    rx = np.zeros(noiseLen,dtype=np.complex128)
    rx[sigStartIdx:len(signal)+sigStartIdx] = signal
    rx = rx+noise
    return noise, rx

def makeCPFSKsyms(bits, baud, m=2, h=0.5, up=8, phase=0.0):
    '''
    Chose the same defaults as the comms toolbox in Matlab.
    Bits are expressed in 1s and 0s. Digital data (e.g. +/-1) is converted
    within the function itself.
    
    Note: this function assumes a rectangular pulse of amplitude 1/2T, length T.
    It has no 'pulse memory' of any other symbols. This is different from the idea
    of the accumulator, which accumulates the 'pulsed phase' value from all prior symbols.
    '''
    T = 1.0/baud;
    fs = baud * up
    data = bits * m - 1
    
    theta = np.zeros(len(bits) * up)
    
    # numpy version
    i_list = np.floor(np.arange(len(theta)) / up).astype(np.uint32)
    t_list = np.arange(len(theta)) / fs
    a_list = np.hstack(([0], np.cumsum(data)))[:len(data)] # accumulator of phase
    a_list = np.repeat(a_list, up) 
    
    theta = ( data[i_list] * np.pi * h * (t_list - i_list * T ) / T ) + np.pi * h * a_list + phase
    
    sig = np.exp(1j*theta)
    
    return sig, fs, theta, data


    

@jit(nopython=True)
def makeFreq(length, fs):
    freq = np.zeros(length)
    for i in range(length):
        freq[i] = i/length * fs
        if freq[i] >= fs/2:
            freq[i] = freq[i] - fs
    return freq