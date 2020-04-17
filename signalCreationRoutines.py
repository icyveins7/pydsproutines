# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:18:01 2020

@author: Seo
"""

import numpy as np

def randPSKbits(length, m, dtype=np.uint8):
    return np.random.randint(0,m,length,dtype)

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

def makeFreq(length, fs):
    freq = np.zeros(length)
    for i in range(length):
        freq[i] = i/length * fs
        if freq[i] >= fs/2:
            freq[i] = freq[i] - fs
    return freq