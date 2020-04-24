# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 15:18:01 2020

@author: Seo
"""

import numpy as np
import scipy as sp
import scipy.signal as sps

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

def addSigToNoise(noiseLen, sigStartIdx, signal, bw_signal, chnBW, snr_inband_linear, sigPwr = 1.0, fshift = None):
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
    
    if fshift is not None:
        tone = np.exp(1j*2*np.pi*fshift*np.arange(noiseLen)/chnBW)
        rx = rx * tone
        
        return noise, rx, tone
    else:
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
    data = bits.astype(np.int8) * m - 1
    
    theta = np.zeros(len(bits) * up)
    
    # numpy version
    i_list = np.floor(np.arange(len(theta)) / up).astype(np.uint32)
    t_list = np.arange(len(theta)) / fs
    a_list = np.hstack(([0], np.cumsum(data)))[:len(data)] # accumulator of phase
    a_list = np.repeat(a_list, up) 
    
    theta = ( data[i_list] * np.pi * h * (t_list - i_list * T ) / T ) + np.pi * h * a_list + phase
    
    sig = np.exp(1j*theta)
    
    return sig, fs, data

def makePulsedCPFSKsyms(bits, baud, g=np.ones(8)/16, m=2, h=0.5, up=8, phase=0.0):
    '''
    Uses the pulse shape g to create the signal.
    g is applied to the phase (in a convolutional way) before the actual symbols are created.
    This is in contrast to PSK where the pulse shape is applied to the symbols.
	With the default settings (pulse shape constant over one symbol), this should result in the same array
	as the non-pulsed function. The pulse shape is by default normalised to have integral 0.5 over the one symbol.
    
    In particular, the pulse shape function g is expected to already be scaled by the upsampling rate.
    In other words, the assumption within this function is that the calculations are done with a normalized sampling rate.
    
    Note: this function will return the full convolution. It is up to the user to
    define where the end or start index is depending on the pulse shape centre, and the
    corresponding delay introduced. In the default rect-pulse, there is no clear 'delay', so the
    output can start from index 0 (as it is in the non pulsed function), and the ending len(g) indices
    can be trimmed off.
    '''
    T = 1.0/baud # symbol period
    fs = baud * up # sample period
    data = bits.astype(np.int8) * m - 1
    
    theta = np.zeros(len(bits) * up + 1)
    
    # first create the upsampled version of the data (pad zeros between elements)
    theta[1::up] = data # we need the zero at the start (before accumulation)
    
    # then convolve it with the pulse
    c = sps.convolve(theta, g)
    
    # and now accumulate the phase (from a starting idx)
    # note, this assumes a normalized sampling rate (fs = 1 Hz)! 
    # likewise this assumes that the pulse shape has accounted for this value
    cs = np.cumsum(c)
    
    # scale by 2 pi h and initial phase
    css = cs * 2 * np.pi * h + phase

    sig = np.exp(1j*css)
    
    return sig, fs, data
    
    

@jit(nopython=True)
def makeFreq(length, fs):
    freq = np.zeros(length)
    for i in range(length):
        freq[i] = i/length * fs
        if freq[i] >= fs/2:
            freq[i] = freq[i] - fs
    return freq