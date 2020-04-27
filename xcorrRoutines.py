# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 17:03:53 2020

@author: Seo
"""

import numpy as np
import scipy as sp

def fastXcorr(cutout, rx, freqsearch=False, outputCAF=False, shifts=None):
    """ Optional frequency scanning xcorr."""
    if shifts is None:
        shifts = np.arange(len(rx)-len(cutout)+1)
    
    if not freqsearch:
        print('No frequency scanning xcorr..')
        result = np.zeros(len(shifts),dtype=np.float64)
        cutoutNormSq = np.linalg.norm(cutout)**2.0
        for i in range(len(shifts)):
            s = shifts[i]
            result[i] = sp.absolute(np.vdot(rx[s:s+len(cutout)], cutout))**2.0 # vdot already takes conj of first arg
            rxNormPartSq = np.linalg.norm(rx[s:s+len(cutout)])**2.0
            result[i] = result[i]/cutoutNormSq/rxNormPartSq
            
        return result
    
    elif not outputCAF:
        print('Frequency scanning, but no CAF output (flattened to time)..')
        result = np.zeros(len(shifts),dtype=np.float64)
        freqlist = np.zeros(len(shifts),dtype=np.uint32)
        cutoutNormSq = np.linalg.norm(cutout)**2.0
        for i in range(len(shifts)):
            s = shifts[i]
            pdt = rx[s:s+len(cutout)] * cutout.conj()
            pdtfft = sp.fft(pdt)
            pdtfftsq = pdtfft**2.0
            imax = np.argmax(np.abs(pdtfftsq))
            freqlist[i] = imax
            pmax = np.abs(pdtfftsq[imax])

            rxNormPartSq = np.linalg.norm(rx[s:s+len(cutout)])**2.0
            result[i] = pmax/cutoutNormSq/rxNormPartSq
            
        return result, freqlist
    
    else:
        print('Frequency scanning, outputting raw CAF...')
        result = np.zeros((len(shifts), len(cutout)), dtype=np.float64)
        cutoutNormSq = np.linalg.norm(cutout)**2.0
        for i in range(len(shifts)):
            s = shifts[i]
            pdt = rx[s:s+len(cutout)] * cutout.conj()
            pdtfft = sp.fft(pdt)
            pdtfftsq = np.abs(pdtfft**2.0)

            rxNormPartSq = np.linalg.norm(rx[s:s+len(cutout)])**2.0
            result[i] = pdtfftsq/cutoutNormSq/rxNormPartSq

        return result
    

def fineFreqTimeSearch(x_aligned, y_aligned, fineRes, freqfound, freqRes, fs, td_scan_range, steeringvec=None):
    '''
    Performs a finer search to align frequency and time, in two separate processes.
    As such, this function may not result in the global minimum for TDOA/FDOA in 2-D, but will perform much
    faster, since it only searches one index in both dimensions, rather than the full 2-D space.
    
    The best fine frequency will be searched first (assumes that the sample-aligned arrays are correctly time-aligned).
    Then, using the new value for the fine frequency alignment, the two arrays are then sub-sample time aligned.
    '''
    
    # compute best freq
    for i in range(len(fineRes)):
        fineFreq = np.arange(freqfound-freqRes, freqfound+freqRes, fineRes)
        
        precomputed = y_aligned.conj() * x_aligned
        pp = np.zeros(len(fineFreq))
        fineshifts = np.zeros(len(fineFreq), len(x_aligned))
        for j in range(len(fineFreq)):
            fineshifts[j] = np.exp(1j*2*np.pi*-fineFreq[j]*np.arange(len(x_aligned))/fs)
        
            pp[j] = np.vdot(precomputed,fineshifts[j])
        
        fineFreq_ind = np.argmax(np.abs(pp))
        freqfound = fineFreq[fineFreq_ind]
        
    finefreqfound = freqfound
        
    # compute best time alignment
    if steeringvec is None:
        steeringvec = makeTimeScanSteervec(td_scan_range)
    
    x_aligned = x_aligned * fineshifts[fineFreq_ind]
    
    x_fft = np.fft.fft(x_aligned)
    y_fft = np.fft.fft(y_aligned)
    rx_vec = x_fft * y_fft.conj()
    
    cost_vec = np.dot(rx_vec, steeringvec.conj().T)/np.linalg.norm(x_fft)/np.linalg.norm(y_fft)
    idx_td = np.argmax(np.abs(cost_vec))
    timediff = td_scan_range[idx_td]
    
    return finefreqfound, timediff
    
def makeTimeScanSteervec(td_scan_range):
    print('not yet implemented')
    return 1

def convertQF2toSNR(qf2):
    """For xcorr against pure signal."""
    return qf2/(1.0-qf2)

def convertQF2toEffSNR(qf2):
    """For xcorr of two noisy signals."""
    return 2.0*qf2/(1.0-qf2)

def expectedEffSNR(snr1, snr2=None):
    """For calculating expected SNR of two noisy signals."""
    if snr2 is None:
        snr2 = snr1
    
    y = 1.0/(0.5 * (1/snr1 + 1/snr2 + 1/snr1/snr2))
    return y

def sigmaDTO(signalBW, noiseBW, integTime, effSNR):
    '''
    Taken from Algorithms for Ambiguity Function Processing. SEYMOUR STEIN. 
    '''
    beta = np.pi / np.sqrt(3) * signalBW
    s = 1.0/beta / np.sqrt(noiseBW * integTime * effSNR)
    return s

def theoreticalMultiPeak_SampleLevel(startIdx1, startIdx2):
    mat = np.zeros((len(startIdx1), len(startIdx1))) # expected same length anyway
    
    for i in range(len(mat)):
        mat[i] = startIdx2[i] - startIdx1
        
    mat = mat.flatten()
    
    return np.unique(mat)