# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 17:03:53 2020

@author: Seo
"""

import numpy as np
import scipy as sp
import cupy as cp
from signalCreationRoutines import makeFreq

GPU_RAM_LIM_BYTES = 6e9 # use to roughly judge if it will fit


def cp_fastXcorr(cutout, rx, freqsearch=True, outputCAF=False, shifts=None, absResult=True, BATCH=1024):
    """
    Equivalent to fastXcorr, designed to run on gpu.
    """
    
    # need both to be same type, maintain the type throughout
    if cutout.dtype is not rx.dtype:
        raise Exception("Cutout and Rx must be same type, please cast one of them manually.")
    
    if shifts is None:
        shifts = np.arange(len(rx)-len(cutout)+1)
    
    # common numbers in all consequent methods
    cutoutNorm = np.linalg.norm(cutout)
    cutoutNormSq = cutoutNorm**2.0
    
    if not freqsearch:
        print('Not implemented.')
        
    elif not outputCAF:
        
        if absResult is True:
            print('Frequency scanning, but no CAF output (flattened to time)..')
            # h_freqlist = np.zeros(len(shifts),dtype=np.uint32)
            d_freqlist = cp.zeros(len(shifts),dtype=cp.uint32)
        
        
            print('Returning normalized QF^2 real values..')
            # h_result = np.zeros(len(shifts),dtype=np.float64)
            d_result = cp.zeros(len(shifts),dtype=cp.float64)

            # first copy the data in
            d_cutout = cp.asarray(cutout)
            d_rx = cp.asarray(rx)
            
            numIter = int(np.ceil(len(shifts)/BATCH))
            
            # allocate data arrays on gpu
            d_pdt_batch = cp.zeros((BATCH,len(cutout)), dtype=cp.complex128) # let's try using it in place all the way
            d_rxNormPartSq_batch = cp.zeros((BATCH), dtype=cp.float64)
            
            # now iterate over the number of iterations required
            print("Starting cupy loop")
            for i in range(numIter):
                if i == numIter-1: # on the last iteration, may have to clip
                    # print("FINAL BATCH")
                    TOTAL_THIS_BATCH = len(shifts) - BATCH * (numIter-1)
                else:
                    TOTAL_THIS_BATCH = BATCH
                # print(i)
                # print (TOTAL_THIS_BATCH)
                    
                for k in range(TOTAL_THIS_BATCH):
                    s = shifts[i*BATCH + k]
                    
                    d_pdt_batch[k] = d_rx[s:s+len(cutout)] * d_cutout.conj()
                    
                    d_rxNormPartSq_batch[k] = cp.linalg.norm(d_rx[s:s+len(cutout)])**2.0
                    
                # perform the fft (row-wise is done automatically)
                d_pdtfft_batch = cp.fft.fft(d_pdt_batch)
                d_pdtfft_batch = cp.abs(d_pdtfft_batch**2.0) # is now abs(pdtfftsq)
                
                imax = cp.argmax(d_pdtfft_batch, axis=-1) # take the arg max for each row
                
                # assign to d_freqlist output by slice
                d_freqlist[i*BATCH : i*BATCH + TOTAL_THIS_BATCH] = imax[:TOTAL_THIS_BATCH]
                
                
                # assign to d_result
                for k in range(TOTAL_THIS_BATCH):    
                    d_result[i*BATCH + k] = d_pdtfft_batch[k, imax[k]] / d_rxNormPartSq_batch[k] / cutoutNormSq
                    
            
            
            # copy all results back
            h_result = cp.asnumpy(d_result)
            h_freqlist = cp.asnumpy(d_freqlist)
                
            return h_result, h_freqlist
        
        else:
            
            print('Not implemented.')
            
    else:
        print('Not implemented.')
        
        


def fastXcorr(cutout, rx, freqsearch=False, outputCAF=False, shifts=None, absResult=True):
    """
    Optional frequency scanning xcorr.
    
    When absResult is set to False, the result is not absoluted, and is also not given as a 
    QF^2 value. It is left as a complex value, scaled only by the norm (not norm-squared!) 
    of the two corresponding array slices.
    
    Consequently, when absResult is True (default), then the returned values are QF^2 
    normalized values.
    """
    if shifts is None:
        shifts = np.arange(len(rx)-len(cutout)+1)
    
    # common numbers in all consequent methods
    cutoutNorm = np.linalg.norm(cutout)
    cutoutNormSq = cutoutNorm**2.0
    
    if not freqsearch:
        print('No frequency scanning xcorr..')
    
        if absResult is True:
            print('Returning normalized QF^2 real values..')
            result = np.zeros(len(shifts),dtype=np.float64)

            for i in range(len(shifts)):
                s = shifts[i]
                result[i] = sp.absolute(np.vdot(rx[s:s+len(cutout)], cutout))**2.0 # vdot already takes conj of first arg
                rxNormPartSq = np.linalg.norm(rx[s:s+len(cutout)])**2.0
                result[i] = result[i]/cutoutNormSq/rxNormPartSq
                
            return result
        
        else:
            print('Returning normalized QF complex values..')
            result = np.zeros(len(shifts),dtype=np.complex128)
            
            for i in range(len(shifts)):
                s = shifts[i]
                result[i] = np.vdot(rx[s:s+len(cutout)], cutout) # vdot already takes conj of first arg
                rxNormPart = np.linalg.norm(rx[s:s+len(cutout)])
                result[i] = result[i]/cutoutNorm/rxNormPart
                
            return result
    
    elif not outputCAF:
        print('Frequency scanning, but no CAF output (flattened to time)..')
        freqlist = np.zeros(len(shifts),dtype=np.uint32)
        
        if absResult is True:
            print('Returning normalized QF^2 real values..')
            result = np.zeros(len(shifts),dtype=np.float64)

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
            print('Returning normalized QF complex values..')
            result = np.zeros(len(shifts),dtype=np.complex128)
            
            for i in range(len(shifts)):
                s = shifts[i]
                pdt = rx[s:s+len(cutout)] * cutout.conj()
                pdtfft = sp.fft(pdt)
                imax = np.argmax(np.abs(pdtfft))
                freqlist[i] = imax
                pmax = pdtfft[imax]
    
                rxNormPart = np.linalg.norm(rx[s:s+len(cutout)])
                result[i] = pmax/cutoutNorm/rxNormPart
                
            return result, freqlist
    
    else:
        print('Frequency scanning, outputting raw CAF...')
        
        if absResult is True:
            print('Returning normalized QF^2 real values..')
            result = np.zeros((len(shifts), len(cutout)), dtype=np.float64)

            for i in range(len(shifts)):
                s = shifts[i]
                pdt = rx[s:s+len(cutout)] * cutout.conj()
                pdtfft = sp.fft(pdt)
                pdtfftsq = np.abs(pdtfft**2.0)
    
                rxNormPartSq = np.linalg.norm(rx[s:s+len(cutout)])**2.0
                result[i] = pdtfftsq/cutoutNormSq/rxNormPartSq
    
            return result
        
        else:
            print('Returning normalized QF complex values..')
            result = np.zeros((len(shifts), len(cutout)), dtype=np.complex128)
            
            for i in range(len(shifts)):
                s = shifts[i]
                pdt = rx[s:s+len(cutout)] * cutout.conj()
                pdtfft = np.fft.fft(pdt)
    
                rxNormPart = np.linalg.norm(rx[s:s+len(cutout)])
                result[i] = pdtfft/cutoutNorm/rxNormPart
    
            return result
    

def fineFreqTimeSearch(x_aligned, y_aligned, fineRes, freqfound, freqRes, fs, td_scan_range, steeringvec=None):
    '''
    Performs a finer search to align frequency and time, in two separate processes.
    As such, this function may not result in the global minimum for TDOA/FDOA in 2-D, but will perform much
    faster, since it only searches one index in both dimensions, rather than the full 2-D space.
    
    The best fine frequency will be searched first (assumes that the sample-aligned arrays are correctly time-aligned).
    Then, using the new value for the fine frequency alignment, the two arrays are then sub-sample time aligned.
    '''
    
    if len(fineRes) > 0:
        # compute best freq
        for i in range(len(fineRes)):
            fineFreq = np.arange(freqfound-freqRes, freqfound+freqRes, fineRes[i])
            
            precomputed = y_aligned.conj() * x_aligned
            pp = np.zeros(len(fineFreq), dtype=np.complex128)
            fineshifts = np.zeros((len(fineFreq), len(x_aligned)), dtype=np.complex128)
            for j in range(len(fineFreq)):
                fineshifts[j] = np.exp(1j*2*np.pi*-fineFreq[j]*np.arange(len(x_aligned))/fs)
            
                pp[j] = np.vdot(precomputed,fineshifts[j])
            
            fineFreq_ind = np.argmax(np.abs(pp))
            freqfound = fineFreq[fineFreq_ind]
            
        finefreqfound = freqfound
        
        x_aligned = x_aligned * fineshifts[fineFreq_ind]
    else:
        finefreqfound = None
        
    # compute best time alignment
    if steeringvec is None:
        steeringvec = makeTimeScanSteervec(td_scan_range, fs, len(x_aligned))

    x_fft = np.fft.fft(x_aligned)
    y_fft = np.fft.fft(y_aligned)
    rx_vec = x_fft * y_fft.conj()
    
    cost_vec = np.dot(rx_vec, steeringvec.conj().T)/np.linalg.norm(x_fft)/np.linalg.norm(y_fft)
    idx_td = np.argmax(np.abs(cost_vec))
    timediff = td_scan_range[idx_td]
    
    return finefreqfound, timediff, cost_vec
    
def makeTimeScanSteervec(td_scan_range, fs, siglen):
    sigFreq = makeFreq(siglen, fs)
    
    mat = np.exp(1j*2*np.pi*sigFreq * td_scan_range.reshape((-1,1)))
    
    return mat

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

def theoreticalMultiPeak(startIdx1, startIdx2, snr_linear_1=None, snr_linear_2=None):
    '''
    Calculates parameters resulting from cross-correlation of multiple copies of a signal in two receivers.
    Works with both indices and floating-point, but of course floating-point may result in 'unique' values being repeated.
    '''
    
    # if no snr supplied, just return the indices
    if snr_linear_1 is None and snr_linear_2 is None:
        mat = np.zeros((len(startIdx1), len(startIdx1))) # expected same length anyway
        
        for i in range(len(mat)):
            mat[i] = startIdx2[i] - startIdx1
            
        mat = mat.flatten()
        
        return np.unique(mat)
    else:
        mat = np.zeros((len(startIdx1), len(startIdx1))) # expected same length anyway
        matEffSNR = np.zeros((len(startIdx1), len(startIdx1))) # expected same length anyway
        
        for i in range(len(mat)):
            mat[i] = startIdx2[i] - startIdx1
            
            tmp = 0.5 * ( (1/snr_linear_1) + (1/snr_linear_2[i]) + (1/snr_linear_1/snr_linear_2[i]) )
            
            matEffSNR[i] = 1/tmp
            
        mat = mat.flatten()
        matEffSNR = matEffSNR.flatten()
        
        u,indices = np.unique(mat, return_index=True)
        return u, matEffSNR[indices]
        
        
        
        