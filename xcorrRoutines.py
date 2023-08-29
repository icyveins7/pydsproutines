# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 17:03:53 2020

@author: Seo
"""

import numpy as np
import scipy as sp
import scipy.signal as sps
import time
from spectralRoutines import czt, CZTCached
from signalCreationRoutines import makeFreq
from musicRoutines import MUSIC
# from numba import jit
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3 as sq
import concurrent.futures
from tqdm import tqdm

try:
    import cupy as cp
    from cupyExtensions import *
    
    def cp_fastXcorr(cutout, rx, freqsearch=True, outputCAF=False, shifts=None, absResult=True, BATCH=1024):
        """
        Equivalent to fastXcorr, designed to run on gpu.
        """
        # Query the ram of the device, but we need extra space for the ffts to work, so use 1/4 to estimate
        if cutout.nbytes * BATCH > cp.cuda.device.Device().mem_info[1] / 4: 
            raise MemoryError("Not enough memory with this batch. Try lowering the batch value.")
        
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
                # print('Frequency scanning, but no CAF output (flattened to time)..')
                # h_freqlist = np.zeros(len(shifts),dtype=np.uint32)
                d_freqlist = cp.zeros(len(shifts),dtype=cp.uint32)
            
            
                # print('Returning normalized QF^2 real values..')
                # h_result = np.zeros(len(shifts),dtype=np.float64)
                d_result = cp.zeros(len(shifts),dtype=cp.float64)
    
                # first copy the data in
                d_cutout_conj = cp.asarray(cutout).conj()
                d_rx = cp.asarray(rx)
                
                numIter = int(np.ceil(len(shifts)/BATCH))
                
                # allocate data arrays on gpu
                d_pdt_batch = cp.zeros((BATCH,len(cutout)), dtype=rx.dtype) # let's try using it in place all the way
                d_rxNormPartSq_batch = cp.zeros((BATCH), dtype=cp.float64)
                
                # new copy method allocs
                yStarts = cp.arange(BATCH, dtype=cp.int32) * len(cutout)
                lengths = cp.zeros(BATCH, dtype=cp.int32) + len(cutout)
                
                
                # now iterate over the number of iterations required
                # print("Starting cupy loop")
                for i in tqdm(range(numIter)):
                    if i == numIter-1: # on the last iteration, may have to clip
                        # print("FINAL BATCH")
                        TOTAL_THIS_BATCH = len(shifts) - BATCH * (numIter-1)
                    else:
                        TOTAL_THIS_BATCH = BATCH
                    # print(i)
                    # print (TOTAL_THIS_BATCH)
                        
                    # Slice shifts for the batch
                    xStarts = cp.asarray(shifts[i*BATCH : i*BATCH+TOTAL_THIS_BATCH], dtype=cp.int32)
                    # Copy groups
                    # cupyCopyGroups32fc(d_rx, d_pdt_batch, xStarts, yStarts[:TOTAL_THIS_BATCH], lengths[:TOTAL_THIS_BATCH])
                    # cupyCopyEqualSlicesToMatrix_32fc(d_rx, xStarts, len(cutout), d_pdt_batch[:TOTAL_THIS_BATCH,:])
                    cupyCopyIncrementalEqualSlicesToMatrix_32fc(d_rx, 
                                                                shifts[i*BATCH], 
                                                                shifts[1] - shifts[0], # assume constant increment # shifts[i*BATCH+1]-shifts[i*BATCH],
                                                                len(cutout),
                                                                TOTAL_THIS_BATCH,
                                                                d_pdt_batch[:TOTAL_THIS_BATCH,:]) # very minimal improvement of ~10%
                    # Calculate norms
                    d_rxNormPartSq_batch = cp.linalg.norm(d_pdt_batch, axis=1)**2
                    # Perform the multiply
                    cp.multiply(d_pdt_batch, d_cutout_conj, out=d_pdt_batch)

                    # Then the ffts
                    # d_pdtfft_batch = cp.abs(cp.fft.fft(d_pdt_batch))**2 # already row-wise by default
                    
                    # imax = cp.argmax(d_pdtfft_batch, axis=-1) # take the arg max for each row
                    imax, d_max = cupyArgmaxAbsRows_complex64(
                        cp.fft.fft(d_pdt_batch[:TOTAL_THIS_BATCH,:]),
                        d_argmax=d_freqlist[i*BATCH : i*BATCH+TOTAL_THIS_BATCH],
                        returnMaxValues=True,
                        THREADS_PER_BLOCK=1024
                    )
                    
                    # assign to d_freqlist output by slice
                    # d_freqlist[i*BATCH : i*BATCH + TOTAL_THIS_BATCH] = imax[:TOTAL_THIS_BATCH]
                    
                    # assign to d_result
                    # for k in range(TOTAL_THIS_BATCH):    
                        # d_result[i*BATCH + k] = d_pdtfft_batch[k, imax[k]] / d_rxNormPartSq_batch[k] / cutoutNormSq
                    d_result[i*BATCH : i*BATCH + TOTAL_THIS_BATCH] = d_max[:TOTAL_THIS_BATCH]**2 / d_rxNormPartSq_batch[:TOTAL_THIS_BATCH] / cutoutNormSq
                        

                # copy all results back
                h_result = cp.asnumpy(d_result)
                h_freqlist = cp.asnumpy(d_freqlist)
                    
                return h_result, h_freqlist
            
            else:
                
                print('Not implemented.')
                
        else:
            print('Not implemented.')
except:
    print("Failed to load cupy?")
        

def musicXcorr(cutout, rx, f_search, ftap, fs, dsr, plist, musicrows=130, shifts=None):
    # Preconjugate
    cutoutconj = cutout.conj()
    # First create the music object
    music = MUSIC(musicrows, snapshotJump=1, fwdBwd=True)
    # Calculate the downsampled fs
    fs_ds = fs/dsr
    
    # Loop over shifts as usual
    if shifts is None:
        shifts = np.arange(len(rx)-len(cutout)+1)
    
    resultsgrid = {p: np.zeros((len(shifts), len(f_search))) for p in plist}
        
    for i in range(len(shifts)):
        s = shifts[i]
        rxslice = rx[s:s+len(cutout)]
        pdt = rxslice * cutoutconj
        # Filter the product
        pdtfilt = sps.lfilter(ftap,1,pdt)
        # Create a dictionary of the downsampled arrays for all phases
        pdtfiltdsdict = {k: pdtfilt[int(len(ftap)/2)+k::dsr] for k in range(dsr)}
        # Run the music algo on it
        f, u, s, vh, Rx = music.run(pdtfiltdsdict, f_search/fs_ds, plist, useSignalAsNumerator=True)
        for k in range(len(plist)):
            p = plist[k]
            resultsgrid[p][i,:] = f[k,:] # results dictionary is labelled by p-val as key, with 'CAF'-like matrix as value
    

    return resultsgrid


def cztXcorr(cutout, rx, f_searchMin, f_searchMax, fs, cztStep=0.1, outputCAF=False, shifts=None):
    cztobj = CZTCached(cutout.size, f_searchMin, f_searchMax, cztStep, fs)
    
    # Create the freq array
    f_search = cztobj.getFreq() # np.arange(f_searchMin, f_searchMax, cztStep)
    if shifts is None:
        shifts = np.arange(len(rx)-len(cutout)+1)
    
    # common numbers in all consequent methods
    cutoutNorm = np.linalg.norm(cutout)
    cutoutNormSq = cutoutNorm**2.0
    
    if outputCAF:
        result = np.zeros((len(shifts), f_search.size))
        cutoutconj = cutout.conj()    
        for i in np.arange(len(shifts)):
            s = shifts[i]
            rxslice = rx[s:s+len(cutout)]
            rxNormPartSq = np.linalg.norm(rxslice)**2
            pdt = rxslice * cutoutconj
            # pdtczt = czt(pdt, f_search[0],f_search[-1]+cztStep/2, cztStep, fs)
            pdtczt = cztobj.run(pdt)
            result[i,:] = np.abs(pdtczt)**2.0 / rxNormPartSq / cutoutNormSq
            
        return result, f_search
        
    else:
        result = np.zeros(shifts.size, dtype=rx.dtype)
        freqs = np.zeros(shifts.size, dtype=np.float64)
        cutoutconj = cutout.conj()
        for i in np.arange(len(shifts)):
            s = shifts[i]
            rxslice = rx[s:s+len(cutout)]
            rxNorm = np.linalg.norm(rxslice)
            pdt = rxslice * cutoutconj
            # pdtczt = czt(pdt, f_search[0], f_search[-1]+cztStep/2, cztStep, fs)
            pdtczt = cztobj.run(pdt)
            abspdtczt = np.abs(pdtczt)
            mi = np.argmax(abspdtczt)
            result[i] = pdtczt[mi] / rxNorm / cutoutNorm
            freqs[i] = f_search[mi]
            
        return result, freqs
    

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
        # print('No frequency scanning xcorr..')
    
        if absResult is True:
            # print('Returning normalized QF^2 real values..')
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
        # print('Frequency scanning, but no CAF output (flattened to time)..')
        freqlist = np.zeros(len(shifts),dtype=np.uint32)
        
        if absResult is True:
            # print('Returning normalized QF^2 real values..')
            result = np.zeros(len(shifts),dtype=np.float64)

            for i in range(len(shifts)):
                s = shifts[i]
                pdt = rx[s:s+len(cutout)] * cutout.conj()
                pdtfft = sp.fft.fft(pdt)
                pdtfftsq = pdtfft**2.0
                imax = np.argmax(np.abs(pdtfftsq))
                freqlist[i] = imax
                pmax = np.abs(pdtfftsq[imax])
    
                rxNormPartSq = np.linalg.norm(rx[s:s+len(cutout)])**2.0
                result[i] = pmax/cutoutNormSq/rxNormPartSq
                
            return result, freqlist
        
        else:
            # print('Returning normalized QF complex values..')
            result = np.zeros(len(shifts),dtype=np.complex128)
            
            for i in range(len(shifts)):
                s = shifts[i]
                pdt = rx[s:s+len(cutout)] * cutout.conj()
                pdtfft = sp.fft.fft(pdt)
                imax = np.argmax(np.abs(pdtfft))
                freqlist[i] = imax
                pmax = pdtfft[imax]
    
                rxNormPart = np.linalg.norm(rx[s:s+len(cutout)])
                result[i] = pmax/cutoutNorm/rxNormPart
                
            return result, freqlist
    
    else:
        # print('Frequency scanning, outputting raw CAF...')
        
        if absResult is True:
            # print('Returning normalized QF^2 real values..')
            result = np.zeros((len(shifts), len(cutout)), dtype=np.float64)

            for i in range(len(shifts)):
                s = shifts[i]
                pdt = rx[s:s+len(cutout)] * cutout.conj()
                pdtfft = sp.fft.fft(pdt)
                pdtfftsq = np.abs(pdtfft**2.0)
    
                rxNormPartSq = np.linalg.norm(rx[s:s+len(cutout)])**2.0
                result[i] = pdtfftsq/cutoutNormSq/rxNormPartSq
    
            return result
        
        else:
            # print('Returning normalized QF complex values..')
            result = np.zeros((len(shifts), len(cutout)), dtype=np.complex128)
            
            for i in range(len(shifts)):
                s = shifts[i]
                pdt = rx[s:s+len(cutout)] * cutout.conj()
                pdtfft = np.fft.fft(pdt)
    
                rxNormPart = np.linalg.norm(rx[s:s+len(cutout)])
                result[i] = pdtfft/cutoutNorm/rxNormPart
    
            return result
    

def fineFreqTimeSearch(x_aligned, y_aligned, 
                       fineRes, freqfound, 
                       freqRes, fs, 
                       td_scan_range, steeringvec=None,
                       td_scan_freqBounds=None):
    '''
    Performs a finer search to align frequency and time, in two separate processes.
    As such, this function may not result in the global minimum for TDOA/FDOA in 2-D, but will perform much
    faster, since it only searches one index in both dimensions, rather than the full 2-D space.
    
    The best fine frequency will be searched first (assumes that the sample-aligned arrays are correctly time-aligned).
    Then, using the new value for the fine frequency alignment, the two arrays are then sub-sample time aligned.
    
    Convention is for positive values to mean that y_aligned to be LATER than x_aligned i.e. timediff = tau_y - tau_x
    Example: 
        for positive tau, x_aligned = x(t), y_aligned = x(t-tau), then timediff will return tau.
        for positive tau, x_aligned = x(t), y_aligned = x(t+tau), then timediff will return -tau.
    
    Usually coupled with a standard sample-wise xcorr from fastXcorr functions above. In that scenario, a selected signal
    named 'cutout' will have a certain (sample period * num samples) delay against another 'rx' array. After selecting
    y_aligned from the 'rx' array and using this function, the total time difference should be 'delay + timediff'.
    
    Example pseudocode: 
        result = fastXcorr(...)
        td = np.argmax(result) * T
        td = td + timediff
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

    if td_scan_freqBounds is not None:
        freqvec = makeFreq(y_fft.size, fs)
        rx_vec[np.argwhere(np.logical_or(
            freqvec < td_scan_freqBounds[0], freqvec >= td_scan_freqBounds[1]
        )).reshape(-1)] = 0
    
    cost_vec = np.dot(rx_vec, steeringvec.conj().T)/np.linalg.norm(x_fft)/np.linalg.norm(y_fft)
    idx_td = np.argmax(np.abs(cost_vec))
    timediff = td_scan_range[idx_td]
    
    return finefreqfound, timediff, cost_vec
    
def makeTimeScanSteervec(td_scan_range, fs, siglen):
    sigFreq = makeFreq(siglen, fs)
    
    mat = np.exp(1j*2*np.pi*sigFreq * td_scan_range.reshape((-1,1)))
    
    return mat

#%% TODO: refactor , add docstrings
class GenXcorr:
    def __init__(self, td_scan_range, fs, siglen):
        self.td_scan_range = td_scan_range
        self.fs = fs
        self.sigFreq = makeFreq(siglen, fs)
        self.steeringvec = self._makeTimeScanSteervec()
        self.td_scan_freqBounds = None

    def _makeTimeScanSteervec(self):
        mat = np.exp(1j*2*np.pi*self.sigFreq * self.td_scan_range.reshape((-1,1)))
        return mat
    
    def setTDscan_freqBounds(self, td_scan_freqBounds: np.ndarray):
        self.td_scan_freqBounds = td_scan_freqBounds

    def xcorr(self, x: np.ndarray, y: np.ndarray):
        x_fft = np.fft.fft(x)
        y_fft = np.fft.fft(y)
        rx_vec = x_fft * y_fft.conj()

        if self.td_scan_freqBounds is not None:
            rx_vec[np.argwhere(np.logical_or(
                self.sigFreq < self.td_scan_freqBounds[0], self.sigFreq >= self.td_scan_freqBounds[1]
            )).reshape(-1)] = 0
        
        cost_vec = np.dot(rx_vec, self.steeringvec.conj().T)/np.linalg.norm(x_fft)/np.linalg.norm(y_fft)
        idx_td = np.argmax(np.abs(cost_vec))
        timediff = self.td_scan_range[idx_td]
        
        return timediff, cost_vec

#%%
def convertQF2toSNR(qf2):
    """For xcorr against pure signal."""
    return qf2/(1.0-qf2)

def convertQF2toEffSNR(qf2):
    """For xcorr of two noisy signals."""
    return 2.0*qf2/(1.0-qf2)

def convertEffSNRtoQF2(effSNR):
    """For back-conversion."""
    return effSNR/(2 + effSNR)

def expectedEffSNR(snr1, snr2=np.inf, OSR=1):
    '''
    Effective SNR is defined as 1/(0.5 * (1/y1 + 1/y2 + 1/y1y2)),
    where y1 and y2 are the respective SNRs, with respect to the noise BW.
    
    Example:
        10 dB, in-band SNR signal.
        Pure signal. (inf SNR)
        OSR = 2
        
        Effective SNR = 20.0/2 = 10.0
    
    Taken from Algorithms for Ambiguity Function Processing. SEYMOUR STEIN. 
    '''

    y = 1.0/(0.5 * (1/snr1 + 1/snr2 + 1/snr1/snr2))
    y = y/OSR # Scale by OSR
    return y

def sigmaDTO(signalBW, noiseBW, integTime, effSNR):
    '''
    Taken from Algorithms for Ambiguity Function Processing. SEYMOUR STEIN. 
    '''
    beta = np.pi / np.sqrt(3) * signalBW
    s = 1.0/beta / np.sqrt(noiseBW * integTime * effSNR)
    return s

def sigmaDFO(noiseBW, integTime, effSNR):
    '''
    Taken from Algorithms for Ambiguity Function Processing. SEYMOUR STEIN. 
    '''
    s = 0.55/integTime / np.sqrt(noiseBW * integTime * effSNR)
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
        
def argmax2d(m: np.ndarray):
    '''
    Returns the 2-D indices that mark the maximum value in the matrix m.

    Parameters
    ----------
    m : np.ndarray
        Input array.

    Returns
    -------
    maxind: tuple
        Indices of the max value. m[maxind[0],maxind[1]] should be the value.

    '''
    return np.unravel_index(np.argmax(m), m.shape)

def calcQF2(x: np.ndarray, y: np.ndarray):
    '''
    Simple one-line function to calculate QF2 for two equal length arrays (already aligned).
    '''
    x_energy = np.linalg.norm(x)**2
    y_energy = np.linalg.norm(y)**2
    qf2 = np.abs(np.vdot(x,y))**2 / x_energy / y_energy
    return qf2


#%%
class GroupXcorr:
    def __init__(self, y: np.ndarray, starts: np.ndarray, lengths: np.ndarray,
                 freqs: np.ndarray, fs: int,
                 autoConj: bool=True, autoZeroStarts: bool=True):
        '''
        Parameters
        ----------
        y : np.ndarray
            Full length array which would usually be used for correlation.
        starts : np.ndarray
            (Sorted) Start indices for each group.
        lengths : np.ndarray
            Corresponding lengths for each group (in samples).
        freqs : np.ndarray
            Frequencies to scan over.
        fs : int
            Sampling rate of y.
        autoConj : bool
            Conjugates the y array for use in xcorr. Default is True.
        autoZeroStarts : bool
            Automatically zeros the 'starts' array based on the first start
            (to ensure that the signal 'starts from 0'). Default is True.
        '''
        assert(starts.size == lengths.size) # this is the number of groups
        
        self.starts = starts
        if autoZeroStarts:
            self.starts = self.starts - self.starts[0]
        self.lengths = lengths
        self.numGroups = self.starts.size # easy referencing
        self.freqs = freqs
        self.fs = fs
        
        # Generate the stitched groups of y
        if autoConj:
            self.yconcat = np.hstack([y.conj()[starts[i]:starts[i]+lengths[i]] for i in range(self.numGroups)])
        else:
            self.yconcat = np.hstack([y[starts[i]:starts[i]+lengths[i]] for i in range(self.numGroups)])
        self.yconcatNormSq = np.linalg.norm(self.yconcat)**2
        
        # Generate the frequency matrix
        maxfreqMat = np.exp(-1j*2*np.pi*freqs.reshape((-1,1))*np.arange(y.size)/fs) # minus sign is FFT convention
        self.freqMat = np.hstack([maxfreqMat[:,starts[i]:starts[i]+lengths[i]] for i in range(self.numGroups)])
        
    def xcorr(self, rx: np.ndarray, shifts: np.ndarray=None):
        if shifts is None:
            shifts = np.arange(len(rx)-(self.starts[-1]+self.lengths[-1])+1)
        else:
            assert(shifts[-1] + self.starts[-1] + self.lengths[-1] < rx.size) # make sure it can access it
            
        xc = np.zeros(shifts.size)
        freqpeaks = np.zeros(shifts.size)
        for i in np.arange(shifts.size):

            shift = shifts[i]
            # Perform slicing of rx
            rxconcat = np.hstack([rx[shift + self.starts[g] : shift + self.starts[g] + self.lengths[g]] for g in np.arange(self.numGroups)])
            rxconcatNormSq = np.linalg.norm(rxconcat)**2
            # Now multiply by y
            p = rxconcat * self.yconcat
            # And the freqmat
            pf = self.freqMat @ p
            # Pick the max value
            pfabs = np.abs(pf)
            pmaxind = np.argmax(pfabs)
            # Save output (with normalisations)
            xc[i] = pfabs[pmaxind]**2 / rxconcatNormSq / self.yconcatNormSq
            freqpeaks[i] = self.freqs[pmaxind]
            
        return xc, freqpeaks        
        
class GroupXcorrCZT:
    def __init__(self, y: np.ndarray, starts: np.ndarray, lengths: np.ndarray,
                 f1: float, f2: float, binWidth: float, fs: int, autoConj: bool=True, autoZeroStarts: bool=True):
        
        assert(starts.size == lengths.size)
        self.starts = starts
        if autoZeroStarts:
            self.starts = self.starts - self.starts[0]
        self.lengths = lengths
        self.numGroups = self.starts.size # easy referencing
        self.fs = fs
        # CZT parameters
        self.f1 = f1
        self.f2 = f2
        self.binWidth = binWidth
        
        # Generate the stacked matrix of groups of y, padded to longest length
        self.maxLength = np.max(self.lengths)
        self.ystack = np.zeros((self.numGroups, self.maxLength), y.dtype)
        for i in range(self.numGroups):
            self.ystack[i,:self.lengths[i]] = y[starts[i]:starts[i]+lengths[i]]
        if autoConj:
            self.ystack = self.ystack.conj()
        
        self.ystackNormSq = np.linalg.norm(self.ystack.flatten())**2
        
        # Create a CZT object for some optimization
        self.cztc = CZTCached(self.maxLength, f1, f2, binWidth, fs)
        
    def xcorr(self, rx: np.ndarray, shifts: np.ndarray=None):
        # We are returning CAF for this (for now?)
        if shifts is None:
            shifts = np.arange(len(rx)-(self.starts[-1]+self.lengths[-1])+1)
        else:
            assert(shifts[-1] + self.starts[-1] + self.lengths[-1] < rx.size) # make sure it can access it
            
        xc = np.zeros((shifts.size, int((self.f2-self.f1)/self.binWidth + 1)))
        # Pre-calculate the phases for each group
        cztFreq = np.arange(self.f1, self.f2+self.binWidth/2, self.binWidth)
        groupPhases = np.exp(-1j*2*np.pi*cztFreq*self.starts.reshape((-1,1))/self.fs)
        
        for i in np.arange(shifts.size):
            shift = shifts[i]
            
            pdtcztPhased = np.zeros((self.numGroups,cztFreq.size), np.complex128)
            rxgroupNormSqCollect = np.zeros(self.numGroups)
            
            for g in np.arange(self.numGroups):
                ygroup = self.ystack[g,:self.lengths[g]]
                rxgroup = rx[shift+self.starts[g] : shift+self.starts[g]+self.lengths[g]]
                rxgroupNormSqCollect[g] = np.linalg.norm(rxgroup)**2
                pdt = np.zeros(self.maxLength, dtype=rxgroup.dtype)
                pdt[:ygroup.size] = ygroup * rxgroup
                # # Run the czt on the pdt now
                # pdtczt = czt(pdt, self.f1, self.f2+self.binWidth/2, self.binWidth, self.fs)
                # Use the cached CZT object
                pdtczt = self.cztc.run(pdt)
                # Shift the czt by a phase
                pdtcztPhased[g,:] = pdtczt * groupPhases[g,:]
            
            # Now sum coherently across the groups
            pdtcztCombined = np.sum(pdtcztPhased, axis=0)
            rxgroupNormSq = np.sum(rxgroupNormSqCollect)
            
            xc[i,:] = np.abs(pdtcztCombined)**2 / rxgroupNormSq / self.ystackNormSq
            
        return xc, cztFreq
    
try:
    import cupy as cp # Used to raise exception
    from spectralRoutines import CZTCachedGPU
    from cupyExtensions import *

    class GroupXcorrFFT:
        def __init__(self, ygroups: np.ndarray, starts: np.ndarray, fs: int,
                     autoConj: bool=True, fftlen=None, autoZeroStarts: bool=True):
            assert(starts.size == ygroups.shape[0])
            self.starts = starts
            if autoZeroStarts:
                self.starts = self.starts - self.starts[0]
            self.numGroups = self.starts.size
            self.fs = fs
            self.ygroups = ygroups
            self.ygroupLen = self.ygroups.shape[1] # For easy access
            if fftlen is None:
                self.fftlen = self.ygroupLen 
            else:
                self.fftlen = fftlen # Use for padding for the fft
                
            self.ygroupNormSq = np.linalg.norm(self.ygroups.flatten())**2
            
            if autoConj:
                self.ygroups = self.ygroups.conj()

            # Some pre-computes
            self.fftfreq = makeFreq(self.fftlen, self.fs)
            self.groupPhases = np.exp(-1j*2*np.pi*self.fftfreq*self.starts.reshape((-1,1))/self.fs)
            
            
        def _xcorrThread(self, rx: np.ndarray, shifts: np.ndarray):
            xc_slice = np.zeros((shifts.size, self.fftlen))

            for i, shift in enumerate(shifts):
                pdt = np.zeros((self.numGroups, self.fftfreq.size), rx.dtype)
                rxgroupNormSqCollect = np.zeros(self.numGroups)
                
                for g in np.arange(self.numGroups):
                    ygroup = self.ygroups[g,:]
                    rxgroup = rx[shift + self.starts[g] : shift + self.starts[g] + self.ygroupLen]
                    rxgroupNormSqCollect[g] = np.linalg.norm(rxgroup)**2
                    
                    # pdt[g,:self.ygroupLen] = ygroup * rxgroup
                    np.multiply(ygroup, rxgroup, out=pdt[g,:self.ygroupLen]) # ufunc direct
                    
                # At the end, run fft once on entire block
                pdtfft = np.fft.fft(pdt, n=self.fftlen, axis=1) # FFT each row
                # Then fix the phase
                np.multiply(pdtfft, self.groupPhases, out=pdtfft) # in-place
                # And then sum across the groups (rows)
                pdtfftCombined = np.sum(pdtfft, axis=0)
                rxgroupNormSq = np.sum(rxgroupNormSqCollect)
                # Scale by the normsqs
                xc_slice[i, :] = np.abs(pdtfftCombined)**2 / rxgroupNormSq / self.ygroupNormSq
            
            return xc_slice

        def xcorrThreads(self, rx: np.ndarray, shifts: np.ndarray=None, NUM_THREADS: int=4):
            if shifts is None:
                shifts = np.arange(len(rx)-(self.starts[-1]+self.fftlen)+1)
            else:
                assert(shifts[-1] + self.starts[-1] + self.fftlen < rx.size)

            xc = np.zeros((shifts.size, self.fftlen))

            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                futures = {executor.submit(self._xcorrThread, rx, shifts[i::NUM_THREADS]): i for i in range(NUM_THREADS)}
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    xc[i::NUM_THREADS, :] = future.result()

            return xc

        def xcorr(self, rx: np.ndarray, shifts: np.ndarray=None, flattenToTime: bool=True):
            if shifts is None:
                shifts = np.arange(len(rx)-(self.starts[-1]+self.fftlen)+1)
            else:
                assert(shifts[-1] + self.starts[-1] + self.fftlen < rx.size)
                
            if flattenToTime:
                xc = np.zeros(shifts.size)
                fi = np.zeros(shifts.size, dtype=np.uint32)
            else:
                xc = np.zeros((shifts.size, self.fftlen))

            for i, shift in enumerate(shifts):
                pdt = np.zeros((self.numGroups, self.fftfreq.size), rx.dtype)
                rxgroupNormSqCollect = np.zeros(self.numGroups)
                
                for g in np.arange(self.numGroups):
                    ygroup = self.ygroups[g,:]
                    rxgroup = rx[shift + self.starts[g] : shift + self.starts[g] + self.ygroupLen]
                    rxgroupNormSqCollect[g] = np.linalg.norm(rxgroup)**2
                    
                    # pdt[g,:self.ygroupLen] = ygroup * rxgroup
                    np.multiply(ygroup, rxgroup, out=pdt[g,:self.ygroupLen]) # ufunc direct
                    
                # At the end, run fft once on entire block
                pdtfft = np.fft.fft(pdt, n=self.fftlen, axis=1) # FFT each row
                # Then fix the phase
                np.multiply(pdtfft, self.groupPhases, out=pdtfft) # in-place
                # And then sum across the groups (rows)
                pdtfftCombined = np.sum(pdtfft, axis=0)
                rxgroupNormSq = np.sum(rxgroupNormSqCollect)
                # Scale by the normsqs
                if flattenToTime:
                    ff = np.abs(pdtfftCombined)**2 / rxgroupNormSq / self.ygroupNormSq
                    fmi = np.argmax(ff)
                    xc[i] = ff[fmi]
                    fi[i] = fmi
                    
                else:
                    xc[i, :] = np.abs(pdtfftCombined)**2 / rxgroupNormSq / self.ygroupNormSq
                
            if flattenToTime:
                return xc, fi
            else:
                return xc

        def xcorrGPU(self, rx: cp.ndarray, shifts: np.ndarray=None, flattenToTime: bool=True):
            if shifts is None:
                shifts = np.arange(len(rx)-(self.starts[-1]+self.fftlen)+1)
            else:
                assert(shifts[-1] + self.starts[-1] + self.fftlen < rx.size)
                
            if flattenToTime:
                xc = cp.zeros(shifts.size)
                fi = cp.zeros(shifts.size, dtype=cp.uint32)
            else:
                xc = cp.zeros((shifts.size, self.fftlen))

            # GPU pre-alloc
            pdt = cp.zeros((self.numGroups, self.fftlen), rx.dtype)
            rxgroups = cp.zeros_like(pdt)
            d_ygroups = cp.asarray(self.ygroups, dtype=cp.complex64)
            d_groupPhases = cp.asarray(self.groupPhases)
            pdtfftCombined = cp.zeros(self.fftlen, rx.dtype)
            dstarts = cp.asarray(self.starts, dtype=cp.int32)
            xStarts = cp.zeros(self.numGroups, cp.int32)
            yStarts = cp.arange(self.numGroups, dtype=cp.int32) * self.fftlen # Constant
            lengths = cp.zeros(self.numGroups, cp.int32) + self.ygroupLen # Constant

            for i, shift in enumerate(shifts):
                # Zero-ing
                pdt[:] = 0
                rxgroups[:] = 0
                xStarts[:] = 0
                
                # Note: creation of this matrix is inefficient on GPU, should multiply directly if possible
                # Construct the matrix of all the groups
                # DEPRECATED
                # for g in np.arange(self.numGroups):
                #     rxgroups[g, :self.ygroupLen] = rx[shift + self.starts[g] : shift + self.starts[g] + self.ygroupLen]
                # Use the new kernel copy
                cp.add(shift, dstarts, out=xStarts)
                cupyCopyGroups32fc(rx, rxgroups, xStarts, yStarts, lengths)

                # Calculate all the rxgroup norms
                # rxgroupNormSqCollect = cp.linalg.norm(rxgroups, axis=1)**2 # Deprecated
                # Multiply all the groups together element-wise
                cp.multiply(d_ygroups, rxgroups, out=pdt)
                    
                # At the end, run fft once on entire block
                pdtfft = cp.fft.fft(pdt, n=self.fftlen, axis=1) # FFT each row
                # Then fix the phase
                cp.multiply(pdtfft, d_groupPhases, out=pdtfft) # in-place
                # And then sum across the groups (rows)
                # pdtfftCombined = cp.sum(pdtfft, axis=0) # deprecated
                cp.sum(pdtfft, axis=0, out=pdtfftCombined)
                # rxgroupNormSq = cp.sum(rxgroupNormSqCollect) # Deprecated
                rxgroupNormSq = cp.sum(cp.abs(rxgroups)**2)
                # Scale by the normsqs
                if flattenToTime:
                    ff = cp.abs(pdtfftCombined)**2
                    fmi = cp.argmax(ff)
                    xc[i] = ff[fmi] / rxgroupNormSq / self.ygroupNormSq
                    fi[i] = fmi
                    
                else:
                    xc[i, :] = cp.abs(pdtfftCombined)**2 / rxgroupNormSq / self.ygroupNormSq
                
            if flattenToTime:
                return xc, fi
            else:
                return xc

    class GroupXcorrCZT_Permutations:
        '''
        Purpose of this class is to automatically xcorr different permutations of the groups, without unnecessary repeats.
        Example case with 2 groups, template looks like
        
        T0_____T1_____
        
        but each template - T0, T1 - is selected from a set; example 
        T0 <=> (T0_a, T0_b)
        T1 <=> (T1_a, T1_b, T1_c)

        where each group may have a different sized set. In this case, there must be 2 X 3 total permutations i.e. 6 total correlations.
        However, the 5 groups may be correlated individually, and then combined to form the group correlation values.
        This, in general, reduces the computational load to the size of the sets, rather than the product of the size of the sets.
        
        So the class iterates over the group index, and the template index for each group index.
        Assumption here is that each group has the same length, unlike the previous classes. This is to optimize the size of the czt for re-use.
        '''
        def __init__(self, ygroups: np.ndarray, ygroupIdxs: np.ndarray, groupStarts: np.ndarray,
                     f1: float, f2: float, binWidth: float, fs: int, autoConj: bool=True):
            
            assert(ygroups.shape[0] == ygroupIdxs.size) # ensure numTemplates corresponds
            assert(np.unique(ygroupIdxs).size == groupStarts.size) # ensure numGroups corresponds
            
            self.numTemplates = ygroupIdxs.size
            self.numGroups = groupStarts.size
            print("Total %d templates, %d groups" % (self.numTemplates, self.numGroups))
            
            assert(np.all(np.sort(np.unique(ygroupIdxs)) == np.arange(self.numGroups))) # ensure all groups accounted for and was sorted
            
            # Copying inputs
            self.groupStarts = groupStarts
            self.ygroupIdxs = ygroupIdxs
            self.ygroups = ygroups
            self.fs = fs
            self.length = ygroups.shape[1]
            # CZT parameters
            self.f1 = f1
            self.f2 = f2
            self.binWidth = binWidth
            
            # Auto conj
            if autoConj:
                self.ygroups = self.ygroups.conj()
            self.ygroupsEnergy = np.linalg.norm(self.ygroups, axis=1)**2
            
        def xcorrGPU(self, rx: cp.ndarray, shifts: np.ndarray, batchSz=32):
            '''
            This is meant to be a performant GPU implementation based on maximum CZT batched use.
            As such, it is recommended that rx be converted to complex64, as that is the dtype that will be returned.
            It is also expected that the rx array is passed in already on the GPU.
            
            Since much of the data will be held on the GPU, the user should be aware that using a 
            long 'shifts' vector will have a massive memory footprint.
            '''
            # Check that shifts is not out of range
            assert(shifts[-1] + self.groupStarts[-1] + self.length < rx.size)
            
            # Initialize GPU memory for outputs
            self.d_xcTemplates = cp.zeros((self.numTemplates, shifts.size, int((self.f2-self.f1)/self.binWidth + 1)), dtype=cp.complex64)
            self.d_rxgroupNormSq = cp.zeros((self.numGroups, shifts.size), dtype=cp.float32)
            
            # Pre-calculate the phases for each group (TODO: i should move to this init?)
            cztFreq = cp.arange(self.f1, self.f2+self.binWidth/2, self.binWidth) # Compute as doubles
            groupPhases = cp.exp(-1j*2*cp.pi*cztFreq*cp.array(self.groupStarts.reshape((-1,1)))/self.fs).astype(cp.complex64) # Convert to floats
            
            # Instantiate the CZT object
            cztcg = CZTCachedGPU(self.length, self.f1, self.f2, self.binWidth, self.fs)
            
            # Calculate the number to complete per batch
            numPerBatch = np.hstack((np.zeros(int(shifts.size/batchSz),dtype=np.int32) + batchSz, np.remainder(shifts.size,batchSz,dtype=np.int32)))
            startIdxPerBatch = np.hstack((0, np.cumsum(numPerBatch[:-1])))
            
            # Pre-alloc a batch storage matrices?
            d_mulMatrix = cp.zeros((batchSz,self.ygroups.shape[1]), dtype=cp.complex64)
            d_pdtcztMatrix = cp.zeros((batchSz,cztcg.k), dtype=cp.complex64)
            
            # Counter for normsq completion?
            groupNormSqCompleted = np.zeros(self.numGroups, dtype=np.uint8)
            
            # Loop over the templates instead of shifts
            for k in np.arange(self.numTemplates):
                # Load the current template to device
                d_template = cp.array(self.ygroups[k,:], dtype=cp.complex64)
                # Get the group number for the template
                groupNumber = self.ygroupIdxs[k]
                # And the groupStart index
                groupStartIdx = self.groupStarts[groupNumber]
                
                # Calculate the norms outside the inner batch loop?
                if groupNormSqCompleted[groupNumber] == 0:
                    # Calculate the slice for the shift -> absSq
                    d_rxSliceAbsSq = cp.abs(rx[groupStartIdx+shifts[0]:groupStartIdx+shifts[-1]+self.length])**2
                    for i in np.arange(shifts.size):
                        self.d_rxgroupNormSq[groupNumber,i] = cp.sum(d_rxSliceAbsSq[i:i+self.length])
                
                # Loop over batches
                for b in np.arange(numPerBatch.size):
                    for i in np.arange(numPerBatch[b]):
                        shift = shifts[startIdxPerBatch[b] + i]
                        
                        # Multiply the appropriate slice
                        # mulMatrix[:numPerBatch[i]] = rx[shift+groupStartIdx : shift+groupStartIdx+self.length] * d_template
                        cp.multiply(rx[shift+groupStartIdx : shift+groupStartIdx+self.length],
                                    d_template,
                                    out = d_mulMatrix[i,:]) # Write to memory directly
                        # # Calculate the groupNormSq
                        # if self.d_rxgroupNormSq[groupNumber, i] == 0:
                        #     self.d_rxgroupNormSq[groupNumber,i] = cp.linalg.norm(rx[shift+groupStartIdx : shift+groupStartIdx+self.length])**2
                            
                    # Run czt on the entire batch
                    # d_pdtcztMatrix = cztcg.runMany(d_mulMatrix[:numPerBatch[b]])
                    cztcg.runMany(d_mulMatrix[:numPerBatch[b]], out=d_pdtcztMatrix[:numPerBatch[b]])
                    # Now multiply the phase and save directly
                    # self.d_xcTemplates[k, startIdxPerBatch[b]:startIdxPerBatch[b]+numPerBatch[b], :] = d_pdtcztMatrix * groupPhases[groupNumber,:]
                    cp.multiply(d_pdtcztMatrix[:numPerBatch[b]],
                                groupPhases[groupNumber,:],
                                out=self.d_xcTemplates[k, startIdxPerBatch[b]:startIdxPerBatch[b]+numPerBatch[b], :]) # Write to memory directly
                    
            return cp.asnumpy(cztFreq)
            
        def getCAF_GPU(self, templateIdx: np.ndarray):
            assert(templateIdx.size == self.numGroups)
            
            # Initialize array
            cafcplx = cp.zeros((self.d_xcTemplates.shape[1],self.d_xcTemplates.shape[2]), dtype=np.complex64)

            rxnormsq = cp.zeros(self.d_rxgroupNormSq.shape[1])
            ynormsq = np.zeros(1) # we can leave this on cpu

            # GPU
            for groupNumber in np.arange(templateIdx.size):
                templateNumber = np.argwhere(self.ygroupIdxs == groupNumber)[templateIdx[groupNumber]][0] # cupy needs the 0

                # cafcplx[:,:] = cafcplx[:,:] + self.d_xcTemplates[templateNumber,:,:] 
                cp.add(cafcplx, self.d_xcTemplates[templateNumber,:,:], out = cafcplx)
                # rxnormsq += self.rxgroupNormSq[groupNumber,:]
                cp.add(rxnormsq, self.d_rxgroupNormSq[groupNumber,:], out=rxnormsq)
                # ynormsq += self.ygroupsEnergy[templateNumber]
                np.add(ynormsq, self.ygroupsEnergy[templateNumber], out=ynormsq)
                
            caf = cp.abs(cafcplx)**2/rxnormsq.reshape(-1,1)/ynormsq[0]
            
            return caf
        
        ## CPU Methods
        def xcorr(self, rx: np.ndarray, shifts: np.ndarray=None, numThreads: int=1):
            # We are returning CAF for this (for now?)
            if shifts is None:
                shifts = np.arange(len(rx)-(self.groupStarts[-1]+self.length)+1)
            else:
                assert(shifts[-1] + self.groupStarts[-1] + self.length < rx.size) # make sure it can access it
            
            # Init memory for outputs
            self.xcTemplates = np.zeros((self.numTemplates, shifts.size, int((self.f2-self.f1)/self.binWidth + 1)), dtype=np.complex128)
            self.rxgroupNormSq = np.zeros((self.numGroups, shifts.size))
            
            # Pre-calculate the phases for each group
            cztFreq = np.arange(self.f1, self.f2+self.binWidth/2, self.binWidth)
            groupPhases = np.exp(-1j*2*np.pi*cztFreq*self.groupStarts.reshape((-1,1))/self.fs)
            
            # # Debug single thread
            # self._xcorrThread(shifts,rx,self.numGroups,self.numTemplates,cztFreq,
            #                   self.ygroups,self.ygroupsEnergy,self.ygroupIdxs,self.groupStarts,groupPhases,
            #                   self.length, self.f1, self.f2, self.binWidth,self.fs, 
            #                   xcTemplates, rxgroupNormSq, 1,0)
            # Start threads
            with ThreadPoolExecutor(max_workers=numThreads) as executor:
                future_x = {executor.submit(self._xcorrThread, shifts, rx, self.numGroups, self.numTemplates, cztFreq,
                                            self.ygroups, self.ygroupsEnergy, self.ygroupIdxs, self.groupStarts, groupPhases,
                                            self.length, self.f1, self.f2, self.binWidth, self.fs,
                                            self.xcTemplates, self.rxgroupNormSq,
                                            numThreads, i) : i for i in np.arange(numThreads)}
            # After threads are done,
            # breakpoint()
            
            return cztFreq
        
        def getCAF(self, templateIdx: np.ndarray, numThreads: int=4):
            assert(templateIdx.size == self.numGroups)
            
            # Initialize array
            cafcplx = np.zeros((self.xcTemplates.shape[1],self.xcTemplates.shape[2]), dtype=np.complex128)
            rxnormsq = np.zeros(self.rxgroupNormSq.shape[1])
            ynormsq = np.zeros(1)

            # # V1, unthreaded
            # for groupNumber in range(templateIdx.size):
            #     templateNumber = np.argwhere(self.ygroupIdxs == groupNumber)[templateIdx[groupNumber]]
                
            #     cafcplx[:,:] = cafcplx[:,:] + self.xcTemplates[templateNumber,:,:]
            #     rxnormsq += self.rxgroupNormSq[groupNumber,:]
            #     ynormsq += self.ygroupsEnergy[templateNumber]
            
            # V2, threaded (but insignificant speedup here..)
            with ThreadPoolExecutor(max_workers=numThreads) as executor:
                future_x = {executor.submit(self._getCAFThread, templateIdx,
                                            cafcplx, rxnormsq, ynormsq,
                                            i, numThreads) : i for i in np.arange(numThreads)}
                
            caf = np.abs(cafcplx)**2/rxnormsq.reshape(-1,1)/ynormsq
            
            return caf
                
        def _getCAFThread(self, templateIdx: np.ndarray, 
                          cafcplx: np.ndarray, rxnormsq: np.ndarray, ynormsq: np.ndarray,
                          thIdx: int, numThreads: int):
            
            # Define the indices to work on for this thread
            numPerThread = int(self.xcTemplates.shape[1] / numThreads)
            numLastThread = self.xcTemplates.shape[1] - numPerThread * (numThreads-1)
            
            for groupNumber in np.arange(templateIdx.size):
                templateNumber = np.argwhere(self.ygroupIdxs == groupNumber)[templateIdx[groupNumber]]
                
                # First thread computes the norm squared values
                if thIdx == 0: 
                    rxnormsq += self.rxgroupNormSq[groupNumber,:]
                    ynormsq += self.ygroupsEnergy[templateNumber]
                
                # Last thread takes the remainder rows
                if thIdx == numThreads - 1:
                    cafcplx[-numLastThread:,:] = cafcplx[-numLastThread:,:] + self.xcTemplates[templateNumber,-numLastThread:,:]
                    # np.add(cafcplx[-numLastThread:,:],
                    #        self.xcTemplates[templateNumber,-numLastThread:,:],
                    #        out=cafcplx[-numLastThread:,:])
                else: # All previous threads compute based on the thread number
                    cafcplx[thIdx*numPerThread:(thIdx+1)*numPerThread,:] = cafcplx[thIdx*numPerThread:(thIdx+1)*numPerThread,:] + self.xcTemplates[templateNumber,thIdx*numPerThread:(thIdx+1)*numPerThread,:]
                    # np.add(cafcplx[thIdx*numPerThread:(thIdx+1)*numPerThread,:],
                    #        self.xcTemplates[templateNumber,thIdx*numPerThread:(thIdx+1)*numPerThread,:],
                    #        out=cafcplx[thIdx*numPerThread:(thIdx+1)*numPerThread,:])
                    
                
            
        @staticmethod
        # @jit(nopython=True, nogil=True) # make sure it releases the gil?
        def _xcorrThread(shifts, rx, numGroups, numTemplates, cztFreq,
                         ygroups, ygroupsEnergy, ygroupIdxs, groupStarts, groupPhases,
                         length, f1, f2, binWidth, fs,
                         xcTemplates, rxgroupNormSq, # outputs, written directly to array
                         numThreads, thIdx):
            # Create the czt cached object in the thread
            cztc = CZTCached(length, f1, f2, binWidth, fs)
            
            # Loop with numThreads strides over shifts
            for i in np.arange(thIdx,shifts.size,numThreads):
                shift = shifts[i]
                
                # Loop over the templates
                for k in np.arange(numTemplates):
                    groupNumber = ygroupIdxs[k] # Get the group number for this template
                    ygroup = ygroups[k, :] # Get the array for this template
                    # Slice the correct start index for this group
                    rxgroup = rx[shift+groupStarts[groupNumber] : shift+groupStarts[groupNumber]+length]
                    if rxgroupNormSq[groupNumber, i] == 0: # Write to the normSq if not yet done i.e. only first template of the group
                        rxgroupNormSq[groupNumber, i] = np.linalg.norm(rxgroup)**2 # may want to wrap this in if-else since re-calculated

                    pdt = ygroup * rxgroup
                    # Run the czt from the cached object
                    pdtczt = cztc.run(pdt)
                    # Shift the czt by a phase appropriate to the group number
                    # But save it to the template index
                    xcTemplates[k, i, :] = pdtczt * groupPhases[groupNumber,:]
            
            # END THREAD. Normalisations to be done outside

            
            
        

    #%%
    group_xcorr_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void group_xcorr_kernel(const complex<float>* d_rx, 
                            const complex<float>* d_y, const int ylen, const float yNormSq,
                            const float* d_nFreqs, const int numFreqs,
                            const int* d_gIdx, // this has length ylen
                            const int* d_shifts, const int numShifts, const int numShiftsPerBlk,
                            float* d_xc, int* d_freqinds){
                                            
        // allocate shared memory
        extern __shared__ double s[];
        
        complex<float> *s_y = (complex<float>*)s; // (ylen) complex floats
        float *s_nFreqs = (float*)&s_y[ylen]; // (numFreqs) real floats
        complex<float> *s_rx = (complex<float>*)&s_nFreqs[numFreqs]; // (ylen) complex floats
        int *s_gIdx = (int*)&s_rx[ylen]; // (ylen) ints
        float *s_absVals = (float*)&s_gIdx[ylen]; // (numFreqs) real floats
        /* Tally: (ylen*2)*32fc + (numFreqs*2)*32f + (ylen)*32s */
        
        // load shared memory
        for (int t = threadIdx.x; t < ylen; t = t + blockDim.x){
            s_y[t] = d_y[t];
            s_gIdx[t] = d_gIdx[t];
        }
        for (int t = threadIdx.x; t < numFreqs; t = t + blockDim.x){
            s_nFreqs[t] = d_nFreqs[t];
        }
        // nothing to load for s_rx, it's a workspace
        
        __syncthreads();
                         
        // loop over the shifts for this block
        int shift, shiftIdx;
        float rxNormSq;
        double fprod_real, fprod_imag;
        complex<double> val;
        int maxIdx;
        float maxVal;
        
        for (int blkShift = 0; blkShift < numShiftsPerBlk; blkShift++){
            shiftIdx = blockIdx.x * numShiftsPerBlk + blkShift;
            shift = d_shifts[shiftIdx];
            
            // load the values from d_rx appropriately
            for (int t = threadIdx.x; t < ylen; t = t + blockDim.x){
                s_rx[t] = d_rx[shift + s_gIdx[t]];
            }
            
            __syncthreads(); // sync before calculating normSq otherwise some array values not written yet
            
            // each thread just calculates the rxNormSq for itself
            rxNormSq = 0;
            for (int i = 0; i < ylen; i++){
                rxNormSq = fmaf(abs(s_rx[i]), abs(s_rx[i]), rxNormSq);
            }
            
            __syncthreads(); // must sync before multiplying or else some threads will have wrong normSq
            
            // multiply y in-place
            for (int t = threadIdx.x; t < ylen; t = t + blockDim.x){
                s_rx[t] = s_rx[t] * s_y[t];
            }
            
            // now each thread calculates the dot product with an appropriate frequency vector
            // i.e. thread (t): frequency index, loops over (i): index of rx
            for (int t = threadIdx.x; t < numFreqs; t = t + blockDim.x){
                val = 0;
                for (int i = 0; i < ylen; i++){
                    sincospi(-2.0 * (double)s_nFreqs[t] * (double)s_gIdx[i], &fprod_imag, &fprod_real); // this is extremely slow
                    val = val + complex<double>(fprod_real, fprod_imag) * complex<double>(s_rx[i]); // no FMA intrinsics for complex..
                }
                // when val is complete, write it to shared mem array for storage first
                s_absVals[t] = (float)abs(val); // cast to float when writing
                    
            }
            
            __syncthreads(); // wait for all absVals to be written
                                
            // use the first thread to scan for the maximum
            if (threadIdx.x == 0){
                maxIdx = 0;
                maxVal = s_absVals[0];
                for (int i = 1; i < numFreqs; i++){
                    if (s_absVals[i] > maxVal){
                        maxIdx = i;
                        maxVal = s_absVals[i];
                    }
                }
                
                // and write directly to the global mem
                d_freqinds[shiftIdx] = maxIdx;
                d_xc[shiftIdx] = maxVal * maxVal / rxNormSq / yNormSq;
                // d_xc[shiftIdx] = rxNormSq; // cheating debug statement

            }
            
        }
        
    }
    ''','group_xcorr_kernel')

    group_xcorr_kernelv2 = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void group_xcorr_kernelv2(const complex<float>* d_rx,
                            const complex<float>* d_y, const int ylen, const float yNormSq,
                            const complex<float>* d_freqMat, const int numFreqs,
                            const int* d_gIdx, // this has length ylen
                            const int* d_shifts, const int numShifts, const int numShiftsPerBlk,
                            float* d_xc, int* d_freqinds){
                                            
        // allocate shared memory
        extern __shared__ double s[];
        
        complex<float> *s_y = (complex<float>*)s; // (ylen) complex floats
        complex<float> *s_rx = (complex<float>*)&s_y[ylen]; // (ylen) complex floats
        int *s_gIdx = (int*)&s_rx[ylen]; // (ylen) ints
        float *s_absVals = (float*)&s_gIdx[ylen]; // (numFreqs) real floats
        /* Tally: (ylen*2)*32fc + (numFreqs)*32f + (ylen)*32s */
        
        // load shared memory
        for (int t = threadIdx.x; t < ylen; t = t + blockDim.x){
            s_y[t] = d_y[t];
            s_gIdx[t] = d_gIdx[t];
        }
        // nothing to load for s_rx, it's a workspace
        
        __syncthreads();
                         
        // loop over the shifts for this block
        int shift, shiftIdx;
        float rxNormSq;
        complex<float> val;
        int maxIdx;
        float maxVal;
        
        for (int blkShift = 0; blkShift < numShiftsPerBlk; blkShift++){
            shiftIdx = blockIdx.x * numShiftsPerBlk + blkShift;
            shift = d_shifts[shiftIdx];
            
            // load the values from d_rx appropriately
            for (int t = threadIdx.x; t < ylen; t = t + blockDim.x){
                s_rx[t] = d_rx[shift + s_gIdx[t]];
            }
            
            __syncthreads(); // sync before calculating normSq otherwise some array values not written yet
            
            // each thread just calculates the rxNormSq for itself
            rxNormSq = 0;
            for (int i = 0; i < ylen; i++){
                rxNormSq = fmaf(abs(s_rx[i]), abs(s_rx[i]), rxNormSq);
            }
            
            __syncthreads(); // must sync before multiplying or else some threads will have wrong normSq
            
            // multiply y in-place
            for (int t = threadIdx.x; t < ylen; t = t + blockDim.x){
                s_rx[t] = s_rx[t] * s_y[t];
            }
            
            // now each thread calculates the dot product with an appropriate frequency vector
            // i.e. thread (t): frequency index, loops over (i): index of rx
            for (int t = threadIdx.x; t < numFreqs; t = t + blockDim.x){
                val = 0;
                for (int i = 0; i < ylen; i++){
                    val = val + d_freqMat[t*ylen + i] * s_rx[i];
                }
                // when val is complete, write it to shared mem array for storage first
                s_absVals[t] = abs(val); // cast to float when writing
            }
            
            __syncthreads(); // wait for all absVals to be written
                                
            // use the first thread to scan for the maximum
            if (threadIdx.x == 0){
                maxIdx = 0;
                maxVal = s_absVals[0];
                for (int i = 1; i < numFreqs; i++){
                    if (s_absVals[i] > maxVal){
                        maxIdx = i;
                        maxVal = s_absVals[i];
                    }
                }
                
                // and write directly to the global mem
                d_freqinds[shiftIdx] = maxIdx;
                d_xc[shiftIdx] = maxVal * maxVal / rxNormSq / yNormSq;
                // d_xc[shiftIdx] = rxNormSq; // cheating debug statement

            }
            
        }
        
    }
    ''','group_xcorr_kernelv2')


    class GroupXcorrGPU(GroupXcorr):
        def __init__(self, y: np.ndarray, starts: np.ndarray, lengths: np.ndarray, freqs: np.ndarray, fs: int):
            super().__init__(y, starts, lengths, freqs, fs)
            
            self.d_yconcat = cp.array(self.yconcat, cp.complex64)
            self.d_yconcatNormSq = cp.array(self.yconcatNormSq)
            self.d_freqMat = cp.array(self.freqMat, cp.complex64)
            self.d_freqs = cp.array(self.freqs)
            
        def xcorr(self, rx: np.ndarray, shifts: np.ndarray=None):
            if shifts is None:
                shifts = np.arange(len(rx)-(self.starts[-1]+self.lengths[-1])+1)
            else:
                assert(shifts[-1] + self.starts[-1] + self.lengths[-1] < rx.size) # make sure it can access it
                
            # Move to gpu
            d_rx = cp.array(rx)
                
            d_xc = cp.zeros(shifts.size, cp.float64)
            d_freqpeaks = cp.zeros(shifts.size, cp.float64)
            for i in np.arange(shifts.size):

                shift = shifts[i]
                # Perform slicing of rx
                d_rxconcat = cp.hstack([d_rx[shift + self.starts[g] : shift + self.starts[g] + self.lengths[g]] for g in np.arange(self.numGroups)])
                d_rxconcatNormSq = cp.linalg.norm(d_rxconcat)**2
                # Now multiply by y
                d_p = d_rxconcat * self.d_yconcat
                # And the freqmat
                d_pf = self.d_freqMat @ d_p
                # Pick the max value
                d_pfabs = cp.abs(d_pf)
                d_pmaxind = cp.argmax(d_pfabs)
                # Save output (with normalisations)
                d_xc[i] = d_pfabs[d_pmaxind]**2 / d_rxconcatNormSq / self.d_yconcatNormSq
                d_freqpeaks[i] = self.d_freqs[d_pmaxind]
                
            # Move to cpu
            xc = cp.asnumpy(d_xc)
            freqpeaks = cp.asnumpy(d_freqpeaks)
            
            return xc, freqpeaks
        
        def xcorrKernel(self, rx, shifts, numShiftsPerBlk=2, verbTiming=False):
            '''
            Experimental. Uses kernel (not fully optimized, but faster than the xcorr call).
            '''
            # Host-side computes
            nFreqs = self.freqs/self.fs
            gIdx = np.hstack([np.arange(self.starts[i], self.starts[i]+self.lengths[i]) for i in range(self.numGroups)])
            
            tg1 = time.time()
            # Assert shifts factor requirements
            assert(shifts.size % numShiftsPerBlk == 0)
            
            # Check shared memory requirements
            ylen = int(self.yconcat.size)
            numFreqs = int(nFreqs.size)
            d_nFreqs = cp.array(nFreqs, dtype=cp.float32)
            d_gIdx = cp.array(gIdx, dtype=cp.int32)

            d_rx = cp.array(rx, dtype=cp.complex64)
            d_shifts = cp.array(shifts, dtype=cp.int32)
            
            # smReq = int(2*ylen*8 + numFreqs*2*4 + ylen*4)
            # if(smReq > 48000): # Maximum 48000 shared memory bytes
            #     print("y + rx workspace (32fc) total: %d bytes." % (ylen*2*8))
            #     print("nFreqs + interrim workspace (32f) total: %d bytes." % (numFreqs*2*4))
            #     print("gIdx (32s) total: %d bytes." % (ylen*4))
            #     raise MemoryError("Shared memory requested exceeded 48kB.")
            
            # For v2, less SM required
            smReq = int(2*ylen*8 + numFreqs*4 + ylen*4)
            if(smReq > 48000): # Maximum 48000 shared memory bytes
                print("y + rx workspace (32fc) total: %d bytes." % (ylen*2*8))
                print("interrim workspace (32f) total: %d bytes." % (numFreqs*4))
                print("gIdx (32s) total: %d bytes." % (ylen*4))
                raise MemoryError("Shared memory requested exceeded 48kB.")
            
            # Allocate output
            d_xc = cp.zeros(shifts.size, dtype=cp.float32)
            d_freqinds = cp.zeros(shifts.size, dtype=np.int32)
            
            THREADS_PER_BLOCK = 128
            NUM_BLOCKS = int(np.round(shifts.size / numShiftsPerBlk))
            
            tg2 = time.time()
            # group_xcorr_kernel((NUM_BLOCKS,),(THREADS_PER_BLOCK,), 
            #                    (d_rx, 
            #                     self.d_yconcat, ylen, self.yconcatNormSq.astype(np.float32),
            #                     d_nFreqs, numFreqs,
            #                     d_gIdx, 
            #                     d_shifts, int(shifts.size), int(numShiftsPerBlk),
            #                     d_xc, d_freqinds),
            #                    shared_mem=smReq)
            
            group_xcorr_kernelv2((NUM_BLOCKS,),(THREADS_PER_BLOCK,), 
                               (d_rx, 
                                self.d_yconcat, ylen, self.yconcatNormSq.astype(np.float32),
                                self.d_freqMat, numFreqs,
                                d_gIdx, 
                                d_shifts, int(shifts.size), int(numShiftsPerBlk),
                                d_xc, d_freqinds),
                               shared_mem=smReq)
            
            cp.cuda.Stream.null.synchronize()
            tg3 = time.time()
            
            xc = cp.asnumpy(d_xc)
            freqinds = cp.asnumpy(d_freqinds)
        
            tg4 = time.time()
            
            if verbTiming:
                print("Prep time(includes transfers): %fs " %(tg2-tg1))
                print("Kernel runtime: %fs" % (tg3-tg2))
                print("Output conversion to CPU: %fs" % (tg4-tg3))
        
            return xc, freqinds

except:
    print("Ignoring cupy-related classes.")


#%% Cythonised classes
import os
if os.name == 'nt': # Load the directory on windows
    os.add_dll_directory(os.path.join(os.environ['IPPROOT'], 'redist', 'intel64')) # Configure IPP dll reliance

try: 
    from cython_ext.CyGroupXcorrFFT import CyGroupXcorrFFT
    
except Exception as e:
    print("Unable to load cythonised xcorrRoutines: %s" % (e))

# Import the cythonised fastXcorr
try:
    from cython_ext.CyIppXcorrFFT import CyIppXcorrFFT

except Exception as e:
    print("Unable to load cythonised xcorrRoutines: %s" % (e))

#%% Computational complexity estimation
from spectralRoutines import next_fast_len

def computeFastXcorrComplexity(
    N: int,
    K: int=1
):
    """
    Computes the complexity of the fast xcorr routines, like fastXcorr.
    This is a sliding window multiply -> FFT, so most of the cost is spent in the FFT.

    Parameters
    ----------
    N : int or np.ndarray
        Length of cutout i.e. length of FFT. Log2 complexity is assumed.
    K : int or np.ndarray
        Number of sample shifts i.e. number of FFTs used, since 1 FFT per sample shift.
    """
    return K * N * np.log2(N)

def computeGroupXcorrCZTcomplexity(
    m: int,
    L: int,
    n: int,
    K: int=1
):
    """
    Computes the complexity of the group xcorr routines using CZT, like groupXcorrCZT.
    This is a sliding window multiply -> CZT, so most of the cost is spent in the CZT.
    Note that the CZT incurs a cost of 2 FFTs, each of slightly longer length (usually)
    than the actual cutout.

    The cost of the phase corrections required for each group have been ignored here.

    Parameters
    ----------
    m : int
        Number of bursts/groups.
    L : int
        Length of each group in samples.
    n : int
        Number of points in the CZT. The complexity of the CZT is essentially 2 FFTs
        of length (CZT points + L)
    K : int
        Number of sample shifts i.e. number of CZTs used, since 1 CZT per sample shift.
    """
    Lc = next_fast_len(L + n)
    return K * m * 2 * Lc * np.log2(Lc)

    

