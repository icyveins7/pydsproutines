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

try:
    import cupy as cp
    
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
    # Create the freq array
    f_search = np.arange(f_searchMin, f_searchMax, cztStep)
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
            pdtczt = czt(pdt, f_search[0],f_search[-1]+cztStep/2, cztStep, fs)
            result[i,:] = np.abs(pdtczt)**2.0 / rxNormPartSq / cutoutNormSq
            
        return result, f_search
        
    else:
        raise NotImplementedError("To do..")
    

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
                pdtfft = sp.fft.fft(pdt)
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
                pdtfft = sp.fft.fft(pdt)
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
                pdtfft = sp.fft.fft(pdt)
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

def convertEffSNRtoQF2(effSNR):
    """For back-conversion."""
    return effSNR/(2 + effSNR)

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
        
        
#%%
class GroupXcorr:
    def __init__(self, y: np.ndarray, starts: np.ndarray, lengths: np.ndarray, freqs: np.ndarray, fs: int, autoConj: bool=True):
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
        '''
        assert(starts.size == lengths.size) # this is the number of groups
        
        self.starts = starts
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
                 f1: float, f2: float, binWidth: float, fs: int, autoConj: bool=True):
        
        assert(starts.size == lengths.size)
        self.starts = starts
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
        groupPhases = np.exp(1j*2*np.pi*cztFreq*self.starts.reshape((-1,1))/self.fs)
        
        for i in np.arange(shifts.size):
            shift = shifts[i]
            
            pdtcztPhased = np.zeros((self.numGroups,cztFreq.size), np.complex128)
            rxgroupNormSqCollect = np.zeros(self.numGroups)
            
            for g in np.arange(self.numGroups):
                ygroup = self.ystack[g,:]
                rxgroup = rx[shift+self.starts[g] : shift+self.starts[g]+self.lengths[g]]
                rxgroupNormSqCollect[g] = np.linalg.norm(rxgroup)**2
                pdt = ygroup * rxgroup
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
          
