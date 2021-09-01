# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:08:45 2021

@author: Seo
"""

import numpy as np
import scipy as sp
import scipy.signal as sps
import matplotlib.pyplot as plt
from signalCreationRoutines import *
from spectralRoutines import *

#%%
def musicAlg(x, freqlist, rows, plist, snapshotJump=None, fwdBwd=False,
             useSignalAsNumerator=False, averageToToeplitz=False, useAutoCorr=False):
    '''
    p (i.e. plist) is the dimensionality of the signal subspace. The function will return an array based on freqlist
    for every value of p in plist.
    
    x is expected as a 1-dim array (flattened). Alternatively, if x is passed in as a dictionary,
    each value in the dictionary is expected to be a 1-dim array; each 1-dim array is parsed into its own
    matrix using 'snapshotJump', and then the matrices are stacked horizontally to form a large data matrix.
    This is then used to estimate the covariance matrix. 
    Note: the 1-dim arrays do not need to be the same length,
    since they are processed using the snapshotJump parameter.
    
    snapshotJump is the index jump per column vector. By default this jump is equal to rows,
    i.e. each column vector is unique (matrix constructed via reshape), but may not resolve frequencies well.
    
    fwdBwd is a boolean which toggles the use of the Forward-Backward correction of the covariance matrix (default False).
    
    averageToToeplitz is a boolean which toggles averaging of the covariance matrix along each diagonal.
    This ensures a full rank matrix. Defaults to False.
    
    useAutoCorr is a boolean which toggles the autocorrelation method of creating the covariance matrix.
    Defaults to False.
    '''
    
    if not np.all(np.abs(freqlist) <= 1.0):
        raise ValueError("Frequency list input must be normalized.")

    # Autocorrelation method is fundamentally different; will not allow averageToToeplitz,
    # Forward-Backward, or snapshotJump options.
    if useAutoCorr:
        autocorr = sps.correlate(x, x)
        Rx = sp.linalg.toeplitz(autocorr[len(x)-1:len(x)-1+rows] / (len(x) - np.arange(rows)))

    # All other options revolve around first rearranging the input vector in a particular way
    else:
        if isinstance(x, dict): # If input is multiple 1-d arrays
            # Instantiate the final mat (empty at first)
            xs = np.zeros((rows,0), np.complex128)
            # Iterate over every 1-d array in the dictionary
            for xi in x.values():
                if snapshotJump is None: # Refer to the single-array version for comments
                    xi = xi.reshape((-1,1)) # vectorize
                    cols = int(np.floor(len(xi)/rows))
                    xslen = rows * cols
                    xs_i = xi[:xslen] # we cut off the ending bits
                    xs_i = xs_i.reshape((cols, rows)).T
                else: # Refer to the single-array version for comments 
                    if snapshotJump <= 0:
                        raise ValueError("snapshotJump must be at least 1.")
                        
                    xi = xi.flatten() # in case it's not flat
                    cols = (xi.size - rows) / snapshotJump # calculate the required columns
                    xs_i = np.zeros((rows, int(cols+1)), xi.dtype)
                    print("Matrix dim is %d, %d" % (xs.shape[0], xs.shape[1]))
                    for i in range(xs_i.shape[1]): # fill the columns
                        xs_i[:,i] = xi[i * snapshotJump : i * snapshotJump + rows]
                
                # Then, concatenate the matrix
                xs = np.hstack((xs, xs_i))
            
            
        else: # If input is just a single 1-d array
            if snapshotJump is None: # then we snapshot via slices
                # 1 2 3 4 5 6 ->
                # 1 3 5
                # 2 4 6  for example
                x = x.reshape((-1,1)) # vectorize
                cols = int(np.floor(len(x)/rows))
                xslen = rows * cols
                xs = x[:xslen] # we cut off the ending bits
                xs = xs.reshape((cols, rows)).T
            else: # we use the rate to populate our columns
                # e.g. for snapshot jump = 1,
                # 1 2 3 4 5 6 ->
                # 1 2 3 4 5
                # 2 3 4 5 6
                if snapshotJump <= 0:
                    raise ValueError("snapshotJump must be at least 1.")
                    
                x = x.flatten() # in case it's not flat
                cols = (x.size - rows) / snapshotJump # calculate the required columns
                xs = np.zeros((rows, int(cols+1)), x.dtype)
                print("Matrix dim is %d, %d" % (xs.shape[0], xs.shape[1]))
                for i in range(xs.shape[1]): # fill the columns
                    xs[:,i] = x[i * snapshotJump : i * snapshotJump + rows]
                
        Rx = (1/cols) * xs @ xs.conj().T
        if fwdBwd:
            J = np.eye(Rx.shape[0])
            # Reverse row-wise to form the antidiagonal exchange matrix
            J = J[:,::-1]
            Rx = 0.5 * (Rx + J @ Rx.T @ J)
            print("Using forward-backward covariance.")
        
        if averageToToeplitz:
            diagIdx = np.arange(-Rx.shape[0]+1, Rx.shape[1])
            
            Rx_tp = np.zeros_like(Rx)
            for k in diagIdx:
                diagArr = np.zeros(Rx.shape[0]-np.abs(k), Rx.dtype) + np.mean(np.diag(Rx,k))
                diagMat = np.diag(diagArr, k) # make it into a matrix again
                Rx_tp = Rx_tp + diagMat
            
    
    u, s, vh = np.linalg.svd(Rx)
    
    # Instead of iterations, generate a one-time Vandermonde matrix of eh
    ehlist = np.exp(-1j*2*np.pi*freqlist.reshape((-1,1))*np.arange(rows)) # generate the e vector for every frequency i.e. Vandermonde
    
    # Generate output
    numerator = 1.0 # default numerator
    if not hasattr(plist, '__len__'): # if only one value of p
        d = ehlist @ u[:,plist:]
        denom = np.sum(np.abs(d)**2, axis=1)
        
        if useSignalAsNumerator: # Construct the generalised inverse (we have the SVD results already)
            ssp = s[:plist]**-0.5 # Take the p eigenvalues and reciprocal + root them -> generalised inverse root eigenvalues
            siginv = u[:,:plist] * ssp # assume u = v, i.e. u^H = v^H, scale by the inverse eigenvalues
            n = ehlist @ siginv
            numerator = np.sum(np.abs(n)**2, axis=1)
            
        f = numerator / denom
        
    else: # for multiple values of p
        f = np.zeros((len(plist), len(freqlist)))
        for i in range(len(plist)):
            p = plist[i]
            d = ehlist @ u[:,p:]
            denom = np.sum(np.abs(d)**2, axis=1)
            
            if useSignalAsNumerator:
                ssp = s[:p]**-0.5 # Take the p eigenvalues and reciprocal + root them -> generalised inverse root eigenvalues
                siginv = u[:,:p] * ssp # assume u = v, i.e. u^H = v^H, scale by the inverse eigenvalues
                n = ehlist @ siginv
                numerator = np.sum(np.abs(n)**2, axis=1)
                
            f[i,:] = numerator / denom
        
    return f, u, s, vh

#############################
class CovarianceTechnique:
    def __init__(self, rows, snapshotJump=None, fwdBwd=False, avgToToeplitz=False, useEigh=False):
        '''
        snapshotJump is the index jump per column vector. By default this jump is equal to rows,
        i.e. each column vector is unique (matrix constructed via reshape), but may not resolve frequencies well.
        
        fwdBwd is a boolean which toggles the use of the Forward-Backward correction of the covariance matrix (default False).
        
        averageToToeplitz is a boolean which toggles averaging of the covariance matrix along each diagonal.
        This ensures a full rank matrix. Defaults to False.
        '''
        if snapshotJump is not None and snapshotJump <= 0:
            raise ValueError("snapshotJump must be at least 1.")
        
        self.rows = rows
        self.snapshotJump = snapshotJump
        self.fwdBwd = fwdBwd
        self.avgToToeplitz = avgToToeplitz
        self.useEigh = useEigh
        self.L = None
        
    def setPrewhiteningMatrix(self, L):
        self.L = L
        
    def preprocessSnapshots(self, x):
        '''
        Helper method to slice dictionary of arrays into a matrix of column vectors.
        Used in covariance matrix generation.
        '''
        if isinstance(x, dict): # If input is multiple 1-d arrays
            # Instantiate the final mat (empty at first)
            xs = np.zeros((self.rows,0), np.complex128)
            # Iterate over every 1-d array in the dictionary
            for xi in x.values():
                if self.snapshotJump is None: # Refer to the single-array version for comments
                    xi = xi.reshape((-1,1)) # vectorize
                    cols = int(np.floor(len(xi)/self.rows))
                    xslen = self.rows * cols
                    xs_i = xi[:xslen] # we cut off the ending bits
                    xs_i = xs_i.reshape((cols, self.rows)).T
                else: # Refer to the single-array version for comments 
    
                    xi = xi.flatten() # in case it's not flat
                    cols = (xi.size - self.rows) / self.snapshotJump # calculate the required columns
                    xs_i = np.zeros((self.rows, int(cols+1)), xi.dtype)
                    print("Matrix dim is %d, %d" % (xs.shape[0], xs.shape[1]))
                    for i in range(xs_i.shape[1]): # fill the columns
                        xs_i[:,i] = xi[i * self.snapshotJump : i * self.snapshotJump + self.rows]
                
                # Then, concatenate the matrix
                xs = np.hstack((xs, xs_i))
            
            
        else: # If input is just a single 1-d array
            if self.snapshotJump is None: # then we snapshot via slices
                # 1 2 3 4 5 6 ->
                # 1 3 5
                # 2 4 6  for example
                x = x.reshape((-1,1)) # vectorize
                cols = int(np.floor(len(x)/self.rows))
                xslen = self.rows * cols
                xs = x[:xslen] # we cut off the ending bits
                xs = xs.reshape((cols, self.rows)).T
            else: # we use the rate to populate our columns
                # e.g. for snapshot jump = 1,
                # 1 2 3 4 5 6 ->
                # 1 2 3 4 5
                # 2 3 4 5 6
                    
                x = x.flatten() # in case it's not flat
                cols = (x.size - self.rows) / self.snapshotJump # calculate the required columns
                xs = np.zeros((self.rows, int(cols+1)), x.dtype)
                print("Matrix dim is %d, %d" % (xs.shape[0], xs.shape[1]))
                for i in range(xs.shape[1]): # fill the columns
                    xs[:,i] = x[i * self.snapshotJump : i * self.snapshotJump + self.rows]
        
        
        Rx = (1/cols) * xs @ xs.conj().T
        
        return Rx
        
        
    def estPrewhiteningMatrix(self, noise, removeUncorrelated=False):
        # Similar to RX, we calculate covariance for (coloured) noise
        Rn = self.preprocessSnapshots(noise)
        
        if removeUncorrelated:
            u,s,vh = np.linalg.svd(Rn)
            Rn = Rn - s[-1] * np.eye(self.rows) # assumes smallest eigenvalue = white noise power
            
        self.L = np.linalg.cholesky(Rn)
        # Complete
            
            
    def calcRx(self, x, findEigs=True):
        '''
        Parameters
        ----------
        x : 1-dim array.
            Input column vector (will be automatically reshaped).
        findEigs : boolean, optional
            Toggles whether svd is called. The default is True.
        '''
        
        # Get the base Rx
        t1 = time.time()
        Rx = self.preprocessSnapshots(x)
        t2 = time.time()
        print("Took %fs for matrix snapshot reshaping." % (t2-t1))
       
        if self.fwdBwd:
            J = np.eye(Rx.shape[0])
            # Reverse row-wise to form the antidiagonal exchange matrix
            J = J[:,::-1]
            Rx = 0.5 * (Rx + J @ Rx.T @ J)
            print("Using forward-backward covariance.")
        
        if self.avgToToeplitz:
            diagIdx = np.arange(-Rx.shape[0]+1, Rx.shape[1])
            
            Rx_tp = np.zeros_like(Rx)
            for k in diagIdx:
                diagArr = np.zeros(Rx.shape[0]-np.abs(k), Rx.dtype) + np.mean(np.diag(Rx,k))
                diagMat = np.diag(diagArr, k) # make it into a matrix again
                Rx_tp = Rx_tp + diagMat
                
            Rx = Rx_tp
            
        # MUSIC/ESPRIT
        if findEigs is True:
            t3 = time.time()
            if self.useEigh:
                s, u = np.linalg.eigh(Rx) # this should be equivalent (minus some numerical errors?)
                # sort it backwards because eigh returns it in ascending order
                s = s[::-1]
                u = u[:,::-1]
                vh = None # 
            else:
                u, s, vh = np.linalg.svd(Rx)
            t4 = time.time()
            print("Completed SVD/EIG in %fs." % (t4-t3))

            return u, s, vh, Rx
        # CAPON
        else:
            return Rx
    
class MUSIC(CovarianceTechnique):
    def __init__(self, rows, snapshotJump=None, fwdBwd=False, avgToToeplitz=False, useEigh=False):
        super().__init__(rows, snapshotJump, fwdBwd, avgToToeplitz, useEigh)
        
    def run(self, x, freqlist, plist, useSignalAsNumerator=False, prewhiten=False):
        '''
        Parameters
        ----------
        x : 1-dim array or dictionary of 1-dim arrays.
            Input array/dictionary. If x is passed in as a dictionary,
            each value in the dictionary is expected to be a 1-dim array; each 1-dim array is parsed into its own
            matrix using 'snapshotJump', and then the matrices are stacked horizontally to form a large data matrix.
            This is then used to estimate the covariance matrix. 
            Note: the 1-dim arrays do not need to be the same length,
            since they are processed using the snapshotJump parameter.
        freqlist : 1-dim array or list
            List of freq values to calculate pseudospectrum at.
        plist : scalar/1-dim array or list
            p (i.e. plist) is the dimensionality of the signal subspace. The function will return an array based on freqlist
            for every value of p in plist.
        useSignalAsNumerator : bool, optional
            Toggles the use of the signal subspace as the numerator. The default is False.

        Returns
        -------
        f : 1-dim array or 2-d matrix.
            Pseudospectrum values. If plist has multiple values, each row corresponds to the pseudospectrum based on that p value.
        u : Matrix
            Output from SVD.
        s : Matrix
            Output from SVD.
        vh : Matrix
            Output from SVD.

        '''
        if prewhiten and self.L is None:
            raise ValueError("Please set the pre-whitening matrix explicitly using setPrewhiteningMatrix or use estPrewhiteningMatrix to estimate it from some noise.")
        
        u, s, vh, Rx = self.calcRx(x)
        
        if prewhiten:
            Linv = np.linalg.inv(self.L)
            Rx = Linv @ Rx @ Linv.conj().T
        
        # Instead of iterations, generate a one-time Vandermonde matrix of eh
        t1 = time.time()
        ehlist = np.exp(-1j*2*np.pi*freqlist.reshape((-1,1))*np.arange(self.rows)) # generate the e vector for every frequency i.e. Vandermonde
        t2 = time.time()
        print("%fs to generate Vandermonde tone matrix." % (t2-t1))
        
        # Generate output
        numerator = 1.0 # default numerator
        if not hasattr(plist, '__len__'): # if only one value of p
            d = ehlist @ u[:,plist:]
            denom = np.sum(np.abs(d)**2, axis=1)
            
            if useSignalAsNumerator: # Construct the generalised inverse (we have the SVD results already)
                ssp = s[:plist]**-0.5 # Take the p eigenvalues and reciprocal + root them -> generalised inverse root eigenvalues
                siginv = u[:,:plist] * ssp # assume u = v, i.e. u^H = v^H, scale by the inverse eigenvalues
                n = ehlist @ siginv
                numerator = np.sum(np.abs(n)**2, axis=1)
                
            f = numerator / denom
            
        else: # for multiple values of p
            f = np.zeros((len(plist), len(freqlist)))
            for i in range(len(plist)):           
                p = plist[i]
                d = ehlist @ u[:,p:]
                denom = np.sum(np.abs(d)**2, axis=1)
                
                if useSignalAsNumerator:
                    ssp = s[:p]**-0.5 # Take the p eigenvalues and reciprocal + root them -> generalised inverse root eigenvalues
                    siginv = u[:,:p] * ssp # assume u = v, i.e. u^H = v^H, scale by the inverse eigenvalues
                    n = ehlist @ siginv
                    numerator = np.sum(np.abs(n)**2, axis=1)
                    
                f[i,:] = numerator / denom
                
        t3 = time.time()
        print("%fs to generate output." % (t3-t2))
        
        # Return
        return f,u,s,vh,Rx
        
class CAPON(CovarianceTechnique):
    def __init__(self, rows, snapshotJump=None, fwdBwd=False, avgToToeplitz=False, useEigh=False):
        super().__init__(rows, snapshotJump, fwdBwd, avgToToeplitz, useEigh)
        
    def run(self, x, freqlist):
        Rx = self.calcRx(x, findEigs=False)
        invRx = np.linalg.inv(Rx)
        
        # Overtly, no way to split Rx so must iterate over every frequency
        # May be possible if use SVD and then root the eigenvalues?

        # Generate output
        numerator = 1.0 # default numerator

        f = np.zeros(freqlist.size, x.dtype)
        
        for i in range(len(freqlist)):
            eh = np.exp(-1j*2*np.pi*freqlist[i]*np.arange(self.rows)).reshape((1,-1))
            denom = eh @ invRx @ eh.conj().T
            f[i] = numerator/denom

        # Return
        return f,Rx
        
class ESPRIT(CovarianceTechnique):
    def __init__(self, rows, snapshotJump=None, fwdBwd=False, avgToToeplitz=False, useEigh=False):
        super().__init__(rows, snapshotJump, fwdBwd, avgToToeplitz, useEigh)
        
    def run(self, x, plist, fs):
        u,s,vh,Rx = self.calcRx(x)
        
        
        # Generate output
        numerator = 1.0 # default numerator
        if not hasattr(plist, '__len__'): # if only one value of p
            sigU = u[:,:plist]

            phi, residuals, rank, singularVals = np.linalg.lstsq(sigU[:self.rows-1,:], sigU[1:,:], rcond=None) # suppress warning with rcond specification
            w, v = np.linalg.eig(phi)
            normomegas = np.angle(w) # our version doesn't have minus?
            freqs = normomegas / (2*np.pi) * fs
            

        
        #     d = ehlist @ u[:,plist:]
        #     denom = np.sum(np.abs(d)**2, axis=1)
            
        #     if useSignalAsNumerator: # Construct the generalised inverse (we have the SVD results already)
        #         ssp = s[:plist]**-0.5 # Take the p eigenvalues and reciprocal + root them -> generalised inverse root eigenvalues
        #         siginv = u[:,:plist] * ssp # assume u = v, i.e. u^H = v^H, scale by the inverse eigenvalues
        #         n = ehlist @ siginv
        #         numerator = np.sum(np.abs(n)**2, axis=1)
                
        #     f = numerator / denom
            
        # else: # for multiple values of p
        #     f = np.zeros((len(plist), len(freqlist)))
        #     for i in range(len(plist)):
        #         p = plist[i]
        #         d = ehlist @ u[:,p:]
        #         denom = np.sum(np.abs(d)**2, axis=1)
                
        #         if useSignalAsNumerator:
        #             ssp = s[:p]**-0.5 # Take the p eigenvalues and reciprocal + root them -> generalised inverse root eigenvalues
        #             siginv = u[:,:p] * ssp # assume u = v, i.e. u^H = v^H, scale by the inverse eigenvalues
        #             n = ehlist @ siginv
        #             numerator = np.sum(np.abs(n)**2, axis=1)
                    
        #         f[i,:] = numerator / denom
                
        # Return
        return freqs,u,s,vh,Rx
        

#%%
if __name__ == '__main__':
    #%%
    plt.close("all")
    fs = 1e5
    length = 0.1*fs # at 0.1s, 20Hz sinc span
    fdiff = 18
    f0 = -40
    padding = 1000
    # f_true = [f0, f0+fdiff, f0+fdiff*3.5, f0+fdiff*5] # Arbitrary
    # f_true = f0 + np.arange(4) * fdiff # Uniform
    # f_true = f0 + np.arange(0,10) * fdiff # Single many-tet (is it possible to resolve if we are beyond the sinc spacing?)
    f_true = f0 + np.arange(0,7) * fdiff # Single septet (~18 Hz min?)
    # f_true = f0 + np.arange(0,6) * fdiff # Single sextet (~15 Hz min ? hard to say at this point)
    # f_true = f0 + np.arange(0,5) * fdiff # Single quintet (~12 Hz min)
    # f_true = f0 + np.arange(0,4) * fdiff # Single quartet (~9 Hz min)
    # f_true = f0 + np.hstack((np.arange(0,3),np.arange(14,17))) * fdiff # 2 triplets
    # f_true = f0 + np.arange(0,3) * fdiff # Single triplet (~6 Hz min)
    # f_true = f0 + np.hstack((np.arange(0,2),np.arange(4,6),np.arange(10,12))) * fdiff # 3 Pairs
    # f_true = f0 + np.hstack((np.arange(0,2),np.arange(5,7))) * fdiff # 2 Pairs
    # f_true = f0 + np.arange(0,2) * fdiff # Single pair (~3 Hz min)
    # f_true = [f0] # Single tone
    numTones = len(f_true)
    # Add tones
    x = np.zeros(int(length + padding),dtype=np.complex128)
    for i in range(numTones):
        x = x + np.pad(np.exp(1j*2*np.pi*f_true[i]*np.arange(length)/fs + 1j*np.random.rand()*2*np.pi), (0,padding))
    # Add some noise
    noisePwr = 1e-1
    xn = x + (np.random.randn(x.size) + np.random.randn(x.size)*1j) * np.sqrt(noisePwr)

    # Calculate CZT
    fineFreqStep = 0.1
    fineFreqRange = 30 # peg to the freqoffset
    fineFreqVec = np.arange(np.min(f_true)-fineFreqRange,np.max(f_true)+fineFreqRange + 0.1*fineFreqStep, fineFreqStep)
    xczt = czt(xn, np.min(f_true)-fineFreqRange,np.max(f_true)+fineFreqRange, fineFreqStep, fs)
    
    freqlist = np.arange(np.min(f_true)-fdiff*1,np.max(f_true)+fdiff*1,0.1)
    
    #%% One-shot evaluation for all desired p values
    onlyDoFilteredVersion = True
    
    if not onlyDoFilteredVersion:
        plist = np.arange(len(f_true)-1,len(f_true)+1)
        rows = 1000
        music = MUSIC(rows, snapshotJump=1, useEigh=False)
        t1 = time.time()
        # Standard
        f, u, s, vh, Rx = music.run(xn, freqlist/fs, plist)
    
        # Forward-Backward
        music.fwdBwd = True
        f_fb, u_fb, s_fb, vh_fb, Rx_fb = music.run(xn, freqlist/fs, plist)
    
        # Forward-Backward with Signal Subspace
        f_fb_ns, u_fb_ns, s_fb_ns, vh_fb_ns, Rx_fb_ns = music.run(xn, freqlist/fs, plist, useSignalAsNumerator=True)
    
        # # Capon
        # capon = CAPON(rows, snapshotJump=1)
        # f_c, Rx_c = capon.run(x, freqlist/fs)
        
        # # Forward-Backward Capon?
        # capon.fwdBwd = True
        # f_cfb, Rx_cfb = capon.run(x, freqlist/fs)
        
        # ESPRIT (improves on resolution from MUSIC, but still fails at 1e-1 with 3 tones..)
        esprit = ESPRIT(rows, snapshotJump=1)
        pe = numTones
        freqse,ue,se,vhe,Rxe = esprit.run(xn, pe, fs)
        
        # ESPRIT + Forward-Backward
        esprit.fwdBwd = True
        freqse_fb,ue_fb,se_fb,vhe_fb,Rxe_fb = esprit.run(xn, pe, fs)
    
        t2 = time.time()
        print("Took %f s." % (t2-t1))
    
        fig,ax = plt.subplots(4,1,num="Comparison")
        ax[0].set_title("Standard MUSIC, %d rows" % (rows))
        # ax[1].set_title("CAPON, %d rows" % (rows))
        ax[1].set_title("ESPRIT, %d rows" % (rows))
        ax[2].set_title("Forward-Backward MUSIC, %d rows" % (rows))
        ax[3].set_title("Forward-Backward MUSIC + Signal Subspace, %d rows" % (rows))
        
        for i in range(len(ax)):
            # plt.plot(makeFreq(len(x),fs), np.abs(xfft)/np.max(np.abs(xfft)))
            ax[i].plot(fineFreqVec, np.abs(xczt)/ np.max(np.abs(xczt)), label='CZT')
            ax[i].vlines(f_true,0,1,colors='r', linestyles='dashed',label='Actual')
            
        # Plot capon out of loop (usually garbage compared to music)
        # ax[1].plot(freqlist, np.abs(f_c)/np.max(np.abs(f_c)), label='CAPON')
        # ax[1].plot(freqlist, np.abs(f_cfb)/np.max(np.abs(f_cfb)), label='CAPON + Forward-Backward')
        # ax[1].legend()
        # ax[1].set_xlim([fineFreqVec[0],fineFreqVec[-1]])
        
        # Plot ESPRIT out of loop
        ax[1].vlines(freqse, 0, 1, colors='b', label='ESPRIT, p='+str(pe))
        ax[1].vlines(freqse_fb, 0, 1, colors='g', label='ESPRIT+ForwardBackward, p='+str(pe))
        ax[1].legend()
        ax[1].set_xlim([fineFreqVec[0],fineFreqVec[-1]])
       
        for i in range(f.shape[0]):
            ax[0].plot(freqlist, f[i]/np.max(f[i]), label='MUSIC, p='+str(plist[i]))
            ax[0].legend()
            ax[0].set_xlim([fineFreqVec[0],fineFreqVec[-1]])
            
            
            
            ax[2].plot(freqlist, f_fb[i]/np.max(f_fb[i]), label='MUSIC, p='+str(plist[i]))
            ax[2].legend()
            ax[2].set_xlim([fineFreqVec[0],fineFreqVec[-1]])
            
            ax[3].plot(freqlist, f_fb_ns[i]/np.max(f_fb_ns[i]), label='MUSIC, p='+str(plist[i]))
            ax[3].legend()
            ax[3].set_xlim([fineFreqVec[0],fineFreqVec[-1]])
            
    
        plt.figure("Eigenvalues")
        plt.plot(np.log10(s),'x-',label='Standard MUSIC')
        # plt.plot(np.log10(s_tp),'x-',label='Averaged Toeplitz MUSIC')
        # plt.plot(np.log10(s_ac),'x-',label='Auto-correlation MUSIC')
        plt.plot(np.log10(s_fb),'x-',label='Forward-Backward MUSIC')
        plt.plot(np.log10(s_fb_ns),'x-',label='Forward-Backward + Signal Subspace MUSIC')
        plt.plot(np.log10(se),'x-',label='ESPRIT')
        plt.plot(np.log10(se_fb),'x-',label='Forward-Backward ESPRIT')
        plt.legend()

    #%% Experiment with filtering
    alsoPrewhiten = False
    
    dsr = 100
    ftap = sps.firwin(1000, 1/dsr)
    xn_filt = sps.lfilter(ftap,1,xn)
    xn_filtds = xn_filt[int(len(ftap)/2):int(len(ftap)/2+length):dsr]
    
    figds, axds = plt.subplots(2,1,num="Filter+Downsample")

    xn_filtdsczt = czt(xn_filtds, np.min(f_true)-fineFreqRange,np.max(f_true)+fineFreqRange, fineFreqStep, fs/dsr)
    # ax[0].plot(fineFreqVec,np.abs(xczt)/np.max(np.abs(xczt)), label='CZT')
    axds[0].plot(fineFreqVec, np.abs(xn_filtdsczt)/np.max(np.abs(xn_filtdsczt)), label='Filter+Downsample CZT')
    
    # Now run music on it
    dsrows = 90
    plist =  [len(f_true)] # [len(f_true), int(dsrows/2)] # although there's another knee in the eigenvalues, not worth using
    musicds = MUSIC(dsrows, snapshotJump=1, fwdBwd=True)
    fds, uds, sds, vhds, Rxds = musicds.run(xn_filtds, freqlist/(fs/dsr), plist, useSignalAsNumerator=True)
    for i in range(len(plist)):
        axds[0].plot(freqlist, fds[i]/np.max(fds[i]), label='MUSIC, p='+str(plist[i]))
        
    # Actually, no reason to ignore the other downsample phases
    xn_filtdsdict = {}
    for i in range(dsr):
        xn_filtdsdict[i] = xn_filt[i::dsr]
        
    fdsd, udsd, sdsd, vhdsd, Rxdsd = musicds.run(xn_filtdsdict, freqlist/(fs/dsr), plist, useSignalAsNumerator=True)
    for i in range(len(plist)):
        axds[0].plot(freqlist, fdsd[i]/np.max(fdsd[i]), label='MUSIC, all downsample phases, p='+str(plist[i]))
        
        # Detect peaks simply
        peakinds, peakprops = sps.find_peaks(fdsd[i])
        axds[0].vlines(freqlist[peakinds], 0,1, colors='k', label='Detected, total '+str(len(peakinds)))
        minarg = np.argmin(fdsd[i,peakinds])
        print("Lowest peak is at %g Hz, norm val = %f" % (freqlist[peakinds[minarg]], fdsd[i,peakinds[minarg]]))
    # It's just plain better now, obviously
    
    # # Esprit as well for good measure (but this is really bad now, maybe because different downsample phases are not rotationally invariant?)
    # espritds = ESPRIT(dsrows, snapshotJump=1)
    # freqseds,ueds,seds,vheds,Rxeds = espritds.run(xn_filtdsdict, plist[0], fs/dsr)
    # axds[0].vlines(freqseds, 0, 1, colors='k', label='ESPRIT')
        
    axds[0].vlines(f_true,0,1,colors='r', linestyles='dashed',label='Actual')
    axds[0].legend()
    axds[0].set_xlim([freqlist[0],freqlist[-1]])
    

    axds[1].plot(np.log10(sds), 'x-', label='Not prewhitened')
    axds[1].set_title("Eigenvalues")
    
    
    
    #%% Experiment with filtering and pre-whitening (insignificant due to downsample=>quite white already)
    if alsoPrewhiten:
        longnoiselen = int(0.1*fs)
        longnoise = (np.random.randn(longnoiselen) + np.random.randn(longnoiselen)*1j) * np.sqrt(noisePwr)
        filterednoise = sps.lfilter(ftap,1,longnoise)
        filtnoisedict = {i: filterednoise[i::dsr] for i in range(dsr)}
        # Call the object method to calculate and set prewhitening matrix
        musicds.estPrewhiteningMatrix(filtnoisedict)
        
        # Now call the music method with prewhitening?
        fdsdpw, udsdpw, sdsdpw, vhdsdpw, Rxdsdpw = musicds.run(xn_filtdsdict, freqlist/(fs/dsr), plist, useSignalAsNumerator=True, prewhiten=True)
        for i in range(len(plist)):
            axds[0].plot(freqlist, fdsdpw[i]/np.max(fdsd[i]), label='MUSIC, all downsample phases+prewhiten, p='+str(plist[i]))
        axds[0].legend()
            
        axds[1].plot(np.log10(sdsdpw), 'x-', label='Prewhitened')
        axds[1].legend()
        
    assert(False)
    
    #%% Experiment with separated bursts
    numBursts = 10
    burstLength = int(fs * 0.1)
    burstGap = int(fs * 0.9)
    totalLength = numBursts * burstLength + (numBursts-1) * burstGap
    noisePwr = 1e-1
    
    # Add tones
    x = np.zeros(int(totalLength),dtype=np.complex128)
    for i in range(numTones):
        x = x + np.exp(1j*2*np.pi*f_true[i]*np.arange(totalLength)/fs + 1j*np.random.rand()*2*np.pi)
    
    # Window out the bursts only
    xwindow = np.zeros(len(x))
    for i in range(numBursts):
        xwindow[i*(burstLength+burstGap) : i*(burstLength+burstGap)+burstLength] = 1
    # plt.figure()
    # plt.plot(xwindow)
    
    # Create the bursty signal    
    x = x * xwindow + (np.random.randn(len(x)) + 1j*np.random.randn(len(x))) * np.sqrt(noisePwr)
    
    # Slice the bursts for easier access later
    xbursts = {}
    for i in range(numBursts):
        xbursts[i] = x[i*(burstLength+burstGap) : i*(burstLength+burstGap)+burstLength]
    
    # Create the overall czt, as well as the individual czts
    xczt = czt(x, np.min(f_true)-fineFreqRange,np.max(f_true)+fineFreqRange, fineFreqStep, fs)
    xcztbursts = np.zeros((numBursts, fineFreqVec.size), x.dtype)
    for i in range(numBursts):
        xcztbursts[i,:] = czt(xbursts[i],
                              np.min(f_true)-fineFreqRange,np.max(f_true)+fineFreqRange, fineFreqStep, fs)
    # Compare the MUSICs
    bfig, bax = plt.subplots(1,1,num="Bursty Tone MUSIC")
    bax.vlines(f_true,0,1,colors='r', linestyles='dashed',label='Actual')

    # MUSIC each burst individually
    musicBursts = {}
    p = numTones
    for i in range(numBursts):
        f_fb_ns, u_fb_ns, s_fb_ns, vh_fb_ns = musicAlg(xbursts[i], freqlist/fs, rows, p, snapshotJump=1, fwdBwd=True, useSignalAsNumerator=True)
        musicBursts[i] = f_fb_ns
        bax.plot(freqlist, musicBursts[i]/np.max(musicBursts[i]), label='MUSIC, p='+str(p)+', burst ' + str(i))
    
    # Now MUSIC both bursts as a single process
    f_fb_ns_stitched, u_fb_ns_stitched, s_fb_ns_stitched, vh_fb_ns_stitched = musicAlg(xbursts, freqlist/fs, rows, p, snapshotJump=1, fwdBwd=True, useSignalAsNumerator=True)
    bax.plot(freqlist, f_fb_ns_stitched/np.max(f_fb_ns_stitched), '.-', label='MUSIC, p='+str(p)+', bursts combined')
    
    bax.legend()
    
    # Compare Full CZT against Full MUSIC
    plt.figure()
    plt.vlines(f_true,0,1,colors='r', linestyles='dashed',label='Actual')
    plt.plot(fineFreqVec, np.abs(xczt)/np.max(np.abs(xczt)), label="Full length CZT")
    # for i in range(numBursts):
    #     plt.plot(fineFreqVec, np.abs(xcztbursts[i,:])/np.max(np.abs(xcztbursts[i,:])), label="CZT of Burst " + str(i))
    plt.plot(freqlist, f_fb_ns_stitched/np.max(f_fb_ns_stitched), label='MUSIC, p='+str(p)+', bursts combined')
    plt.title("%d bursts" % (numBursts))
    plt.legend()
    
