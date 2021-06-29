# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 11:55:54 2021

@author: Seo
"""

# Note that this is faster when comparing purepython to purepython with the softvit function earlier,
# BUT ONLY with all numba disabled (remember to disable it for Fmat_direct_slice as well, which gets called)
# timings are ~0.35 for this, ~1.35 for the softvit i.e. this is faster for pure python, need to test numba

import numpy as np
import scipy as sp
import scipy.signal as sps
import matplotlib.pyplot as plt

class ViterbiDemodulator:
    '''
    Assumes constant global phase/amplitude information is already embedded in pulses.
    '''
    def __init__(self, alphabet, pretransitions, pulses, omegas, up, allowedStartIdx=np.array([0])):
        self.alphabet = alphabet
        self.alphabetlen = len(self.alphabet)
        self.pretransitions = pretransitions
        if len(self.alphabet) != self.pretransitions.shape[0]:
            raise ValueError("Number of transitions is inconsistent.")
        self.pulses = pulses
        self.pulselen = self.pulses.shape[1]
        self.omegas = omegas
        self.up = up
        self.L = len(self.omegas)
        if self.L != self.pulses.shape[0]:
            raise ValueError("Number of sources is inconsistent.")
            
        # Calculate the number of symbols the pulse extends over
        self.pulseLenInSyms = int(self.pulses.shape[1]/up)
        
        # Define the allowed starting indices
        self.allowedStartIdx = allowedStartIdx # only start from 0 for example
        
        print("Instantiated ViterbiDemodulator")
        
    def run(self, y, pathlen):
        if (y.ndim > 1):
            raise ValueError("Please flatten y before input.")
            
        # Pregenerate omega vectors
        self.genOmegaVectors(len(y))
        
        # Construct paths
        paths = np.zeros((self.alphabetlen, pathlen), dtype=self.alphabet.dtype)
        self.temppaths = np.zeros_like(paths)
        # Construct path metrics
        pathmetrics = np.zeros(self.alphabetlen, dtype=np.float64) + np.inf
        self.temppathmetrics = np.zeros_like(pathmetrics)
        
        # Construct the first symbol path metric
        for a in np.arange(self.alphabetlen):
            if a not in self.allowedStartIdx:
                continue
            paths[a,0] = self.alphabet[a]
            guess = paths[a]
            
            # KEEP IT SIMPLE FOR NOW, UPSAMPLE THE WHOLE PATH
            upguess = np.zeros(pathlen * self.up, dtype=paths.dtype)
            upguess[::self.up] = guess
            # print(upguess[:self.up*2])
            
            # Loop over all sources
            x_all = np.zeros((self.L, self.pulselen), dtype=np.complex128)
            for i in np.arange(self.L): 
                xc = np.convolve(self.pulses[i], upguess[:1])[-self.pulselen:]

                xcs = np.exp(1j*(-self.omegas[i]*np.arange(0*self.up,0*self.up+self.pulselen))) * xc
                # xcs = np.exp(1j*(-self.omegas[i]*np.arange(len(xc)))) * xc
                x_all[i,:] = xcs[-self.pulselen:]
                
            summed = np.sum(x_all, axis=0)
            
            # print("Writing to pathmetric[%d]" % (a))
            pathmetrics[a] = np.linalg.norm(y[0*self.up:1*self.up] - summed[:self.up])**2
            
        # print(pathmetrics)
        # print(paths)
        
        # Iterate over the rest of the symbols
        for n in np.arange(1, pathlen):
            # Calculate all branches
            branchmetrics, shortbranchmetrics = self.calcAllBranchMetrics(y, paths, pathmetrics, n)
            
            # print(branchmetrics)
            # print(shortbranchmetrics)
            
            # Extract and update best paths
            self.calcPathMetrics(shortbranchmetrics, branchmetrics, paths, pathmetrics, n)
            
            # if n == 20:
            #     break
            
            # # DEBUG
            # print("Branchmetrics")
            # print(branchmetrics)
            # print("Shortbranchmetrics")
            # print(shortbranchmetrics)
            # print("New paths:")
            # print(paths)
            # print("New pathmetrics")
            # print(pathmetrics)
        
            # print("--------------------------")
            
        # get best path
        bestPathIdx = np.argmin(pathmetrics)
        bestPath = paths[bestPathIdx,:]
        
        return bestPath, pathmetrics, paths
        
        
    def calcAllBranchMetrics(self, y, paths, pathmetrics, n):
        '''
        Calculate branches leading to next symbol at index n.
        '''
        
        if (y.ndim > 1):
            raise ValueError("Please flatten y before input.")
        
        # Path length
        pathlen = paths.shape[1]
        
        # Allocate branchmetrics
        branchmetrics = np.zeros(self.pretransitions.shape)
        shortbranchmetrics = np.zeros_like(branchmetrics)
        
        # Preallocate vectors
        guess = np.zeros(pathlen, dtype=paths.dtype)
        upguess = np.zeros(pathlen * self.up, dtype=paths.dtype)
        
        # Select the current symbol
        for p in np.arange(paths.shape[0]):
            # Select a valid pre-transition path
            for t in np.arange(len(self.pretransitions[p])):
                # if self.pretransitions[p,t] != 0: # DEBUG
                #     continue
                
                
                if pathmetrics[self.pretransitions[p,t]] == np.inf:
                    # print("Pretransition is inf, skipping!")
                    branchmetrics[p,t] = np.inf
                    shortbranchmetrics[p,t] = np.inf
                    continue
                
                # print("Alphabet %d->%d at index %d" % (self.pretransitions[p,t],p,n))
                
                # guess = np.copy(paths[self.pretransitions[p,t]]) # move this out of the loop without a copy, set values in here
                guess[:] = paths[self.pretransitions[p,t]] # like this
                guess[n] = self.alphabet[p]
                # print("Guess:")
                # print(guess)
                # KEEP IT SIMPLE FOR NOW, UPSAMPLE THE WHOLE PATH
                # upguess = np.zeros(pathlen * self.up, dtype=paths.dtype) # move this out of the loop and set values
                upguess[:] = 0 # zero out first
                upguess[::self.up] = guess
                # print(upguess[:n*self.up+1:self.up])
                # assert(np.all(upguess[::self.up] == guess))
                
                # Loop over all sources
                s = np.max([n*self.up - self.pulselen + 1,0])
                x_all = np.zeros((self.L, self.pulselen), dtype=np.complex128)
                for i in np.arange(self.L): 
                    
                    # # this is equivalent, as tested below
                    upguesspad = np.pad(upguess[s:n*self.up+1], (0,self.pulselen-1)) # pad zeros to pulselen-1
                    xc = sps.lfilter(self.pulses[i], 1, upguesspad)[-self.pulselen:]
                    
                    # # original
                    # xc2 = np.convolve(self.pulses[i], upguess[s:n*self.up+1])[-self.pulselen:]
                    # if (not np.all(xc==xc2)):
                    #     print("What$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                    
                    # xc = np.convolve(self.pulses[i], upguess[n*self.up-self.pulselen:n*self.up+1])[-self.pulselen:]
                    # xcs = np.exp(1j*(-self.omegas[i]*np.arange(n*self.up,n*self.up+self.pulselen))) * xc
                    xcs = self.omegavectors[i,n*self.up:n*self.up+self.pulselen] * xc
                    
                    x_all[i,:] = xcs
                    


                summed = np.sum(x_all, axis=0)
                
                # print("Writing to branchmetrics[%d,%d]" % (p,t))
                branchmetrics[p,t] = np.linalg.norm(y[self.up*n:self.up*n+self.pulselen] - summed)**2
                shortbranchmetrics[p,t] = np.linalg.norm(y[self.up*n:self.up*(n+1)] - summed[:self.up])**2
                
        # Complete
        return branchmetrics, shortbranchmetrics
        
    
    
    def calcPathMetrics(self, shortbranchmetrics, branchmetrics, paths, pathmetrics, n):

        self.temppaths[:,:] = paths[:,:]
        self.temppathmetrics[:] = pathmetrics[:]
        
        for p in np.arange(branchmetrics.shape[0]):

            if np.all(branchmetrics[p,:] == np.inf):
                self.temppathmetrics[p] = np.inf
                continue
            bestPrevIdx = np.argmin(branchmetrics[p,:])
            self.temppaths[p,:] = paths[self.pretransitions[p,bestPrevIdx],:] # copy the whole path over
            self.temppaths[p,n] = self.alphabet[p]
            self.temppathmetrics[p] = pathmetrics[self.pretransitions[p,bestPrevIdx]] + shortbranchmetrics[p,bestPrevIdx]
            

        paths[:,:] = self.temppaths[:,:]
        pathmetrics[:] = self.temppathmetrics[:]
        
        
        
    def genOmegaVectors(self, ylength):
        self.omegavectors = np.zeros((len(self.omegas),ylength),dtype=np.complex128)
        for i in range(len(self.omegas)):
            self.omegavectors[i] = np.exp(1j*(-self.omegas[i]*np.arange(ylength)))
               
    
        
#%% Derived classes
class BurstyViterbiDemodulator(ViterbiDemodulator):
    '''
    Assumes constant global phase/amplitude information is already embedded in pulses.
    Assumes periodic bursts with constant number of symbols per burst and constant
    number of guard symbol periods (i.e. blank for X number of baud periods) per burst.
    '''
    def __init__(self, alphabet, pretransitions, pulses, omegas, up, numBurstSyms, numGuardSyms, allowedStartIdx=None):
        '''
        Note that in this case, allowedStartIdx is checked for the beginning symbol of EACH burst, not just the first symbol.
        It now defaults to all allowed instead of just 0 due to this reason.
        '''
        if allowedStartIdx is None:
            print("Defaulting to fully allowed start indices.")
            allowedStartIdx = np.arange(len(alphabet))
        
        super().__init__(alphabet, pretransitions, pulses, omegas, up, allowedStartIdx) # identical
        self.numBurstSyms = numBurstSyms
        self.numGuardSyms = numGuardSyms
        self.numPeriodSyms = numBurstSyms + numGuardSyms
        
        # Create pretransitions for a new burst
        # Generally fully connected TOWARDS starting burst allowed idxes,
        # We use -1 to denote not allowed i.e. the starting index is not allowed
        self.newBurstPretransitions = np.array([np.arange(self.alphabetlen) 
                                                if j in self.allowedStartIdx else np.zeros(self.alphabetlen)-1 
                                                for j in range(self.alphabetlen)], dtype=np.int32)
    
    # Redefine this to include special skips during guard periods
    def run(self, y, pathlen):
        if (y.ndim > 1):
            raise ValueError("Please flatten y before input.")
            
        # Pregenerate omega vectors
        self.genOmegaVectors(len(y))
        
        # Construct paths
        paths = np.zeros((self.alphabetlen, pathlen), dtype=self.alphabet.dtype)
        self.temppaths = np.zeros_like(paths)
        # Construct path metrics
        pathmetrics = np.zeros(self.alphabetlen, dtype=np.float64) + np.inf
        self.temppathmetrics = np.zeros_like(pathmetrics)
        
        # Construct the first symbol path metric
        for a in np.arange(self.alphabetlen):
            if a not in self.allowedStartIdx:
                continue
            paths[a,0] = self.alphabet[a]
            guess = paths[a]
            
            # KEEP IT SIMPLE FOR NOW, UPSAMPLE THE WHOLE PATH
            upguess = np.zeros(pathlen * self.up, dtype=paths.dtype)
            upguess[::self.up] = guess
            # print(upguess[:self.up*2])
            
            # Loop over all sources
            x_all = np.zeros((self.L, self.pulselen), dtype=np.complex128)
            for i in np.arange(self.L): 
                xc = np.convolve(self.pulses[i], upguess[:1])[-self.pulselen:]

                xcs = np.exp(1j*(-self.omegas[i]*np.arange(0*self.up,0*self.up+self.pulselen))) * xc
                # xcs = np.exp(1j*(-self.omegas[i]*np.arange(len(xc)))) * xc
                x_all[i,:] = xcs[-self.pulselen:]
                
            summed = np.sum(x_all, axis=0)
            
            # print("Writing to pathmetric[%d]" % (a))
            pathmetrics[a] = np.linalg.norm(y[0*self.up:1*self.up] - summed[:self.up])**2
            
        # print(pathmetrics)
        # print(paths)
        
        # Iterate over the rest of the symbols
        for n in np.arange(1, pathlen):
            ## Check if in guard period
            if (n % self.numPeriodSyms >= self.numBurstSyms):
                continue # straight skip, all paths are frozen until next burst
            ## The effect of this is to maintain all possible paths with their associated ending symbols
            ## until the next burst. At the end of burst (i), it is likely that all paths have a valid metric
            ## and thus there will exist a path for every symbol in the alphabet.
            ## The goal is to link each one of these paths to the new burst (i+1), with the possible start symbols.
            ## Then, we fully connect each of these paths to the possible start symbols in the next burst.
            
            # Now check if it's a new burst, this is a special case, and we head straight to the path metrics
            if (n % self.numPeriodSyms == 0):
                newburstBranchmetrics, newburstShortBranchMetrics = self.calcNewBurstBranchMetrics(y, paths, pathmetrics, n)
                self.calcNewBurstPathMetrics(newburstShortBranchMetrics, newburstBranchmetrics, paths, pathmetrics, n)
                
                continue
            
            # Calculate all branches
            branchmetrics, shortbranchmetrics = self.calcAllBranchMetrics(y, paths, pathmetrics, n)
            
            
            # Extract and update best paths
            self.calcPathMetrics(shortbranchmetrics, branchmetrics, paths, pathmetrics, n)
            
            # if n == 20:
            #     break
            
            # # DEBUG
            # print("Branchmetrics")
            # print(branchmetrics)
            # print("Shortbranchmetrics")
            # print(shortbranchmetrics)
            # print("New paths:")
            # print(paths)
            # print("New pathmetrics")
            # print(pathmetrics)
        
            # print("--------------------------")
            
        # get best path
        bestPathIdx = np.argmin(pathmetrics)
        bestPath = paths[bestPathIdx,:]
        
        return bestPath, pathmetrics, paths
    
    def calcNewBurstBranchMetrics(self, y, paths, pathmetrics, n):
        # Path length
        pathlen = paths.shape[1]
        
        # Allocate branchmetrics
        branchmetrics = np.zeros(self.newBurstPretransitions.shape) + np.inf
        shortbranchmetrics = np.zeros_like(branchmetrics) + np.inf
        
        # Preallocate vectors
        guess = np.zeros(pathlen, dtype=paths.dtype)
        upguess = np.zeros(pathlen * self.up, dtype=paths.dtype)
        
        
        print("First symbol of next burst, n = %d" % (n))
        # Iterate over only the allowed start idxes
        for p in self.allowedStartIdx:

            # Now loop over the pre-transitions (default is all are possible)
            for t in np.arange(len(self.newBurstPretransitions[p])):
                print("Calculating for alphabet idx %d, from previous burst alphabet idx %d" % (p, self.newBurstPretransitions[p,t]))
                
                # As usual, check if the pre-transition has a valid path metric from the previous burst
                if pathmetrics[self.newBurstPretransitions[p,t]] == np.inf:
                    print("Skipped due to invalid pre-transition path metric")
                    branchmetrics[p,t] = np.inf
                    shortbranchmetrics[p,t] = np.inf
                    continue
                
                # As usual, form a guess now by copying the existing path
                guess[:] = paths[self.newBurstPretransitions[p,t]] # like this
                guess[n] = self.alphabet[p]
                
                print(guess[:n+1])

                # Upsample the guess
                upguess[:] = 0 # zero out first
                upguess[::self.up] = guess

                # Loop over all sources; but now in order to properly add a branch over the indices we skipped,
                # we must consider a longer section (see below, N = pulselen)
                #
                #  BURST 0              GUARD            BURST 1
                # | ...    | 0 ......              ...0 | n | 0..... 
                # |N-1 elem| numGuardSyms * up elem     | 1 | N-1 elem
                #
                
                # Calculate the upsampled guard len
                guardlen = self.numGuardSyms * self.up
                
                # We now start here, to include the guard period 0s
                s = np.max([(n-self.numGuardSyms)*self.up - self.pulselen + 1,0])
                x_all = np.zeros((self.L, guardlen + self.pulselen), dtype=np.complex128)
                
                # Convenience indexing for extraction with reference to original signal length
                extractionIdx = np.arange((n-self.numGuardSyms)*self.up, n*self.up + self.pulselen)
                shortextractionIdx = np.arange((n-self.numGuardSyms)*self.up, (n+1)*self.up)
                
                # Loop over sources
                for i in np.arange(self.L): 
                    
                    # As usual, extract from upguess and pad it
                    upguesspad = np.pad(upguess[s:n*self.up+1], (0,self.pulselen-1)) # pad zeros to pulselen-1
                    xc = sps.lfilter(self.pulses[i], 1, upguesspad)[-(self.pulselen + guardlen):]
                    
                    # And now we extract
                    xcs = self.omegavectors[i,extractionIdx] * xc
                    
                    x_all[i,:] = xcs
                    


                summed = np.sum(x_all, axis=0)
                
                # print("Writing to branchmetrics[%d,%d]" % (p,t))
                branchmetrics[p,t] = np.linalg.norm(y[extractionIdx] - summed)**2
                shortbranchmetrics[p,t] = np.linalg.norm(y[shortextractionIdx] - summed[:guardlen + self.up])**2
                
        # Complete
        print(branchmetrics)
        print(shortbranchmetrics)
        return branchmetrics, shortbranchmetrics
        
    def calcNewBurstPathMetrics(self, newburstShortBranchMetrics, newburstBranchmetrics, paths, pathmetrics, n):
        print("New burst path start selection")
        
        self.temppaths[:,:] = paths[:,:]
        self.temppathmetrics[:] = pathmetrics[:]
        
        # The newburstBranchmetrics will automatically contain information 
        # regarding the possible start indices
        for p in np.arange(newburstBranchmetrics.shape[0]): 

            if np.all(newburstBranchmetrics[p,:] == np.inf):
                print("Starting burst index %d not allowed " % (p))
                self.temppathmetrics[p] = np.inf
                continue
            bestPrevIdx = np.argmin(newburstBranchmetrics[p,:])
            self.temppaths[p,:] = paths[self.newBurstPretransitions[p,bestPrevIdx],:] # copy the whole path over
            self.temppaths[p,n] = self.alphabet[p]
            self.temppathmetrics[p] = pathmetrics[self.newBurstPretransitions[p,bestPrevIdx]] + newburstShortBranchMetrics[p,bestPrevIdx]
            
            print("Selected (oldburst) %d->%d (newburst)" % (self.newBurstPretransitions[p,bestPrevIdx], p))

        paths[:,:] = self.temppaths[:,:]
        pathmetrics[:] = self.temppathmetrics[:]
        
        print(paths[:,:n+1])
        print(pathmetrics)

   
#%% DEBUG WORKSPACE
# vd = ViterbiDemodulator(alphabet, pretransitionIdx, pulses, omega_l, up)
# vd.calcBranchMetrics(y.flatten(), paths, 1)

vd = BurstyViterbiDemodulator(alphabet, pretransitionIdx, pulses, omega_l, up, 71, 9, np.array([0,2]))
bestPath, pathmetrics, paths = vd.run(ynoise.flatten(), 8000)
