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
    def __init__(self, alphabet, pretransitions, pulses, omegas, up):
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
            if a != 0:
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
        
        # print("New paths:")
        # print(paths)
        # print("New pathmetrics")
        # print(pathmetrics)
        
    def genOmegaVectors(self, ylength):
        self.omegavectors = np.zeros((len(self.omegas),ylength),dtype=np.complex128)
        for i in range(len(self.omegas)):
            self.omegavectors[i] = np.exp(1j*(-self.omegas[i]*np.arange(ylength)))
               
    
        
        
        
# # DEBUG WORKSPACE
# vd = ViterbiDemodulator(alphabet, pretransitionIdx, pulses, omega_l, up)
# vd.calcBranchMetrics(y.flatten(), paths, 1)