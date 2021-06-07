# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 11:55:54 2021

@author: Seo
"""

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
        # Construct path metrics
        pathmetrics = np.zeros(self.alphabetlen, dtype=np.float64) + np.inf
        
        # Construct the first symbol path metric
        for a in np.arange(self.alphabetlen):
            if a != 0:
                continue
            paths[a,0] = self.alphabet[a]
            guess = paths[a]
            
            # KEEP IT SIMPLE FOR NOW, UPSAMPLE THE WHOLE PATH
            upguess = np.zeros(pathlen * self.up, dtype=paths.dtype)
            upguess[::self.up] = guess
            print(upguess[:self.up*2])
            
            # Loop over all sources
            x_all = np.zeros((self.L, self.pulselen), dtype=np.complex128)
            for i in np.arange(self.L): 
                xc = np.convolve(self.pulses[i], upguess[:1])[-self.pulselen:]

                xcs = np.exp(1j*(-self.omegas[i]*np.arange(0*self.up,0*self.up+self.pulselen))) * xc
                # xcs = np.exp(1j*(-self.omegas[i]*np.arange(len(xc)))) * xc
                x_all[i,:] = xcs[-self.pulselen:]
                
            summed = np.sum(x_all, axis=0)
            
            print("Writing to pathmetric[%d]" % (a))
            pathmetrics[a] = np.linalg.norm(y[0*self.up:1*self.up] - summed[:self.up])**2
            
        print(pathmetrics)
        print(paths)
        
        # Iterate over the rest of the symbols
        for n in np.arange(1, pathlen):
            # Calculate all branches
            branchmetrics, shortbranchmetrics = self.calcAllBranchMetrics(y, paths, pathmetrics, n)
            
            # Extract and update best paths
            self.calcPathMetrics(shortbranchmetrics, branchmetrics, paths, pathmetrics, n)
            
            if n == 244:
                break
            
            # print("--------------------------")
        
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
        
        # Select the current symbol
        for p in np.arange(paths.shape[0]):
            # Select a valid pre-transition path
            for t in np.arange(len(self.pretransitions[p])):
                # if self.pretransitions[p,t] != 0: # DEBUG
                #     continue
                
                # print("Alphabet %d->%d at index %d" % (self.pretransitions[p,t],p,n))
                if pathmetrics[self.pretransitions[p,t]] == np.inf:
                    # print("Pretransition is inf, skipping!")
                    branchmetrics[p,t] = np.inf
                    shortbranchmetrics[p,t] = np.inf
                    continue
                
                guess = np.copy(paths[self.pretransitions[p,t]])
                guess[n] = self.alphabet[p]
                # print("Guess:")
                # print(guess)
                # KEEP IT SIMPLE FOR NOW, UPSAMPLE THE WHOLE PATH
                upguess = np.zeros(pathlen * self.up, dtype=paths.dtype)
                upguess[::self.up] = guess
                # print(upguess[:n*self.up+1:self.up])
                # assert(np.all(upguess[::self.up] == guess))
                
                # Loop over all sources
                x_all = np.zeros((self.L, self.pulselen), dtype=np.complex128)
                for i in np.arange(self.L): 
                    s = np.max([n*self.up - self.pulselen,0])
                    xc = np.convolve(self.pulses[i], upguess[s:n*self.up+1])[-self.pulselen:]
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
        paths2 = np.copy(paths)
        pathmetrics2 = np.copy(pathmetrics)
        
        for p in np.arange(branchmetrics.shape[0]):
            if np.all(branchmetrics[p,:] == np.inf):
                pathmetrics2[p] = np.inf
                continue
            bestPrevIdx = np.argmin(branchmetrics[p,:])
            paths2[p,:] = paths[self.pretransitions[p,bestPrevIdx],:] # copy the whole path over
            paths2[p,n] = self.alphabet[p]
            pathmetrics2[p] = pathmetrics[self.pretransitions[p,bestPrevIdx]] + shortbranchmetrics[p,bestPrevIdx]
            
        paths[:,:] = paths2[:,:]
        pathmetrics[:] = pathmetrics2[:]
        
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