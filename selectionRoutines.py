# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:37:43 2022

@author: Lken
"""

import numpy as np
import scipy as sp
import scipy.signal as sps

#%%
class BurstSelector:
    def __init__(self, burstLen: int, medfiltlen: int=None):
        self.burstLen = burstLen
        self.medfiltlen = medfiltlen
        
        # Holding arrays/output
        self.energy = None
        self.peaks = None
        self.peakprops = None
        
    def detect(self, x_abs: np.ndarray, sortPeaks: bool=True, limit: int=None):
        if self.medfiltlen is not None:
            self.f = sps.medfilt(x_abs, self.medfiltlen)
        else:
            self.f = x_abs
            
        # Convolve
        self.energy = sps.convolve(np.ones(self.burstLen), self.f, mode='valid')
        
        # Detect the bursts
        self.peaks, self.peakprops = sps.find_peaks(self.energy, distance=self.burstLen)
        
        if sortPeaks:
            self.peaks = np.sort(self.peaks)
        
        return self.peaks
    
    def plot(self, ax):
        if self.medfiltlen is not None:
            ax.plot(self.f, label='Medfilt')
        
        ax.plot(self.energy/self.burstLen, label='Window Convolved')
        ax.plot(self.peaks, self.energy[self.peaks]/self.burstLen, 'rx', label="Peaks Detected")
        ax.legend()