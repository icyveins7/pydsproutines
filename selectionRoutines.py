# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:37:43 2022

@author: Lken
"""

import numpy as np
import scipy as sp
import scipy.signal as sps
import cupy as cp
import cupyx.scipy.signal as cpss

#%%
class BurstSelector:
    def __init__(self, burstLen: int, medfiltlen: int=None, minHeightFactor: float=None):
        self.burstLen = burstLen
        self.medfiltlen = medfiltlen
        self.minHeightFactor = minHeightFactor
        
        # Holding arrays/output
        self.energy = None
        self.peaks = None
        self.peakprops = None
        self.minHeight = None
        
        # Configurations
        self.useGpuMedfilt = False
        
    def toggleGpuMedfilt(self, enabled=True):
        self.useGpuMedfilt = enabled
        
    def detect(self, x_abs: np.ndarray, sortPeaks: bool=True, limit: int=None):
        if self.medfiltlen is not None:
            if self.useGpuMedfilt:
                self.f = cpss.medfilt(cp.array(x_abs), self.medfiltlen).get()
            else:
                self.f = sps.medfilt(x_abs, self.medfiltlen)
        else:
            self.f = x_abs
            
        # Convolve
        self.energy = sps.convolve(np.ones(self.burstLen), self.f, mode='valid')
        
        # Detect the bursts
        if self.minHeightFactor is not None:
            self.minHeight = np.max(self.energy) * self.minHeightFactor
        else:
            self.minHeight = None
            
        self.peaks, self.peakprops = sps.find_peaks(self.energy, height=self.minHeight, distance=self.burstLen)
        
        if sortPeaks:
            self.peaks = np.sort(self.peaks)
        
        return self.peaks
    
    def plot(self, ax):
        if self.medfiltlen is not None:
            ax.plot(self.f, label='Medfilt')
        
        ax.plot(self.energy/self.burstLen, label='Window Convolved')
        ax.plot(self.peaks, self.energy[self.peaks]/self.burstLen, 'rx', label="Peaks Detected")
        if self.minHeight is not None:
            ax.hlines(self.minHeight/self.burstLen, ax.axis()[0], ax.axis()[1], linestyles='dashed', label="Minimum Height", zorder=10)
        ax.legend()