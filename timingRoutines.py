#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:52:27 2022

@author: seoxubuntu
"""

import time

class Timer:
    def __init__(self):
        self.t = []
        self.labels = []
        
    def reset(self):
        self.t.clear()
        self.labels.clear()
        
    def start(self):
        '''
        Convenience function to start the timer (which by definition should have no label).
        Automatically calls reset() before beginning.
        '''
        self.reset()
        self.evt()       
        
    def evt(self, label: str=""):
        '''
        Adds an event i.e. records the current time using perf_counter().
        Also adds a label if desired, describing the time elapsed since the last event.

        Parameters
        ----------
        label : str, optional
            The default is "".

        '''
        self.t.append(time.perf_counter())
        self.labels.append(label)
        
    def rpt(self, showTotal: bool=True):
        '''
        Reports times elapsed between events.
        '''
        for i in range(1, len(self.t)):
            print("%d->%d : %fs. %s" % (
                i-1, i, self.t[i]-self.t[i-1], self.labels[i]))
            
        if showTotal:
            print("Total: %fs." % (self.t[-1] - self.t[0]))
            
    def end(self, label: str=""):
        '''
        Convenience function to add an event and report immediately.
        '''
        self.evt(label)
        self.rpt()
        
        
if __name__ == "__main__":
    import numpy as np
    
    length = int(10e6)
    timer = Timer()
    timer.start()
    data = np.random.rand(length) + 1j * np.random.rand(length)
    timer.evt("Rand")
    f = np.fft.fft(data)
    timer.end("FFT")
    
    timer.reset()
    timer.start()
    timer.end("Timer overhead")