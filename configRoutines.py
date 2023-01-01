#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 22:50:36 2023

@author: seoxubuntu
"""

from configparser import ConfigParser, RawConfigParser

#%% A wrapper for the most common use-case, a single config file, without having to call the read() again
class DirectSingleConfig(ConfigParser):
    def __init__(self, filename: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read(filename)
        self.currentSection = None
        
    @classmethod
    def new(cls, filename: str, *args, **kwargs):
        open(filename, 'w')
        return cls(filename, *args, **kwargs)
    
    def loadSection(self, section: str):
        '''We use this to overload our own methods, without having to rewrite a SectionProxy class.'''
        self.currentSection = self.__getitem__(section)
        
#%%
class SingleSignalConfig(DirectSingleConfig):
    @property
    def fs(self):
        return self.currentSection.getfloat('fs')
    
    @property
    def fc(self):
        return self.currentSection.getfloat('fc')
    
    @property
    def freqshift(self):
        return self.currentSection.getfloat('freqshift')
    
    @property
    def baud(self):
        return self.currentSection.getfloat('baud')
        
#%%
if __name__ == "__main__":
    import os
    
    #%% Testing simple creation with files
    conf = DirectSingleConfig("test.ini")
    assert(not os.path.exists("test.ini"))
    
    conf = DirectSingleConfig.new('test.ini')
    assert(os.path.exists('test.ini'))
    os.remove('test.ini')
    
    #%% Testing SingleSignalConfig
    conf = SingleSignalConfig("test.ini")
    # Set in memory for now
    fs = 100.0
    fc = 1000.0
    freqshift = 123.0
    baud = 10.0
    conf['s'] = {
        'fs': fs,
        'fc': fc,
        'freqshift': freqshift,
        'baud': baud
    }
    # Check values
    conf.loadSection('s')
    assert(fs == conf.fs)
    assert(fc == conf.fc)
    assert(freqshift == conf.freqshift)
    assert(baud == conf.baud)