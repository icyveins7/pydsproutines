#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 22:50:36 2023

@author: seoxubuntu

The goal of these classes is to adhere to a strict, concise config file arrangement.
As this is a DSP-related config, we define the configs as having sections related to distinct tasks:
    
    1) Sources: Loading/reading samples
    Each source is usually associated with a particular path or directory,
    along with typical things like sample rate, centre frequency etc.
    
    2) Signals
    Each signal to be processed usually requires knowledge of properties such as
    baud rate, centre frequency etc.
    
    3) Processing
    The same signal may be processed in different ways if using different sources;
    maybe different thresholds, or different number of filter taps if sample rates are not identical etc.
    As such, the processing draws a link between a named signal and source.
    
We then combine these into distinct 'workspace' configs which aggregate 1 or more source configs,
signal configs and processing configs. A typical config file may look something like this:
    
    [sig_signal1]
    ...
    
    [src_src1]
    ...
    
    [src_src2]
    ...
    
    [proc_src1_signal1]
    ...
    
    [proc_src2_signal1]
    ...

"""

from configparser import ConfigParser, RawConfigParser

#%% 
class DirectSingleConfig(ConfigParser):
    '''A wrapper for the most common use-case, a single config file, without having to call the read() again.'''
    def __init__(self, filename: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Gonna take the liberty to set this since I always use it
        self.optionxform = str # lambda option: option # preserves upper-case
        # Note that you must set the optionxform before reading
        self.read(filename)
        self.currentSection = None
        
    @classmethod
    def new(cls, filename: str, *args, **kwargs):
        open(filename, 'w')
        return cls(filename, *args, **kwargs)
    
    def loadSection(self, section: str):
        '''We use this to overload our own methods, without having to rewrite a SectionProxy class.'''
        self.currentSection = self.__getitem__(section)
        
    def loadMenu(self):
        sections = self.sections()
        sections.insert(0, 'DEFAULT')
        for i, section in enumerate(sections):
            print("%d: %s" % (i, section))
        idx = int(input("Select section: "))
        self.loadSection(sections[idx])

#%%
class SourceConfigMixin:
    '''This mixin contains the parameters needed to read the samples.'''
    def loadSrc(self, src: str):
        '''This looks for a section prefixed with "src_" and loads it.'''
        self.loadSection('src_' + src)
    
    @property
    def srcdir(self):
        return self.currentSection.get('srcdir')
    
    @property
    def fs(self):
        return self.currentSection.getfloat('fs')
    
    @property
    def fc(self):
        return self.currentSection.getfloat('fc')
    
    @property
    def conjSamples(self):
        return self.currentSection.getbool('conjSamples')
    
    @property
    def headerBytes(self):
        return self.currentSection.getint('headerBytes')
    
    @property
    def dtype(self):
        return self.currentSection.get('dtype')
    
#%%
class SignalConfigMixin:
    '''This mixin contains parameters related to a single target signal in the samples, and how to process it (for demodulation etc.).'''
    def loadSig(self, sig: str):
        '''This looks for a section prefixed with "sig_" and loads it.'''
        self.loadSection('sig_' + sig)
    
    @property
    def target_fc(self):
        return self.currentSection.getfloat('target_fc')
    
    @property
    def freqshift(self):
        return self.currentSection.getfloat('freqshift')
    
    @property
    def baud(self):
        return self.currentSection.getfloat('baud')
    
    @property
    def numTaps(self):
        return self.currentSection.getint('numTaps')
    
    @property
    def target_osr(self):
        return self.currentSection.getint('target_osr')

#%% Intend to deprecate from this onwards, bad format to do things..
class SingleSourceConfigMixin:
    '''This mixin contains the parameters needed to read the samples.'''
    @property
    def srcdir(self):
        return self.currentSection.get('srcdir')
    
    @property
    def fs(self):
        return self.currentSection.getfloat('fs')
    
    @property
    def fc(self):
        return self.currentSection.getfloat('fc')
    
    @property
    def conjSamples(self):
        return self.currentSection.getbool('conjSamples')
    
    @property
    def headerBytes(self):
        return self.currentSection.getint('headerBytes')
    
#%%
class DualSourceConfigMixin(SingleSourceConfigMixin):
    '''
    This mixin contains all the single (primary) source parameters,
    plus an extra set prefixed with 'sec_' to entail a secondary source.
    '''
    @property
    def sec_srcdir(self):
        return self.currentSection.get('sec_srcdir')
    
    @property
    def sec_fs(self):
        return self.currentSection.getfloat('sec_fs')
    
    @property
    def sec_fc(self):
        return self.currentSection.getfloat('sec_fc')
    
    @property
    def sec_conjSamples(self):
        return self.currentSection.getbool('sec_conjSamples')
    
    @property
    def sec_headerBytes(self):
        return self.currentSection.getint('sec_headerBytes')
    
#%% 
class SingleTargetConfigMixin:
    '''This mixin contains parameters related to a single target signal in the samples, and how to process it (for demodulation etc.).'''
    @property
    def target_fc(self):
        return self.currentSection.getfloat('target_fc')
    
    @property
    def freqshift(self):
        return self.currentSection.getfloat('freqshift')
    
    @property
    def baud(self):
        return self.currentSection.getfloat('baud')
    
    @property
    def numTaps(self):
        return self.currentSection.getint('numTaps')
    
    @property
    def target_osr(self):
        return self.currentSection.getint('target_osr')
    
#%%
class SinglePeriodicTargetConfigMixin(SingleTargetConfigMixin):
    @property
    def bitsPerBurst(self):
        return self.currentSection.getint('bitsPerBurst')
    
    @property
    def periodBits(self):
        return self.currentSection.getint('periodBits')    
   
#%% Some common combinations
class Src1Target1Config(SingleSourceConfigMixin, SingleTargetConfigMixin, DirectSingleConfig):
    pass

class Src1PeriodicTarget1Config(SingleSourceConfigMixin, SinglePeriodicTargetConfigMixin, DirectSingleConfig):
    pass

class Src2Target1Config(DualSourceConfigMixin, SingleTargetConfigMixin, DirectSingleConfig):
    pass

class Src2PeriodicTarget1Config(DualSourceConfigMixin, SinglePeriodicTargetConfigMixin, DirectSingleConfig):
    pass
        
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
    conf = Src1Target1Config("test.ini") # SingleSignalConfig("test.ini")
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
    
    #%% Testing periodic config
    conf = Src1PeriodicTarget1Config("test.ini")
    
    bitsPerBurst = 50
    periodBits = 100
    
    conf['s'] = {
        'fs': fs,
        'fc': fc,
        'freqshift': freqshift,
        'baud': baud,
        'bitsPerBurst': bitsPerBurst,
        'periodBits': periodBits
    }
    conf.loadSection('s')
    assert(fs == conf.fs)
    assert(fc == conf.fc)
    assert(freqshift == conf.freqshift)
    assert(baud == conf.baud)
    assert(bitsPerBurst == conf.bitsPerBurst)
    assert(periodBits == conf.periodBits)