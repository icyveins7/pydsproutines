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
    
    [pro_first]
    src = src1
    sig = signal1
    ...
    
    [pro_second]
    src = src2
    sig = signal1
    ...
    
    [pro_firstdifferent]
    src = src1
    sig = signal1
    ...
    
    [myfirstworkspace]
    pro_first
    pro_firstdifferent
    
In general, it is useful to align the different signals and sources at the top, as these are
IMMUTABLE, and usually never change; if a signal is present in a particular source and needs processing,
you generally know its characteristics and they will probably be identical throughout the course of your 'work'.

However, in the course of your work, you may experiment with different ways of processing the same signal,
even if its coming from the same source (maybe you want to use less filter taps for speed comparisons?).
As such, naming your processing sections with 'proc_someDescriptiveName' will help you remember what it is
you are trying with a particular processing section.

Finally, when you want to load an entire workspace configuration, you can then simply identify the
processing sections which you would like to use.

"""

from configparser import ConfigParser, RawConfigParser, SectionProxy
import os
import __main__

#%% 
class DirectSingleConfig(ConfigParser):
    '''
    A wrapper for the most common use-case, a single config file, without having to call the read() again.
    The filename will default to the current script's name, with extension .ini, and this resides in the same folder.
    '''
    def __init__(self, filename: str=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Gonna take the liberty to set this since I always use it
        self.optionxform = str # lambda option: option # preserves upper-case
        # Note that you must set the optionxform before reading
        if filename is None:
            filename = os.path.splitext(__main__.__file__)[0] + ".ini"
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
class SourceSectionProxy(SectionProxy):
    def __repr__(self):
        return '<SourceSection: {}>'.format(self._name)
    
    @property
    def srcdir(self):
        return self.get('srcdir')
    
    @property
    def fs(self):
        return self.getfloat('fs')
    
    @property
    def fc(self):
        return self.getfloat('fc')
    
    @property
    def conjSamples(self):
        return self.getbool('conjSamples')
    
    @property
    def headerBytes(self):
        return self.getint('headerBytes')
    
    @property
    def dtype(self):
        return self.get('dtype')
    
#%%
class SignalSectionProxy(SectionProxy):
    def __repr__(self):
        return '<SignalSection: {}>'.format(self._name)
    
    @property
    def target_fc(self):
        return self.getfloat('target_fc')
    
    @property
    def baud(self):
        return self.getfloat('baud')

    @property
    def numPeriodBits(self):
        '''For periodic signals, this is the sum of the burst (on) and guard (off) duration in bits.'''
        return self.getint('numPeriodBits')

    @property
    def numBurstBits(self):
        '''For periodic signals, this is the single burst (on) duration in bits.'''
        return self.getint('numBurstBits')

    @property
    def numGuardBits(self):
        '''For periodic signals, this is the single guard (off) duration in bits.'''
        return self.getint('numGuardBits')

    @property
    def numBursts(self):
        '''For periodic signals, this is the total number of bursts.'''
        return self.getint('numBursts')

    @property
    def hasChannels(self):
        '''For signals that occupy multiple channels, this returns True.'''
        return self.getbool('hasChannels')

    @property
    def numChannels(self):
        '''For signals that occupy multiple channels, this returns the number of channels.'''
        return self.getint('numChannels')

    @property
    def channelSpacingHz(self):
        '''For signals that occupy multiple channels, this returns the spacing between each channel.'''
        return self.getfloat('channelSpacingHz')
 
    
#%%
class ProcessingSectionProxy(SectionProxy):
    def __repr__(self):
        return '<ProcessingSection: {}>'.format(self._name)
        
    # Redirect to the associated src and sig section objects
    @property
    def src(self):
        return self.parser.getSrc(self.get('src'))
        
    @property
    def sig(self):
        return self.parser.getSig(self.get('sig'))
        
    @property
    def numTaps(self):
        return self.getint('numTaps')
    
    @property
    def target_osr(self):
        return self.getint('target_osr')
    
#%%
class WorkspaceSectionProxy(SectionProxy):
    def __repr__(self):
        return '<WorkspaceSection: {}>'.format(self._name)
    

#%%
class DSPConfig(DirectSingleConfig):
    def __init__(self, filename: str, *args, allow_no_value=True, **kwargs):
        super().__init__(filename, *args, allow_no_value=allow_no_value, **kwargs)
        self.recastSections()
    
    # Note that for these type-specific sections, we strip the prefix
    # Also note that these list everything in the config file.
    @property
    def allSources(self):
        return {self._keySuffix(key): item for key, item in self._proxies.items() if self._isSourceSection(key)}
    
    @property
    def allSignals(self):
        return {self._keySuffix(key): item for key, item in self._proxies.items() if self._isSignalSection(key)}
    
    @property
    def allProcesses(self):
        return {self._keySuffix(key): item for key, item in self._proxies.items() if self._isProcessingSection(key)}
        
    # But usually you want to activate a certain workspace
    def loadMenu(self):
        '''Overloaded the menu to only show workspaces.'''
        workspaces = [section for section in self.sections() if self._isWorkspaceSection(section)]
        
        for i, workspace in enumerate(workspaces):
            print("%d: %s" % (i, workspace))
        idx = int(input("Select workspace: "))
        self.loadSection(workspaces[idx])
        
    # And with a certain workspace, you only want the associated processes
    @property
    def processes(self):
        return {self._keySuffix(key): self._proxies[key] for key in list(self.currentSection.keys())}
        
    # Auxiliary things..
    def _makeSrcKey(self, src: str):
        return 'src_' + src
        
    def getSrc(self, src: str):
        return self._proxies[self._makeSrcKey(src)]
        
    def _makeSigKey(self, sig: str):
        return 'sig_' + sig
        
    def getSig(self, sig: str):
        return self._proxies[self._makeSigKey(sig)]
    
    def recastSections(self):
        # Proxies stored in _proxies
        for key in self._proxies:
            if self._isSourceSection(key):
                self._proxies[key] = SourceSectionProxy(self._proxies[key]._parser, self._proxies[key]._name)
                
            elif self._isSignalSection(key):
                self._proxies[key] = SignalSectionProxy(self._proxies[key]._parser, self._proxies[key]._name)
                
            elif self._isProcessingSection(key):
                self._proxies[key] = ProcessingSectionProxy(self._proxies[key]._parser, self._proxies[key]._name)
                
            else:
                self._proxies[key] = WorkspaceSectionProxy(self._proxies[key]._parser, self._proxies[key]._name)
    
    def _keyPrefix(self, key: str):
        return key[:4]

    def _keySuffix(self, key: str):
        return key[4:]          
      
    def _isSourceSection(self, key: str):
        return self._keyPrefix(key) == 'src_'
            
    def _isSignalSection(self, key: str):
        return self._keyPrefix(key) == 'sig_'
    
    def _isProcessingSection(self, key: str):
        return self._keyPrefix(key) == 'pro_'
    
    def _isWorkspaceSection(self, key: str):
        if not (self._isSourceSection(key) or self._isSignalSection(key) or self._isProcessingSection(key)):
            return True
        else:
            return False
    
#%% Many cases involve processing only one signal at a time i.e. only 1 process for each workspace
class SingleProcessDSPConfig(DSPConfig):
    @property
    def process(self):
        return list(self.processes.items())[0][1]
        
    # We also short circuit the src and signal directly
    @property
    def src(self):
        return self.process.src
        
    @property
    def sig(self):
        return self.process.sig

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
    
    cfg = DSPConfig("")
    cfg['src_src1'] = {
        'fs': 1000    
    }
    cfg['src_src2'] = {
        'fs': 2000    
    }
    cfg['sig_sig1'] = {
        'baud': 100
    }
    cfg['pro_src1_sig1'] = {
        'src': 'src1',
        'sig': 'sig1',
        'numTaps': 16
    }
    # For testing, we manually recast here (don't need to do this if loaded from file)
    cfg.recastSections()
    # See that the repr is outputted correctly
    print(cfg['src_src1'])
    print(cfg['sig_sig1'])
    # See that you can get multiple sources/signals
    sources = cfg.getSources()
    # And can use the property getters comfortably as well
    print(sources['src1'].fs)
    print(sources['src2'].fs)
    
    
    #%%
    # #%% Testing simple creation with files
    # conf = DirectSingleConfig("test.ini")
    # assert(not os.path.exists("test.ini"))
    
    # conf = DirectSingleConfig.new('test.ini')
    # assert(os.path.exists('test.ini'))
    # os.remove('test.ini')
    
    # #%% Testing SingleSignalConfig
    # conf = Src1Target1Config("test.ini") # SingleSignalConfig("test.ini")
    # # Set in memory for now
    # fs = 100.0
    # fc = 1000.0
    # freqshift = 123.0
    # baud = 10.0
    # conf['s'] = {
    #     'fs': fs,
    #     'fc': fc,
    #     'freqshift': freqshift,
    #     'baud': baud
    # }
    # # Check values
    # conf.loadSection('s')
    # assert(fs == conf.fs)
    # assert(fc == conf.fc)
    # assert(freqshift == conf.freqshift)
    # assert(baud == conf.baud)
    
    # #%% Testing periodic config
    # conf = Src1PeriodicTarget1Config("test.ini")
    
    # bitsPerBurst = 50
    # periodBits = 100
    
    # conf['s'] = {
    #     'fs': fs,
    #     'fc': fc,
    #     'freqshift': freqshift,
    #     'baud': baud,
    #     'bitsPerBurst': bitsPerBurst,
    #     'periodBits': periodBits
    # }
    # conf.loadSection('s')
    # assert(fs == conf.fs)
    # assert(fc == conf.fc)
    # assert(freqshift == conf.freqshift)
    # assert(baud == conf.baud)
    # assert(bitsPerBurst == conf.bitsPerBurst)
    # assert(periodBits == conf.periodBits)