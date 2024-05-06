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
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
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
        return self.getboolean('conjSamples')
    
    @property
    def headerBytes(self):
        return self.getint('headerBytes')
    
    @property
    def dtype(self):
        return self.get('dtype')

    @property
    def lonlatalt(self):
        """
        Assumed to be in units deg,deg,m.
        """
        llastr = self.get('lonlatalt')
        if llastr is not None:
            lon,lat,alt = [float(i) for i in llastr.split(",")]
            return lon, lat, alt
        else:
            return None
    
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
        '''
        For signals that occupy multiple channels, this returns True.
        This is defined by checking for the presence of a 'numChannels' key.
        '''
        return self.getint('numChannels') is not None

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
    
    @property
    def threshold(self):
        return self.getfloat('threshold')
    
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
    
    @property
    def allWorkspaces(self):
        # No prefix for workspace
        return {key: item for key, item in self._proxies.items() if self._isWorkspaceSection(key)}
        
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
        return {
            key: self._proxies[self._makeProKey(key)] for key in list(self.currentSection.keys()) if key in self.allProcesses
        }
        
    # Modifiers
    def addSignal(self, signalName: str):
        self.add_section(self._makeSigKey(signalName))

    def removeSignal(self, signalName: str):
        self.remove_section(self._makeSigKey(signalName))

    def addSource(self, sourceName: str):
        self.add_section(self._makeSrcKey(sourceName))

    def removeSource(self, sourceName: str):
        self.remove_section(self._makeSrcKey(sourceName))

    def addProcess(self, processName: str):
        self.add_section(self._makeProKey(processName))

    def removeProcess(self, processName: str):
        self.remove_section(self._makeProKey(processName))

    def addWorkspace(self, workspaceName: str):
        self.add_section(workspaceName) # Just plain with no prefix

    def removeWorkspace(self, workspaceName: str):
        self.remove_section(workspaceName) # Just plain with no prefix

    # Auxiliary things..
    def _makeSrcKey(self, src: str):
        return 'src_' + src
        
    def getSrc(self, src: str):
        return self._proxies[self._makeSrcKey(src)]
        
    def _makeSigKey(self, sig: str):
        return 'sig_' + sig
        
    def getSig(self, sig: str):
        return self._proxies[self._makeSigKey(sig)]
    
    def _makeProKey(self, pro: str):
        return 'pro_' + pro
    
    def getProcess(self, pro: str):
        return self._proxies[self._makeProKey(pro)]
    
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


    @staticmethod 
    def _keyPrefix(key: str):
        return key[:4]

    @staticmethod
    def _keySuffix(key: str):
        return key[4:]          

    @staticmethod
    def _isSourceSection(key: str):
        return DSPConfig._keyPrefix(key) == 'src_'

    @staticmethod
    def _isSignalSection(key: str):
        return DSPConfig._keyPrefix(key) == 'sig_'

    @staticmethod 
    def _isProcessingSection(key: str):
        return DSPConfig._keyPrefix(key) == 'pro_'

    @staticmethod 
    def _isWorkspaceSection(key: str):
        if not (DSPConfig._isSourceSection(key) or DSPConfig._isSignalSection(key) or DSPConfig._isProcessingSection(key) or key=="DEFAULT"):
            return True
        else:
            return False
    
#%% Many cases involve processing only one signal at a time i.e. only 1 process for each workspace
class SingleProcessDSPConfig(DSPConfig):
    """
    This will assume only 1 process is present per workspace.
    If there are more, it will always only read the first one.
    Since there is only 1 process, the 'src' and 'sig' readers have been 
    short-circuited here to grant direct access i.e.
        cfg.src == cfg.process.src
        cfg.sig == cfg.process.sig
    """
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

#%% A good bunch of other cases involve processing signals pair-wise e.g. TDOA/FDOA etc.
class DoubleProcessDSPConfig(DSPConfig):
    """
    This will assume only 2 processes are present per workspace.
    If there are more, it will always only read the first two.
    The processes are accessed via the '.primary' and '.secondary' getters.
    """
    @property
    def primary(self):
        """
        Returns the primary process.
        """
        return list(self.processes.items())[0][1]

    @property
    def secondary(self):
        """
        Returns the secondary process.
        """
        return list(self.processes.items())[1][1]

    # Provide a lot of sensible aliases?
    @property
    def prifs(self):
        """
        Alias for self.primary.src.fs
        """
        return self.primary.src.fs

    @property
    def secfs(self):
        """
        Alias for self.secondary.src.fs
        """
        return self.secondary.src.fs

    @property
    def pridir(self):
        """
        Alias for self.primary.src.srcdir
        """
        return self.primary.src.srcdir

    @property
    def secdir(self):
        """
        Alias for self.secondary.src.srcdir
        """
        return self.secondary.src.srcdir
