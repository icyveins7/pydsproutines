# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 17:50:27 2021

@author: Lken
"""

import numpy as np
import concurrent.futures
import os
import fnmatch
import matplotlib.pyplot as plt
from signalCreationRoutines import *
import scipy.signal as sps
import shutil

#%% Readers for complex data.

def simpleBinRead(filename, numSamps=-1, in_dtype=np.int16, out_dtype=np.complex64):
    '''
    Simple, single-file complex data reader. numSamps refers to the number of complex samples.
    '''
    data = np.fromfile(filename, dtype=in_dtype, count=numSamps*2).astype(np.float32).view(out_dtype)
    
    return data

def multiBinRead(filenames, numSamps, in_dtype=np.int16, out_dtype=np.complex64):
    '''
    Simple, multi-file complex data reader. Calls simpleBinRead().
    numSamps refers to the number of complex samples.
    '''
    alldata = np.zeros(len(filenames)*numSamps, out_dtype)
    for i in range(len(filenames)):
        filename = filenames[i]
        alldata[i*numSamps : (i+1)*numSamps] = simpleBinRead(filename, numSamps, in_dtype, out_dtype)
    
    return alldata

def multiBinReadThreaded(filenames, numSamps, in_dtype=np.int16, out_dtype=np.complex64, threads=2):
    '''
    Threaded multi-file reader. Anything more than 2 threads is usually insignificant/actually worse.
    '''
    alldata = np.zeros(len(filenames)*numSamps, out_dtype)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        future_load = {executor.submit(simpleBinRead, filenames[i], numSamps, in_dtype, out_dtype): i for i in np.arange(len(filenames))}
        for future in concurrent.futures.as_completed(future_load):
            i = future_load[future] # reference dictionary for index
            alldata[i*numSamps: (i+1)*numSamps] = future.result() # write to the mass array
            
    return alldata

#%% Convenience classes
class FolderReader:
    def __init__(self, folderpath, numSampsPerFile, extension=".bin", in_dtype=np.int16, out_dtype=np.complex64):
        self.folderpath = folderpath
        self.extension = extension
        self.refreshFilelists()
        
        self.numSampsPerFile = numSampsPerFile
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        
        self.reset()
        
    def refreshFilelists(self):
        self.filenames = fnmatch.filter(os.listdir(self.folderpath), "*"+self.extension)
        self.filepaths = [os.path.join(self.folderpath, i) for i in self.filenames]
        self.reset() # cannot ensure file indexing after list is reset
        
    def reset(self):
        self.fidx = 0
        
    def get(self, numFiles, start=None):
        if start is None:
            start = self.fidx
        end = start + numFiles
        if end > len(self.filepaths):
            raise ValueError("Insufficient files remaining.")
        self.fidx = end
        
        fps = self.filepaths[start:end]
        alldata = multiBinReadThreaded(fps, self.numSampsPerFile, self.in_dtype, self.out_dtype)
        return alldata, fps
    
    def fastCheck(self, numFiles=None, start=0, plotSpecgram=True, plots=False, fs=None, viewskip=1):
        if numFiles is None:
            numFiles = len(self.filepaths) - start
        
        alldata, *_ = self.get(numFiles, start)
        
        # Run some diagnostics
        maxreal = np.max(np.real(alldata))
        minreal = np.min(np.real(alldata))
        maximag = np.max(np.imag(alldata))
        minimag = np.min(np.imag(alldata))
        print("Max/min real: %d,%d\nMax/min imag: %d,%d" % (int(maxreal),int(minreal),int(maximag),int(minimag)))
        
        # Plot some simple things if requested
        if fs is None:
            print("Setting fs to sampsPerFile = %d" % (self.numSampsPerFile))
            fs = self.numSampsPerFile
        
        if plots:    
            fig, ax = plt.subplots(2,1)
            ax[0].plot(np.arange(alldata.size)[::viewskip]/fs, np.abs(alldata)[::viewskip])
            ax[1].plot(makeFreq(alldata.size, fs)[::viewskip], 20*np.log10(np.abs(np.fft.fft(alldata)))[::viewskip])
        
        if plotSpecgram:
            plt.figure()
            plt.specgram(alldata, NFFT=1024, Fs=fs)
            
    
class SortedFolderReader(FolderReader):
    def __init__(self, folderpath, numSampsPerFile, extension=".bin", in_dtype=np.int16, out_dtype=np.complex64, ensure_incremental=True):
        super().__init__(folderpath, numSampsPerFile, extension, in_dtype, out_dtype)
        # Ensure the filenames are properly sorted
        self.filetimes = np.array([int(os.path.splitext(i)[0]) for i in self.filenames])
        sortidx = np.argsort(self.filetimes).flatten()
        # Use the index list to sort all 3 arrays
        self.filetimes = self.filetimes[sortidx]
        self.filenames = [self.filenames[i] for i in sortidx]
        self.filepaths = [self.filepaths[i] for i in sortidx]
        
        # Error check that the values are incremental (no gaps)
        if ensure_incremental:
            assert(np.all(np.diff(self.filetimes)==1))
            
    def get(self, numFiles, start=None):
        '''
        Parameters
        ----------
        numFiles : int
            Number of files to open & concatenate.
        start : int, optional
            File index (starting from 0 for the first file). The default is None (=0).

        Returns
        -------
        alldata : array
            Data.
        fps : list of strings
            Filepaths.
        fts : list of ints.
            File times.
        '''
        alldata, fps = super().get(numFiles, start)
        fts = self.filetimes[self.fidx-numFiles:self.fidx]
        return alldata, fps, fts
    
    def splitHighAmpSubfolders(self, targetfolderpath:str, selectTimes:list = None,
                               minAmp:float = 1e3, bufFront:int = 1, bufBack:int = 1,
                               fmt:str = "%06d"):
        '''
        Detects files with high amplitudes, and selects a group of files around them based on bufFront/bufBack.
        Groups are then individually written into separate subfolders based on 'fmt', residing in 'targetfolderpath'.

        Parameters
        ----------
        targetfolderpath : str
            Top target directory.
        selectTimes : list, optional
            List of file times to use. The default is None, which will then open the folder to get the groups by comparing to minAmp.
        minAmp : float, optional
            Minimum amplitude for the file to be selected. The default is 1e3.
        bufFront : int, optional
            Number of files to save in front of the high amplitude file. The default is 1.
        bufBack : int, optional
            Number of files to save behind the high amplitude file. The default is 1.
        fmt : str, optional
            Subfolder name format. The default is "%06d".

        Returns
        -------
        selectTimes : list
            The selected times of the files. This is returned so that it may be passed to another SortedFolderReader
            to snapshot the same groups synchronously.
        '''
        
        # First check which files pass the minAmp, we don't want to touch the internal index so don't use get()
        if selectTimes is None:
            selectTimes = []
            for i in range(len(self.filepaths)):
                data = simpleBinRead(self.filepaths[i])
                maxamp = np.max(np.abs(data))
                if maxamp > minAmp:
                    # print("%d amp: %g" % (self.filetimes[i], maxamp))
                    selectTimes.extend(range(self.filetimes[i]-bufFront, self.filetimes[i]+bufBack + 1))
        else:
            print("Using specified selectTimes..")
                
        # Extract only unique times and sort
        selectTimes = list(set(selectTimes))
        selectTimes.sort()
        
        # Now pull groups out where the difference is more than 1
        groupSplitIdx = np.hstack((0,(np.argwhere(np.diff(selectTimes) > 1) + 1).flatten(), len(selectTimes)))
        # print(groupSplitIdx)
        
        # Create the main dir
        if not os.path.isdir(targetfolderpath):
            os.makedirs(targetfolderpath)
            print("Created %s" % targetfolderpath)
        
        # Loop over groups
        for i in range(len(groupSplitIdx)-1):
            grptimes = selectTimes[groupSplitIdx[i]:groupSplitIdx[i+1]]
            grpstring = fmt % i
            subdirpath = os.path.join(targetfolderpath, grpstring)
            if not os.path.isdir(subdirpath):
                os.makedirs(subdirpath)
            srcfilepaths = [os.path.join(self.folderpath, "%d.bin"%(i)) for i in grptimes]
            dstfilepaths = [os.path.join(subdirpath, "%d.bin"%(i)) for i in grptimes]
            for p in range(len(srcfilepaths)):
                print("Group %d: copy %s to %s" % (i, srcfilepaths[p], dstfilepaths[p]))
                try:
                    shutil.copyfile(srcfilepaths[p],dstfilepaths[p])
                except:
                    print("Error occurred while copying %s to %s" % (srcfilepaths[p],dstfilepaths[p]))
            print("-----")
            
    
        
        return selectTimes
        
    
#%% Simple class to contain multiple synced readers
class SyncReaders:
    def __init__(self, folderpaths, numSampsPerFile, extension=".bin", in_dtype=np.int16, out_dtype=np.complex64, ensure_incremental=True):
        self.folderpaths = folderpaths
        self.readers = [SortedFolderReader(folderpath, numSampsPerFile, extension, in_dtype, out_dtype, ensure_incremental)
                        for folderpath in folderpaths]
        
        for i in range(1,len(self.readers)):
            # Ensure all files tally
            assert(self.readers[i].filetimes[0] == self.readers[0].filetimes[0])
            assert(self.readers[i].filetimes[-1] == self.readers[0].filetimes[-1])
            
    @classmethod
    def fromSubdirs(cls, topfolderpath, numSampsPerFile, extension=".bin", in_dtype=np.int16, out_dtype=np.complex64, ensure_incremental=True):
        subdirs = [os.path.join(topfolderpath, i) for i in os.listdir(topfolderpath) if os.path.isdir(os.path.join(topfolderpath, i))]
        return cls(subdirs, numSampsPerFile, extension, in_dtype, out_dtype, ensure_incremental)
            
    def get(self, numFiles, start=None):
        outdata = {}
        outfps = {}
        outfts = {}
        for i in range(len(self.readers)):
            alldata, fps, fts = self.readers[i].get(numFiles, start)
            outdata[i] = alldata
            outfps[i] = fps
            outfts[i] = fts
            
        return outdata, outfps, outfts
    
#%% Simple class to read the groups extracted from the readers
class GroupReaders:
    def __init__(self, folderpaths, numSampsPerFile, extension=".bin", in_dtype=np.int16, out_dtype=np.complex64, ensure_incremental=True):
        self.folderpaths = folderpaths
        self.cGrp = -1 # current group, start from -1 so the first call returns idx 0
        
        # Check that the groups match
        self.groups0 = [i for i in os.listdir(folderpaths[0]) if os.path.isdir(os.path.join(folderpaths[0], i))]
        self.groups0.sort() # sort for later use in other methods
        self.set0 = set(self.groups0)
        for i in range(1,len(folderpaths)):
            groupsi = [k for k in os.listdir(folderpaths[i]) if os.path.isdir(os.path.join(folderpaths[i], k))]
            seti = set(groupsi)
            assert(seti==self.set0) # ensure the groups tally
            
        # Storage for getter methods later
        self.numSampsPerFile = numSampsPerFile
        self.extension = extension
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.ensure_incremental = ensure_incremental
        self.groupdirpaths = None
        
    def nextGroup(self) -> SyncReaders:
        '''
        Increments the group number, and returns a SyncReaders object for the folders, for the new group.
        
        Returns
        -------
        SyncReaders object.
            Use as if manually init'ed on the folder + group directory structure.

        '''
        self.cGrp = self.cGrp + 1
        
        self.groupdirpaths = [os.path.join(p, self.groups0[self.cGrp]) for p in self.folderpaths]
        
        readers = SyncReaders(self.groupdirpaths, self.numSampsPerFile, self.extension, self.in_dtype,
                              self.out_dtype, self.ensure_incremental)
        
        return readers
        
    def resetGroup(self):
        self.cGrp = -1
    
    