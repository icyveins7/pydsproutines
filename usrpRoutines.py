# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 17:50:27 2021

@author: Lken
"""

import numpy as np
import concurrent.futures
import os
import fnmatch

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
        self.filenames = fnmatch.filter(os.listdir(self.folderpath), "*"+extension)
        self.filepaths = [os.path.join(self.folderpath, i) for i in self.filenames]
        
        self.numSampsPerFile = numSampsPerFile
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        
        self.fidx = 0
        
    def reset(self):
        self.fidx = 0
        
    def get(self, numFiles, start=None):
        if start is None:
            start = self.fidx
        end = start + numFiles
        self.fidx = end
        
        fps = self.filepaths[start:end]
        alldata = multiBinRead(fps, self.numSampsPerFile, self.in_dtype, self.out_dtype)
        return alldata, fps
    
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
        alldata, fps = super().get(numFiles, start)
        fts = self.filetimes[self.fidx-numFiles:self.fidx]
        return alldata, fps, fts
        