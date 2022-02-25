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
import psutil
import time

import sqlite3 as sq
import io
import pandas as pd

#%% Sqlite adapters for numpy
def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sq.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# Converts np.array to TEXT when inserting
sq.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sq.register_converter("ARRAY", convert_array)


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
        
        # Calculate memory footprint
        oneElement = np.array([0],dtype=self.out_dtype)
        self.sizeof = oneElement.nbytes
        self.maxsizeof = int(8e9) # in bytes, maximum usage
        
    def refreshFilelists(self):
        self.filenames = fnmatch.filter(os.listdir(self.folderpath), "*"+self.extension)
        self.filepaths = [os.path.join(self.folderpath, i) for i in self.filenames]
        self.reset() # cannot ensure file indexing after list is reset
        
    def reset(self):
        self.fidx = 0
        
    def get(self, numFiles=None, start=None):
        numFiles, start = self._getbounds(numFiles,start)
        
        end = start + numFiles
        if end > len(self.filepaths):
            raise ValueError("Insufficient files remaining.")
        self.fidx = end
        
        fps = self.filepaths[start:end]
        alldata = multiBinReadThreaded(fps, self.numSampsPerFile, self.in_dtype, self.out_dtype)
        return alldata, fps
    
    def _getbounds(self, numFiles, start):
        if start is None:
            start = self.fidx
        if numFiles is None:
            numFiles = len(self.filepaths) - start
            print("Attempting to read entire folder.. %d files.." % (numFiles))
            if numFiles * self.numSampsPerFile * self.sizeof > self.maxsizeof:
                raise MemoryError("Memory requested exceeds internal limit of %d bytes (self.maxsizeof). Please modify this at your own discretion." % (self.maxsizeof))
        return numFiles, start
    
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
            
#%%
class GroupDatabase:
    def __init__(self, dbfilepath: str = "groups.db"):
        self.dbfilepath = dbfilepath
        
        # Make the db
        self.con = sq.connect(dbfilepath)
        self.cur = self.con.cursor()
        
    def addTable(self, tablename: str):
        stmt = "create table if not exists %s(gidx INTEGER UNIQUE, starttime INTEGER, endtime INTEGER)" % tablename
        self.cur.execute(stmt)
        self.con.commit()
        
    def getLatestGroupIdx(self, tablename: str):
        stmt = "select max(gidx) from %s" % tablename
        self.cur.execute(stmt)
        r = self.cur.fetchone()
        print(r)
        return r
        
    def insertGroup(self, tablename: str, gidx: int, starttime: int, endtime: int):
        '''
        Parameters
        ----------
        tablename : str
            Table to insert into.
        gidx : int
            Group index, unique value for each row.
        starttime : int
            Start time, inclusive.
        endtime : int
            End time, inclusive i.e. group is from starttime <= time <= endtime. 

        '''
        stmt = "insert into %s values(?,?,?)" % tablename
        self.cur.execute(stmt, (gidx, starttime, endtime))
        self.con.commit()
        
    def getGroupByIdx(self, tablename: str, gidx: int):
        stmt = "select * from %s where gidx=?" % tablename
        self.cur.execute(stmt, (gidx,))
        _, start, end = self.cur.fetchone()
        return start, end
        
    def getAllGroups(self, tablename: str, returnDataframe: bool = False):
        stmt = "select * from %s" % tablename
        if returnDataframe:
            df = pd.read_sql_query(stmt, self.con)
            return df
        else:
            self.cur.execute(stmt)
            r = self.cur.fetchall()
            return r
        
    
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
            
    def get(self, numFiles=None, start=None):
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
        numFiles,start = self._getbounds(numFiles, start)
        alldata, fps = super().get(numFiles, start)
        fts = self.filetimes[self.fidx-numFiles:self.fidx]
        return alldata, fps, fts
    
    def splitHighAmpSubfolders(self, targetfolderpath:str, selectTimes:list = None,
                               minAmp:float = 1e3, bufFront:int = 1, bufBack:int = 1,
                               fmt:str = "%06d", useDatabase:bool = False):
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
        useDatabase : bool, optional
            Option to not copy files, but instead just write them to a database.

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
            
        # Create the database
        if useDatabase:
            dbfilepath = os.path.join(targetfolderpath,"groups.db")
            print("Writing to database at %s" % dbfilepath)
            gd = GroupDatabase(dbfilepath)
            # tablename = os.path.split(targetfolderpath)[1] # may result in starting with numeric, so don't use this
            tablename = "groups"
            print("Using tablename: %s" % tablename)
            gd.addTable(tablename)
 
        # Loop over groups
        for i in range(len(groupSplitIdx)-1):
            grptimes = selectTimes[groupSplitIdx[i]:groupSplitIdx[i+1]]
            
            if not useDatabase:
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
            else: # if we are using database then append groups
                gidx = i
                starttime = grptimes[0]
                endtime = grptimes[-1]
                
                gd.insertGroup(tablename, int(gidx), int(starttime), int(endtime))
                print("Inserted: gidx=%d, start=%d, end=%d\n" % (gidx, starttime, endtime))
                
            print("-----")
            
    
        
        return selectTimes
        
#%% This is meant to only read one second at a time
class LiveReader(FolderReader):
    def __init__(self, folderpath, numSampsPerFile, extension=".bin", in_dtype=np.int16, out_dtype=np.complex64):
        super().__init__(folderpath, numSampsPerFile, extension, in_dtype, out_dtype)
        # Track by the current filetime
        self.ftnow = int(0)
        # Timeouts
        self.lastTime = 0
        self.timeout = 3
        # Some other optionals
        self.exhaustFolderpath = None # To move files to after reading them
        self.deleteAfter = False
        
        # Calculate input expected size per file
        self.expectedFileSize = numSampsPerFile * np.dtype(in_dtype).itemsize * 2 # x2 for complex
        
    def setTimeout(self, timeout):
        self.timeout = timeout
        
    def setDeleteAfter(self, b: bool):
        self.deleteAfter = b
        
    def setExhaustFolder(self, path):
        self.exhaustFolderpath = path
        
    def getNext(self):
        fp = os.path.join(self.folderpath, "%d%s" % (self.ftnow,self.extension))
        # Check if it exists and is correct file size
        if os.path.isfile(fp) and os.path.getsize(fp) == self.expectedFileSize:
            # Read the file
            alldata = simpleBinRead(fp, self.numSampsPerFile, self.in_dtype, self.out_dtype)
        
            # Update the current filetime
            self.lastTime = time.time()
            self.ftnow = self.ftnow + 1
            
            # Move out if needed
            if self.exhaustFolderpath is not None:
                exhaustpath = os.path.join(self.exhaustFolderpath, "%d%s" % (self.ftnow,self.extension))
                shutil.move(fps[i],exhaustpaths[i])
                
            # Delete if set
            if self.deleteAfter:
                os.remove(fp)
            
            return alldata, fp, self.ftnow
        
        else: # If it doesn't exist or not correct file size, check if we have timed out
            if time.time() - self.lastTime > self.timeout:
                # Search for the next available file time and set it to that
                self.refreshFilelists()
                filetimes = np.sort(np.array([int(os.path.split(i)[-1].split('.')[0]) for i in self.filepaths],dtype=np.int32))
                viableTimes = filetimes[filetimes>self.ftnow]
                if viableTimes.size > 0:
                    self.ftnow = viableTimes[0]
                    print("Found next file at %d" % self.ftnow)
                    
            return None, None, None
        
        
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
        
        
#%% Readers that work with the database
class GroupReadersDB:
    def __init__(self, grpdb, folderpaths, numSampsPerFile, extension=".bin", in_dtype=np.int16, out_dtype=np.complex64):
        self.folderpaths = folderpaths
        # Storage for getter methods later
        self.numSampsPerFile = numSampsPerFile
        self.extension = extension
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        
        self.grpdb = grpdb
        self.resetGroup() # inits cGrp to -1
        self.cStart = None
        self.cEnd = None
        self.cFiles = []
        
    def nextGroup(self, ensure_exists: bool = True):
        self.cGrp += 1
        
        self.cStart, self.cEnd = self.grpdb.getGroupByIdx("groups", self.cGrp) # standard table name
        self.cFiles = [ [os.path.join(folder, "%d" % (i) + self.extension) for i in range(self.cStart, self.cEnd+1)]
                       for folder in self.folderpaths]
        
        if ensure_exists:
            for f in range(len(self.cFiles)):
                for i in range(len(self.cFiles[f])):
                    if not os.path.exists(self.cFiles[f][i]):
                        raise FileNotFoundError(self.cFiles[f][i])
                        
    def getAll(self):
        alldata = []
        for f in range(len(self.cFiles)):
            filelist = self.cFiles[f]
            data = multiBinReadThreaded(filelist, self.numSampsPerFile, in_dtype=self.in_dtype, out_dtype=self.out_dtype)
            alldata.append(data)
        
        return alldata
            
        
    def resetGroup(self):
        self.cGrp = -1
        
        
    
    