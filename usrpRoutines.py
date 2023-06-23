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

def simpleBinRead(filename, numSamps=-1, in_dtype=np.int16, out_dtype=np.complex64, offset=0):
    '''
    Simple, single-file complex data reader. numSamps refers to the number of complex samples.
    '''
    if in_dtype == np.complex64 or in_dtype == np.complex128:
        raise TypeError("in_dtype must be a real type. You likely want float32 or float64 instead.")
    
    data = np.fromfile(filename, dtype=in_dtype, count=numSamps*2, offset=offset).astype(np.float32).view(out_dtype)
    
    return data

def multiBinRead(filenames, numSamps, in_dtype=np.int16, out_dtype=np.complex64, offset=0):
    '''
    Simple, multi-file complex data reader. Calls simpleBinRead().
    numSamps refers to the number of complex samples.
    '''
    alldata = np.zeros(len(filenames)*numSamps, out_dtype)
    for i in range(len(filenames)):
        filename = filenames[i]
        alldata[i*numSamps : (i+1)*numSamps] = simpleBinRead(filename, numSamps, in_dtype, out_dtype, offset=offset)
    
    return alldata

def multiBinReadThreaded(filenames, numSamps, in_dtype=np.int16, out_dtype=np.complex64, offset=0, threads=2):
    '''
    Threaded multi-file reader. Anything more than 2 threads is usually insignificant/actually worse.
    '''
    alldata = np.zeros(len(filenames)*numSamps, out_dtype)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        future_load = {executor.submit(simpleBinRead, filenames[i], numSamps, in_dtype, out_dtype, offset=offset): i for i in np.arange(len(filenames))}
        for future in concurrent.futures.as_completed(future_load):
            i = future_load[future] # reference dictionary for index
            alldata[i*numSamps: (i+1)*numSamps] = future.result() # write to the mass array
            
    return alldata

def futureBinRead(executor: concurrent.futures.ThreadPoolExecutor, filename: str, numSamps: int, in_dtype: type=np.int16, offset: int=0):
    '''
    Uses an existing ThreadPoolExecutor to submit a single bin read, akin to simpleBinRead.
    The result array can be extracted via future.result(). This allows you to pre-load data from disk in another thread.

    Parameters
    ----------
    executor : concurrent.futures.ThreadPoolExecutor
        Pre-initialized ThreadPoolExecutor.
    filename : str
        Single filepath to load.
    numSamps : int
        Number of complex samples to load.
    in_dtype : type, optional
        Data type of each component of the samples (either I or Q). The default is np.int16.
        Do not specify complex64 or complex128 types.
    offset : int, optional
        The offset in bytes from the start of the file. The default is 0.

    Returns
    -------
    future : concurrent.futures.Future
        Future object which contains the data array. Use future.result() to extract the array.

    '''
    if in_dtype == np.complex64 or in_dtype == np.complex128:
        raise TypeError("in_dtype must be a real type. You likely want float32 or float64 instead.")
    
    future = executor.submit(np.fromfile, filename, dtype=in_dtype, count=numSamps*2, offset=offset)
    return future
    
def isInt16Clipping(data, threshold=32000):
    if data.dtype == np.complex64:
        fdata = data.view(np.float32)
    elif data.dtype == np.complex128:
        fdata = data.view(np.float64)
    else: # If just in int16s, it's just an interleaved real array so it's fine
        fdata = data
    
    return np.any(np.abs(fdata) > threshold)

def getAvailableSubdirpaths(maindir):
    contents = os.listdir(maindir)
    subdirpaths = [os.path.join(maindir,i) for i in contents]
    subdirpaths = [i for i in subdirpaths if os.path.isdir(i)]
    
    return subdirpaths
    

#%% Convenience classes
class FolderReader:
    def __init__(self, folderpath, numSampsPerFile, extension=".bin", in_dtype=np.int16, out_dtype=np.complex64, ignoreInsufficientData=True):
        self.numSampsPerFile = numSampsPerFile
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype

        self.folderpath = folderpath
        self.extension = extension
        self.ignoreInsufficientData = ignoreInsufficientData
        self.refreshFilelists()
        
        self.executor = concurrent.futures.ThreadPoolExecutor(1) # Generally good to have only 1 thread on concurrency for reading
        self.futures = []
        
        self.reset()
        
        # Calculate memory footprint
        oneElement = np.array([0],dtype=self.out_dtype)
        self.sizeof = oneElement.nbytes
        self.maxsizeof = int(8e9) # in bytes, maximum usage

    @property
    def hasMoreFiles(self):
        return self.fidx < len(self.filepaths)
        
    def refreshFilelists(self):
        baseObj = np.zeros(1, dtype=self.in_dtype)
        reqMinFilesize = baseObj.itemsize * 2 * self.numSampsPerFile if np.isrealobj(baseObj) else baseObj.itemsize * self.numSampsPerFile
        # reqMinFilesize = np.zeros(1, dtype=self.in_dtype).itemsize * 2 * self.numSampsPerFile

        dircontents = os.listdir(self.folderpath)
        if self.ignoreInsufficientData:
            dircontents = [i for i in dircontents if 
                            os.path.getsize(os.path.join(self.folderpath,i)) >= reqMinFilesize]
        self.filenames = fnmatch.filter(dircontents, "*"+self.extension)
        self.filepaths = [os.path.join(self.folderpath, i) for i in self.filenames]
        self.reset() # cannot ensure file indexing after list is reset
        
    def reset(self):
        '''Brings the reader back to the first file.'''
        self.fidx = 0

    def startAtIndex(self, i: int):
        '''
        Moves the reader to the i'th file.
        Note: calling this explicitly will discard all prefetched data as we cannot be sure we are in order any more.
        '''
        self.fidx = i
        self.futures.clear()

    def get(self, numFiles: int, prefetch: int=0):
        '''
        Reads the next (numFiles) files,
        and prefetches an optional number of them.
        '''

        startingFileIdx = self.fidx
        remainderToRead = numFiles
        data = np.zeros((numFiles, self.numSampsPerFile), dtype=self.out_dtype)
        i = 0 # Counter for what we've read
        # First iterate over the prefetched data, if any
        while len(self.futures) > 0 and remainderToRead > 0:
            future = self.futures.pop(0) # Pop the earliest one
            data[i,:] = future.result().astype(np.float32).view(self.out_dtype) # Save it to our output
            i += 1
            remainderToRead -= 1 # Every file we read, we remove from the remainder
            self.fidx += 1 # We also increment the internal counter

        # If there is a remainder left then we have to explicitly read it ourselves now
        while remainderToRead > 0:
            print("Manually retrieving %d files.." % (remainderToRead))
            data[i,:] = simpleBinRead(
                self.filepaths[self.fidx],
                self.numSampsPerFile,
                self.in_dtype,
                self.out_dtype
            )
            i += 1
            self.fidx += 1
            remainderToRead -= 1

        # And then we prefetch the additional amount required
        additional = prefetch - len(self.futures)
        for a in range(additional):
            if self.fidx + a < len(self.filepaths): # We only prefetch if there's files left
                self.futures.append(
                    futureBinRead(
                        self.executor, self.filepaths[self.fidx+a],
                        self.numSampsPerFile, self.in_dtype
                    )
                )

        # Carve out the filepaths to return
        fps = self.filepaths[startingFileIdx:self.fidx]
        # Flatten
        data = data.reshape(-1) # This is faster than flatten()?

        return data, fps
        
    def getNextFile(self):
        # Specifically only retrieve the next single file, provided as a convenience
        numFiles, start = self._getbounds(1, None)
        
        end = start + numFiles
        if end > len(self.filepaths):
            raise ValueError("Insufficient files remaining.")
        self.fidx = end
        
        fps = self.filepaths[start]
        alldata = simpleBinRead(fps, self.numSampsPerFile, self.in_dtype, self.out_dtype)
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
            
    def getFinalTime(self):
        return self.filetimes[-1]
            
    def startAtTime(self, startTime: int):
        '''
        Skips reader to the file at specified time.
        Note that prefetched data will be cleared when you do this.
        '''
        targetFidx = np.argwhere(self.filetimes == startTime)[0,0]
        self.startAtIndex(targetFidx)
            
    def getPathByTime(self, reqTime):
        return self.filepaths[np.argwhere(self.filetimes == reqTime).flatten()[0]]
    
    def getFileByTime(self, reqTime):
        if isinstance(reqTime, int):
            path = self.getPathByTime(reqTime)
            alldata = multiBinReadThreaded([path], self.numSampsPerFile, self.in_dtype, self.out_dtype)
            return alldata, path
        else: # if array-like
            paths = [self.getPathByTime(i) for i in reqTime]
            alldata = multiBinReadThreaded(paths, self.numSampsPerFile, self.in_dtype, self.out_dtype)
            return alldata, paths

    def get(self, numFiles: int, prefetch: int=0):
        '''
        Extracts the desired number of files and optionally prefetches extra files.

        Parameters
        ----------
        numFiles : int
            The number of files to output. If some are prefetched it will manually pull the remainder.
        prefetch : int, optional
            The additional number of files to prefetch, to speed up consequent calls. By default 0.

        Returns
        -------
        data : np.ndarray
            Output array. Has length numFiles * numSampsPerFile.
        fps : list
            List of file paths that were read, in order.
        fts : np.ndarray
            List of file times that were read, in order.
        '''

        data, fps = super().get(numFiles, prefetch)
        fts = self.filetimes[self.fidx-numFiles:self.fidx]
        return data, fps, fts
    
    def splitHighAmpSubfolders(self, targetfolderpath: str, selectTimes: list=None,
                               minAmp: float=1e3, bufFront: int=1, bufBack: int=1, onlyExtractTimes: bool=False, onlyExtractGroups: bool=False,
                               fmt: str="%06d", useDatabase: bool=False, dbfilepath: str=None):
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
        
        if len(selectTimes) == 0:
            raise IndexError("No groups were found. Perhaps try lowering the minAmp threshold?")
        
        if onlyExtractTimes:
            return selectTimes # return here directly
        
        # Now pull groups out where the difference is more than 1
        groupSplitIdx = np.hstack((0,(np.argwhere(np.diff(selectTimes) > 1) + 1).flatten(), len(selectTimes)))
        if onlyExtractGroups: # This will just return as a list of lists
            return [selectTimes[groupSplitIdx[i]:groupSplitIdx[i+1]] for i in np.arange(groupSplitIdx.size-1)]
        
        
        # Create the database
        if useDatabase:
            if dbfilepath is None:
                dbfilepath = os.path.join(targetfolderpath,"groups.db") # Default if not specified
            print("Writing to database at %s" % dbfilepath)
            gd = GroupDatabase(dbfilepath)
            # tablename = os.path.split(targetfolderpath)[1] # may result in starting with numeric, so don't use this
            tablename = "groups"
            print("Using tablename: %s" % tablename)
            gd.addTable(tablename)
            
        # Create the main dir
        elif not os.path.isdir(targetfolderpath):
            os.makedirs(targetfolderpath)
            print("Created %s" % targetfolderpath)
            
        
 
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
            
        # If using database, append to the metadata so that we can track progress
        if useDatabase:
            gd.updateMetatable(self.getFinalTime()) # Update to the latest available time
            
    
        
        return selectTimes
    
#%%
class GroupReader(SortedFolderReader):
    '''
    This reader is inherited from SortedFolderReader, but adds additional methods to return and track data in groups.
    Here, a group is defined as a series of files that are saved one after another i.e. one second difference in file times.
    If two consecutive files differ by more than one second, the group ends.
    The data can then be extracted group-wise.
    Folder structure should look something like this:
        0.bin (group 0)
        1.bin (group 0)
        4.bin (group 1)
        7.bin (group 2)
        8.bin (group 2)
    '''
    def __init__(self, folderpath, numSampsPerFile, extension=".bin", in_dtype=np.int16, out_dtype=np.complex64):
        super().__init__(folderpath, numSampsPerFile, extension, in_dtype, out_dtype, ensure_incremental=False)
        self.groups = self._parseGroups()
        self.cGrp = -1
        
    def _parseGroups(self):
        # Find where the times differ by more than 1s
        d = np.diff(self.filetimes)
        # Place breaks where these occur; note that +1 is required, and we pad with 0 (to include the first) and the total length (to include the last)
        ii = np.hstack((0, np.argwhere(d > 1).flatten() + 1, self.filetimes.size))
        # Define the groups
        groups = [self.filetimes[ii[j]:ii[j+1]] for j in range(ii.size-1)]
        return groups
    
    def reset(self):
        super().reset()
        self.cGrp = -1 # Reset the group index too
        
    @property
    def hasMoreGroups(self):
        return self.cGrp + 1 < len(self.groups)
    
    @property
    def numGroups(self):
        return len(self.groups)
        
    def getGroup(self, prefetchNextGroup: bool=False):
        '''
        Extracts N files, where N is the number of files in the next group.

        Parameters
        ----------
        prefetchNextGroup : bool, optional
            Prefetches files in the following group. The default is False.

        Returns
        -------
        data : np.ndarray
            Output array. Has length numFiles * numSampsPerFile.
        fps : list
            List of file paths that were read, in order.
        fts : np.ndarray
            List of file times that were read, in order.
            Should not have any files that are more than 1s apart.
        '''
        # First increment the group index
        self.cGrp += 1
        # Get the number of files in the group
        numFiles = self.groups[self.cGrp].size
        # Check how many to prefetch if desired
        if prefetchNextGroup and self.cGrp+1 < len(self.groups):
            prefetch = self.groups[self.cGrp+1].size
        else:
            prefetch = 0
        # Call the standard getter
        data, fps, fts = self.get(numFiles, prefetch)
        return data, fps, fts

            
#%%
class GroupDatabase:
    def __init__(self, dbfilepath: str = "groups.db"):
        self.dbfilepath = dbfilepath
        
        # Make the db
        self.con = sq.connect(dbfilepath)
        self.cur = self.con.cursor()
        
        # Add the meta table
        self.addMetatable()
        
    def addMetatable(self):
        self.cur.execute("create table if not exists meta(lastfiletime INTEGER)")
        self.con.commit()
        
    def updateMetatable(self, lastfiletime: int):
        self.cur.execute("insert or replace into meta values(?)", (int(lastfiletime),))
        self.con.commit()
        
    def getLastProcessedTime(self):
        self.cur.execute("select lastfiletime from meta")
        return self.cur.fetchone()
        
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
        stmt = "insert or replace into %s values(?,?,?)" % tablename
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
        
        
#%% This is meant to only read one second at a time
class LiveReader(FolderReader):
    """
    This is a reader that is meant for live, 1-second at a time, recordings.
    The best way to use this is to place this in a while loop,
    and call getNext() repeatedly, which returns None objects when the next file is absent.
    The user can then 'continue' the loop or perform some other operations.

    This reader will automatically search for the next earliest file in the folder if
    it has timed-out (which occurs a configurable number of seconds after the last successful read).

    """
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

    def setExpectedFileSize(self, expectedFileSizeBytes: int):
        self.expectedFileSize = expectedFileSizeBytes
        
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
                shutil.move(fp,exhaustpath)
                
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



# These classes are best suited for sparse recordings
class FolderedGroupReader:
    def __init__(self, folderpath, numSampsPerFile, extension=".bin", in_dtype=np.int16, out_dtype=np.complex64, ensure_incremental=True):
        self.folderpath = folderpath
        self.cGrp = -1
        self.groups = [i for i in os.listdir(folderpath) if os.path.isdir(os.path.join(folderpath, i))]
        
        # Storage for getter methods later
        self.numSampsPerFile = numSampsPerFile
        self.extension = extension
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.ensure_incremental = ensure_incremental
        
    def resetGroup(self):
        self.cGrp = -1
        
    def nextGroup(self):
        self.cGrp += 1
        reader = SortedFolderReader(os.path.join(self.folderpath, self.groups[self.cGrp]),
                                    self.numSampsPerFile, self.extension,
                                    self.in_dtype, self.out_dtype, self.ensure_incremental)
        data, filepaths, filetimes = reader.get(len(reader.filenames))
        return data, filepaths, filetimes

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
        if not isinstance(folderpaths, list):
            raise TypeError("folderpaths must be a list, even if only one folder is being used!")
            # TODO: probably refactor to just one, not clear how to use more than one at a time?
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
        
        return alldata, self.cStart, self.cEnd
            
        
    def resetGroup(self):
        self.cGrp = -1
        
        
    
    