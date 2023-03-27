# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:12:24 2022

@author: Lken

Other generic cupy extensions that don't fit anywhere else..
"""

import cupy as cp
import os

#%% A block-group paired kernel copy
copy_groups_kernel32fc = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void copy_groups_kernel32fc(const complex<float>* d_x, 
                                complex<float>* d_y, 
                                const int *xi0, 
                                const int *lengths,
                                const int *yi0){
                
        // Each block works on its own group
        int srcStart = xi0[blockIdx.x];
        int dstStart = yi0[blockIdx.x];
        int len = lengths[blockIdx.x]; 
        for (int i = threadIdx.x; i < len; i = i+blockDim.x)
        {
            d_y[dstStart + i] = d_x[srcStart + i];
        }
    }
    ''','copy_groups_kernel32fc')
    
def cupyCopyGroups32fc(x: cp.ndarray, y: cp.ndarray,
                       xStarts: cp.ndarray, yStarts: cp.ndarray,
                       lengths: cp.ndarray, threads_per_blk: int=256):
    '''
    Performs multiple groups of copies with a single kernel, avoiding pythonic
    interpreter loops. Note that all arrays are expected to be already on the 
    GPU.
    Warnings: 
    No dtype checking is performed for the arrays.
    Overlaps of groups and bounds checks for the indices are also not performed.
    
    Parameters
    ----------
    x : cp.ndarray, cp.complex64
        Source.
    y : cp.ndarray, cp.complex64
        Destination.
    xStarts : cp.ndarray, cp.int32
        Starting index for each group from source.
    yStarts : cp.ndarray, cp.int32
        Starting index for each group from destination.
    lengths : cp.ndarray, cp.int32
        Length of each group.
    threads_per_blk : int, optional
        Threads per block to use. The default is 256.

    Returns
    -------
    None.

    '''
    
    assert(len(xStarts)==len(yStarts) and len(xStarts)==len(lengths))
    
    NUM_BLOCKS = len(xStarts)
    copy_groups_kernel32fc((NUM_BLOCKS,), (threads_per_blk,),
                           (x, y, xStarts, lengths, yStarts))


#%% 
with open(os.path.join(os.path.dirname(__file__), "custom_kernels", "multiplySlices.cu"), "r") as fid:
    _module_multiplySlices = cp.RawModule(code=fid.read())
    _multiplySlicesWithIndexedRowsOptimisticKernel = _module_multiplySlices.get_function("multiplySlicesWithIndexedRowsOptimistic")


        
def multiplySlicesOptimistically(
    d_x: cp.ndarray, d_rows: cp.ndarray,
    d_sliceStarts: cp.ndarray, d_sliceLengths: cp.ndarray, d_rowIdxs: cp.ndarray,
    THREADS_PER_BLOCK: int=256, NUM_BLKS: int=None, outlength: int=None
):

    # Require complex64 types
    if d_x.dtype != cp.complex64 or d_rows.dtype != cp.complex64:
        raise TypeError("Inputs x and rows must be complex64.")

    # Require integers
    if d_sliceStarts.dtype != cp.int32 or d_sliceLengths.dtype != cp.int32 or d_rowIdxs.dtype != cp.int32:
        raise TypeError("sliceStarts, sliceLengths & rowIdxs must be int32.")

    # Require dimensions to be correct
    if d_x.ndim != 1:
        raise ValueError("x should be 1-dimensional.")

    if d_rows.ndim != 2:
        raise ValueError("rows should be 2-dimensional.")

    # Get number of slices and check inputs
    numSlices = d_sliceStarts.size
    if d_sliceLengths.size != numSlices or d_rowIdxs.size != numSlices:
        raise ValueError("sliceLengths and rowIdxs should be same length as sliceStarts.")

    # Define all the lengths for clarity
    xlength = d_x.size
    numRows, rowLength = d_rows.shape

    # Check if outlength satisfies all the slices
    if outlength is None:
        outlength = rowLength
    if not np.all(d_sliceLengths.get() < outlength):
        raise ValueError("Some slices exceed the output length!")
    
    # Allocate output
    d_out = cp.zeros((numSlices, outlength), dtype=cp.complex64)

    # Set default number of blocks to fully use the SMs
    if NUM_BLKS is None:
        dev = cupy.cuda.Device()
        maxthreads = dev.attributes['MultiProcessorCount'] * dev.attributes['MaxThreadsPerMultiProcessor']
        NUM_BLKS = maxthreads // THREADS_PER_BLOCK
    
    # Calculate shared mem
    smReq = rowLength * 8

    # Run kernel
    _multiplySlicesWithIndexedRowsOptimisticKernel(
        (NUM_BLKS,),(THREADS_PER_BLOCK,),
        (d_x, xlength, d_rows, rowLength, numRows,
        d_sliceStarts, d_sliceLengths, numSlices, d_rowIdxs,
        d_out, outlength),
        shared_mem=smReq
    )
    
    return d_out

    
    