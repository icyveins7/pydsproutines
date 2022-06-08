# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:12:24 2022

@author: Lken
"""

import cupy as cp

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
    
    