# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:12:24 2022

@author: Lken

Other generic cupy extensions that don't fit anywhere else..
"""

import cupy as cp
import numpy as np
import os

#%% Convenience functions
def cupyModuleToKernelsLoader(modulefilename: str, kernelNames: list):
    """
    Helper function to generate the CuPy kernel objects from a module.
    The module is expected to reside in the custom_kernels folder.

    Examples:
        kernel1, kernel2 = cupyModuleToKernelsLoader("mymodule.cu", ["mykern1","mykern2"])
        kernel1, = cupyModuleToKernelsLoader("mymodule.cu", "mykern1")
    """
    if isinstance(kernelNames, str):
        kernelNames = [kernelNames]
    kernels = []
    with open(os.path.join(os.path.dirname(__file__), "custom_kernels", modulefilename), "r") as fid:
        _module = cp.RawModule(code=fid.read())
        for kernelName in kernelNames:
            kernels.append(_module.get_function(kernelName))

    return kernels

def cupyRequireDtype(dtype: type, var: cp.ndarray):
    """
    Example: cupyRequireDtype(cp.uint32, myarray)
    """
    if var.dtype != dtype:
        raise TypeError("Must be %s, found %s" % (dtype, var.dtype))
    
def cupyCheckExceedsSharedMem(requestedBytes: int, maximumBytes: int=48000):
    if requestedBytes > maximumBytes:
        raise MemoryError("Shared memory requested %d bytes exceeds maximum %d bytes" % (requestedBytes, maximumBytes))

def requireCupyArray(var: cp.ndarray):
    if not isinstance(var, cp.ndarray):
        raise TypeError("Must be cupy array.")
    
def cupyGetEnoughBlocks(length: int, THREADS_PER_BLOCK: int):
    """
    Gets just enough blocks to cover a certain length.
    Assumes every block will compute THREADS_PER_BLOCK elements.
    """
    NUM_BLKS = length // THREADS_PER_BLOCK
    NUM_BLKS = NUM_BLKS if NUM_BLKS % THREADS_PER_BLOCK == 0 else NUM_BLKS + 1
    return NUM_BLKS


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

_copySlicesToMatrix_32fckernel, _copyEqualSlicesToMatrix_32fckernel, \
    _copyIncrementalEqualSlicesToMatrix_32fckernel = cupyModuleToKernelsLoader(
    "copying.cu", 
    ["copySlicesToMatrix_32fc", "copyEqualSlicesToMatrix_32fc", "copyIncrementalEqualSlicesToMatrix_32fc"])

def cupyCopySlicesToMatrix_32fc(
    d_x: cp.ndarray,
    d_sliceBounds: cp.ndarray,
    rowLength: int=None,
    THREADS_PER_BLOCK: int=128
):
    # Checks
    cupyRequireDtype(cp.complex64, d_x)
    cupyRequireDtype(cp.int32, d_sliceBounds)

    # Allocate output
    numSlices = d_sliceBounds.shape[0]
    if rowLength is None:
        rowLength = cp.max(d_sliceBounds[:,1]-d_sliceBounds[:,0]) # Generate the required length (slower)
    d_out = cp.zeros((numSlices, rowLength), dtype=cp.complex64) # Produce the length requested/generated

    # Execute
    NUM_BLKS = numSlices
    _copySlicesToMatrix_32fckernel(
        (NUM_BLKS,),(THREADS_PER_BLOCK,),
        (d_x, d_x.size, d_sliceBounds,
        numSlices, rowLength, d_out)
    )
    return d_out

def cupyCopyEqualSlicesToMatrix_32fc(
    d_x: cp.ndarray,
    d_xStartIdxs: cp.ndarray,
    rowLength: int,
    d_out: cp.ndarray=None
):
    # Checks
    cupyRequireDtype(cp.complex64, d_x)
    cupyRequireDtype(cp.int32, d_xStartIdxs)

    # Allocate output if not specified
    if d_out is None:
        d_out = cp.zeros((d_xStartIdxs.size, rowLength), dtype=cp.complex64)
    else:
        # Check it
        cupyRequireDtype(cp.complex64, d_out)
        if (d_out.shape != (d_xStartIdxs.size, rowLength)):
            raise ValueError("d_out must have the shape %d, %d" % (d_xStartIdxs.size, rowLength))

    # Execute
    NUM_BLKS = d_out.size // 128 + 1
    _copyEqualSlicesToMatrix_32fckernel(
        (NUM_BLKS,),(128,),
        (d_x, d_x.size, d_xStartIdxs, d_xStartIdxs.size, rowLength, d_out)
    )

    return d_out

def cupyCopyIncrementalEqualSlicesToMatrix_32fc(
    d_x: cp.ndarray,
    startIdx: int,
    increment: int,
    rowLength: int,
    numRows: int,
    d_out: cp.ndarray=None
):
    # Checks
    cupyRequireDtype(cp.complex64, d_x)

    # Define the rectangle each block operates on
    blockRows = 16 if numRows >= 16 else numRows
    blockCols = 256 # Constant for now

    # Allocate output if not specified
    if d_out is None:
        d_out = cp.zeros((numRows, rowLength), dtype=cp.complex64)
    else:
        # Check it
        cupyRequireDtype(cp.complex64, d_out)
        if (d_out.shape!= (numRows, rowLength)):
            raise ValueError("d_out must have the shape %d, %d" % (numRows, rowLength))
        
    # Shared mem requirements
    smReq = blockRows * blockCols * 8
    cupyCheckExceedsSharedMem(smReq)

    # Execute
    NUM_BLKS_X = rowLength // blockCols
    if rowLength % blockCols > 0:
        NUM_BLKS_X += 1
    NUM_BLKS_Y = numRows // blockRows
    if numRows % blockRows > 0:
        NUM_BLKS_Y += 1

    _copyIncrementalEqualSlicesToMatrix_32fckernel(
        (NUM_BLKS_X, NUM_BLKS_Y),
        (256,),
        (d_x, d_x.size, startIdx, increment, numRows, rowLength, blockRows, blockCols, d_out),
        shared_mem=smReq
    )

    return d_out


    

#%%
_argmax3d_uint32kernel, _argmaxAbsRows_cplx64kernel = cupyModuleToKernelsLoader(
    "argmax.cu", 
    ["multiArgmax3d_uint32", "multiArgmaxAbsRows_complex64"]
)
def cupyArgmax3d_uint32(d_x: cp.ndarray, THREADS_PER_BLOCK: int=128, alsoReturnMaxValue: bool=False):
    # Input checks
    if d_x.dtype != cp.uint32:
        raise TypeError("d_x must be uint32.")
    if d_x.ndim != 4:
        raise ValueError("d_x must be 4-d. Argmax taken over the last 3 dimensions.")

    # Extract the dimensions
    numItems, dim1, dim2, dim3 = d_x.shape
    # Allocate output
    d_argmax = cp.zeros((numItems, 3), dtype=cp.uint32)
    # Calculate shared mem
    smReq = THREADS_PER_BLOCK * 4 * 2

    # Execute kernel
    NUM_BLKS = numItems

    if alsoReturnMaxValue:
        d_max = cp.zeros(numItems, dtype=cp.uint32)
        _argmax3d_uint32kernel(
            (NUM_BLKS,),(THREADS_PER_BLOCK,),
            (d_x, numItems, dim1, dim2, dim3, d_argmax, d_max),
            shared_mem=smReq
        )

        return d_argmax, d_max

    else:
        _argmax3d_uint32kernel(
            (NUM_BLKS,),(THREADS_PER_BLOCK,),
            (d_x, numItems, dim1, dim2, dim3, d_argmax, 0), # Set nullptr to last arg
            shared_mem=smReq
        )

        return d_argmax

def cupyArgmaxAbsRows_complex64(
    d_x: cp.ndarray,
    d_argmax: cp.ndarray=None,
    d_max: cp.ndarray=None,
    returnMaxValues: bool=False,
    THREADS_PER_BLOCK: int=128
):
    cupyRequireDtype(cp.complex64, d_x)

    # Allocate output
    numRows, length = d_x.shape
    if d_argmax is None:
        d_argmax = cp.zeros(numRows, dtype=cp.uint32)
    else:
        cupyRequireDtype(cp.uint32, d_argmax)
        if d_argmax.shape != (numRows,):
            raise ValueError("d_argmax shape must be 1D of length %d" % numRows)

    if returnMaxValues:
        if d_max is None:
            d_max = cp.zeros(numRows, dtype=cp.float32)
        else:
            cupyRequireDtype(cp.float32, d_max)
            if d_argmax.shape != (numRows,):
                raise ValueError("d_max shape must be 1D of length %d" % numRows)
    else:
        d_max = 0

    # Shared mem req
    smReq = THREADS_PER_BLOCK * (4 + 4)

    # Execute
    NUM_BLKS = numRows
    _argmaxAbsRows_cplx64kernel(
        (NUM_BLKS,), (THREADS_PER_BLOCK,),
        (d_x, numRows, length, d_argmax, d_max),
        shared_mem=smReq
    )

    if returnMaxValues:
        return d_argmax, d_max
    
    return d_argmax

#%% 
# with open(os.path.join(os.path.dirname(__file__), "custom_kernels", "multiplySlices.cu"), "r") as fid:
#     _module_multiplySlices = cp.RawModule(code=fid.read())
#     _multiplySlicesWithIndexedRowsOptimisticKernel = _module_multiplySlices.get_function("multiplySlicesWithIndexedRowsOptimistic")

_multiplySlicesWithIndexedRowsOptimisticKernel, _slidingMultiplyKernel = cupyModuleToKernelsLoader(
    "multiplySlices.cu", 
    ["multiplySlicesWithIndexedRowsOptimistic", "slidingMultiplyNormalised"]
)

        
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
    if not np.all(d_sliceLengths.get() <= outlength):
        raise ValueError("Some slices exceed the output length!")
    
    # Allocate output
    d_out = cp.zeros((numSlices, outlength), dtype=cp.complex64)

    # Set default number of blocks to fully use the SMs
    if NUM_BLKS is None:
        dev = cp.cuda.Device()
        maxthreads = dev.attributes['MultiProcessorCount'] * dev.attributes['MaxThreadsPerMultiProcessor']
        NUM_BLKS = maxthreads // THREADS_PER_BLOCK
    
    # Calculate shared mem
    smReq = rowLength * 8
    cupyCheckExceedsSharedMem(smReq)

    # Run kernel
    _multiplySlicesWithIndexedRowsOptimisticKernel(
        (NUM_BLKS,),(THREADS_PER_BLOCK,),
        (d_x, xlength, d_rows, rowLength, numRows,
        d_sliceStarts, d_sliceLengths, numSlices, d_rowIdxs,
        d_out, outlength),
        shared_mem=smReq
    )
    
    return d_out

def multiplySlidesNormalised(
    d_x: cp.ndarray, # This is the template/cutout (shorter array)
    d_y: cp.ndarray, # This is the searched input (longer array)
    startIdx: int, # First index of d_y to start searching
    idxlen: int, # Number of searched indices i.e. [startIdx, startIdx+idxlen)
    THREADS_PER_BLOCK: int=128,
    numSlidesPerBlk: int=None
):
    """
    Calls the slidingMultiplyNormalised kernel.
    This kernel maximises usage of shared memory by storing both the template d_x
    and as much of the searched input d_y in 1 block as possible.
    """
    
    # Check that inputs are all 32fc
    cupyRequireDtype(cp.complex64, d_x)
    cupyRequireDtype(cp.complex64, d_y)

    # Check that the slides do not exceed bounds
    if startIdx < 0 or startIdx + idxlen > d_y.size:
        raise ValueError("startIdx and idxlen should be within the bounds of d_y.")

    # Calculate shared mem requirements
    if numSlidesPerBlk is None:
        # Calculate the maximum we can use
        numSlidesPerBlk = (48000 - 2*d_x.nbytes - 8*THREADS_PER_BLOCK) // 8
        if numSlidesPerBlk < 1:
            raise MemoryError("x is too large to use this kernel.")
        print("Using %d slides per block" % numSlidesPerBlk)
    smReq = 2*d_x.nbytes + 8*numSlidesPerBlk - 8 + 8*THREADS_PER_BLOCK # Check kernel for details
    cupyCheckExceedsSharedMem(smReq)

    # Allocate output
    d_pdts = cp.zeros((idxlen, d_x.size), dtype=cp.complex64)    
    
    # Execute kernel
    NUM_BLKS = cupyGetEnoughBlocks(idxlen, THREADS_PER_BLOCK)
    _slidingMultiplyKernel(
        (NUM_BLKS,),(THREADS_PER_BLOCK,),
        (
            d_x,
            d_x.size,
            d_y,
            d_y.size,
            startIdx,
            idxlen,
            d_pdts,
            numSlidesPerBlk
        ),
        shared_mem=smReq
    )

    return d_pdts
    
if __name__ == "__main__":
    from signalCreationRoutines import *
    from verifyRoutines import *
    from timingRoutines import *

    timer = Timer()

    # Create a short signal
    x = randnoise(50, 1, 1, 1).astype(np.complex64)
    # Create a long signal
    y = randnoise(100000, 1, 1, 1).astype(np.complex64)

    # Run the sliding multiply on cpu
    startIdx = 0
    idxlen = y.size - x.size + 1
    out = np.zeros((idxlen, x.size), dtype=np.complex64)
    
    timer.start()
    for i in range(startIdx, idxlen):    
        outnormsq = np.linalg.norm(y[i:i+x.size])
        out[i,:] = y[i:i+x.size] * x / outnormsq
    timer.end("numpy")
    
    # Run the sliding multiply on gpu
    d_x = cp.asarray(x)
    d_y = cp.asarray(y)

    # # Loop over cupy functions (this is extremely inefficient at these dimensions, don't run this)
    # d_out = cp.zeros(out.shape, dtype=cp.complex64)
    # timer.start()
    # for i in range(startIdx, idxlen):
    #     outnormsq = cp.linalg.norm(d_y[i:i+x.size])
    #     d_out[i,:] = d_y[i:i+x.size] * d_x / outnormsq
    # timer.end("cupy")

    # Use the custom kernel
    timer.start()
    d_pdts = multiplySlidesNormalised(
        d_x,
        d_y,
        startIdx,
        idxlen,
        THREADS_PER_BLOCK=32
    )
    timer.end("kernel")

    compareValues(
        d_pdts.get().flatten(),
        out.flatten()
    )


