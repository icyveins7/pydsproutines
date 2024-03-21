# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:12:24 2022

@author: Lken

Other generic cupy extensions that don't fit anywhere else..
"""

import cupy as cp
import numpy as np

from cupyHelpers import *


# %% A block-group paired kernel copy
copy_groups_kernel32fc = cp.RawKernel(
    r"""
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
    """,
    "copy_groups_kernel32fc",
)


def cupyCopyGroups32fc(
    x: cp.ndarray,
    y: cp.ndarray,
    xStarts: cp.ndarray,
    yStarts: cp.ndarray,
    lengths: cp.ndarray,
    threads_per_blk: int = 256,
):
    """
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

    """

    assert len(xStarts) == len(yStarts) and len(xStarts) == len(lengths)

    NUM_BLOCKS = len(xStarts)
    copy_groups_kernel32fc(
        (NUM_BLOCKS,), (threads_per_blk,), (x, y, xStarts, lengths, yStarts)
    )


kernels, _ = cupyModuleToKernelsLoader(
    "copying.cu",
    [
        "copySlicesToMatrix_32fc",
        "copyEqualSlicesToMatrix_32fc",
        "copyIncrementalEqualSlicesToMatrix_32fc",
    ],
)
(
    _copySlicesToMatrix_32fckernel,
    _copyEqualSlicesToMatrix_32fckernel,
    _copyIncrementalEqualSlicesToMatrix_32fckernel,
) = kernels  # Unpack


def cupyCopySlicesToMatrix_32fc(
    d_x: cp.ndarray,
    d_sliceBounds: cp.ndarray,
    rowLength: int = None,
    THREADS_PER_BLOCK: int = 128,
):
    # Checks
    cupyRequireDtype(cp.complex64, d_x)
    cupyRequireDtype(cp.int32, d_sliceBounds)

    # Allocate output
    numSlices = d_sliceBounds.shape[0]
    if rowLength is None:
        rowLength = cp.max(
            d_sliceBounds[:, 1] - d_sliceBounds[:, 0]
        )  # Generate the required length (slower)
    d_out = cp.zeros(
        (numSlices, rowLength), dtype=cp.complex64
    )  # Produce the length requested/generated

    # Execute
    NUM_BLKS = numSlices
    _copySlicesToMatrix_32fckernel(
        (NUM_BLKS,),
        (THREADS_PER_BLOCK,),
        (d_x, d_x.size, d_sliceBounds, numSlices, rowLength, d_out),
    )
    return d_out


def cupyCopyEqualSlicesToMatrix_32fc(
    d_x: cp.ndarray, d_xStartIdxs: cp.ndarray, rowLength: int, d_out: cp.ndarray = None
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
        if d_out.shape != (d_xStartIdxs.size, rowLength):
            raise ValueError(
                "d_out must have the shape %d, %d" % (d_xStartIdxs.size, rowLength)
            )

    # Execute
    NUM_BLKS = d_out.size // 128 + 1
    _copyEqualSlicesToMatrix_32fckernel(
        (NUM_BLKS,),
        (128,),
        (d_x, d_x.size, d_xStartIdxs, d_xStartIdxs.size, rowLength, d_out),
    )

    return d_out


def cupyCopyIncrementalEqualSlicesToMatrix_32fc(
    d_x: cp.ndarray,
    startIdx: int,
    increment: int,
    rowLength: int,
    numRows: int,
    d_out: cp.ndarray = None,
):
    # Checks
    cupyRequireDtype(cp.complex64, d_x)

    # Define the rectangle each block operates on
    blockRows = 16 if numRows >= 16 else numRows
    blockCols = 256  # Constant for now

    # Allocate output if not specified
    if d_out is None:
        d_out = cp.zeros((numRows, rowLength), dtype=cp.complex64)
    else:
        # Check it
        cupyRequireDtype(cp.complex64, d_out)
        if d_out.shape != (numRows, rowLength):
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
        (
            d_x,
            d_x.size,
            startIdx,
            increment,
            numRows,
            rowLength,
            blockRows,
            blockCols,
            d_out,
        ),
        shared_mem=smReq,
    )

    return d_out


# %%
kernels, _ = cupyModuleToKernelsLoader(
    "argmax.cu", ["multiArgmax3d_uint32", "multiArgmaxAbsRows_complex64"]
)
_argmax3d_uint32kernel, _argmaxAbsRows_cplx64kernel = kernels  # Unpack


def cupyArgmax3d_uint32(
    d_x: cp.ndarray, THREADS_PER_BLOCK: int = 128, alsoReturnMaxValue: bool = False
):
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
            (NUM_BLKS,),
            (THREADS_PER_BLOCK,),
            (d_x, numItems, dim1, dim2, dim3, d_argmax, d_max),
            shared_mem=smReq,
        )

        return d_argmax, d_max

    else:
        _argmax3d_uint32kernel(
            (NUM_BLKS,),
            (THREADS_PER_BLOCK,),
            (d_x, numItems, dim1, dim2, dim3, d_argmax, 0),  # Set nullptr to last arg
            shared_mem=smReq,
        )

        return d_argmax


def cupyArgmaxAbsRows_complex64(
    d_x: cp.ndarray,
    d_argmax: cp.ndarray = None,
    d_max: cp.ndarray = None,
    returnMaxValues: bool = False,
    THREADS_PER_BLOCK: int = 128,
    useNormSqInstead: bool = False,
):
    """
    Performs a CUDA block->row argmax along the columns for each row.
    Optionally returns the max values themselves, along with the argmax indices.
    Optionally also allowed to use magnSq (i.e. abs()^2) instead of just abs().
    """
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
        (NUM_BLKS,),
        (THREADS_PER_BLOCK,),
        (d_x, numRows, length, d_argmax, d_max, useNormSqInstead),
        shared_mem=smReq,
    )

    if returnMaxValues:
        return d_argmax, d_max

    return d_argmax


# %%
(
    _complex_magnSq_kernel_floatfloat,
    _complex_magnSq_kernel_floatdouble,
    _complex_magnSq_kernel_doubledouble,
), _ = cupyModuleToKernelsLoader(
    "complex_magn.cu",
    [
        "complex_magnSq_kernel<float,float>",
        "complex_magnSq_kernel<float,double>",
        "complex_magnSq_kernel<double,double>",
    ],
)


def cupyComplexMagnSq(
    d_x: cp.ndarray, out_dtype: cp.dtype = cp.float64, THREADS_PER_BLOCK: int = 128
):
    """
    Simple grid-stride kernel invocation to calculate the magnitude squared
    of a complex array.

    Has 3 input-output flavour pairs:
    complex<float> -> float
    complex<float> -> double
    complex<double> -> double

    Parameters
    ----------
    d_x : cp.ndarray
        Input array.
    out_dtype : cp.dtype, optional
        Output data type. Defaults to float64.
    THREADS_PER_BLOCK : int, optional
        Number of threads per block. Defaults to 128.
    """
    NUM_BLKS = cupyGetEnoughBlocks(d_x.size, THREADS_PER_BLOCK)

    # Call appropriate kernel for appropriate type
    if out_dtype == cp.float32:
        cupyRequireDtype(cp.complex64, d_x)
        # Create output
        d_magnSq = cp.zeros(d_x.shape, dtype=cp.float32)
        # Call kernel
        _complex_magnSq_kernel_floatfloat(
            (NUM_BLKS,), (THREADS_PER_BLOCK,), (d_x, d_x.size, d_magnSq)
        )

        return d_magnSq

    elif out_dtype == cp.float64:
        # Create output
        d_magnSq = cp.zeros(d_x.shape, dtype=cp.float64)

        if d_x.dtype == cp.complex64:
            _complex_magnSq_kernel_floatdouble(
                (NUM_BLKS,), (THREADS_PER_BLOCK,), (d_x, d_x.size, d_magnSq)
            )
        elif d_x.dtype == cp.complex128:
            _complex_magnSq_kernel_doubledouble(
                (NUM_BLKS,), (THREADS_PER_BLOCK,), (d_x, d_x.size, d_magnSq)
            )
        else:
            raise TypeError("d_x must be complex64 or complex128.")

        return d_magnSq


# %%
(
    _multiplySlicesWithIndexedRowsOptimisticKernel,
    _slidingMultiplyKernel,
    _multiTemplateSlidingDotKernel,
), _ = cupyModuleToKernelsLoader(
    "multiplySlices.cu",
    [
        "multiplySlicesWithIndexedRowsOptimistic",
        "slidingMultiplyNormalised",
        "multiTemplateSlidingDotProduct",
    ],
)


def multiplySlicesOptimistically(
    d_x: cp.ndarray,
    d_rows: cp.ndarray,
    d_sliceStarts: cp.ndarray,
    d_sliceLengths: cp.ndarray,
    d_rowIdxs: cp.ndarray,
    THREADS_PER_BLOCK: int = 256,
    NUM_BLKS: int = None,
    outlength: int = None,
):

    # Require complex64 types
    if d_x.dtype != cp.complex64 or d_rows.dtype != cp.complex64:
        raise TypeError("Inputs x and rows must be complex64.")

    # Require integers
    if (
        d_sliceStarts.dtype != cp.int32
        or d_sliceLengths.dtype != cp.int32
        or d_rowIdxs.dtype != cp.int32
    ):
        raise TypeError("sliceStarts, sliceLengths & rowIdxs must be int32.")

    # Require dimensions to be correct
    if d_x.ndim != 1:
        raise ValueError("x should be 1-dimensional.")

    if d_rows.ndim != 2:
        raise ValueError("rows should be 2-dimensional.")

    # Get number of slices and check inputs
    numSlices = d_sliceStarts.size
    if d_sliceLengths.size != numSlices or d_rowIdxs.size != numSlices:
        raise ValueError(
            "sliceLengths and rowIdxs should be same length as sliceStarts."
        )

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
        maxthreads = (
            dev.attributes["MultiProcessorCount"]
            * dev.attributes["MaxThreadsPerMultiProcessor"]
        )
        NUM_BLKS = maxthreads // THREADS_PER_BLOCK

    # Calculate shared mem
    smReq = rowLength * 8
    cupyCheckExceedsSharedMem(smReq)

    # Run kernel
    _multiplySlicesWithIndexedRowsOptimisticKernel(
        (NUM_BLKS,),
        (THREADS_PER_BLOCK,),
        (
            d_x,
            xlength,
            d_rows,
            rowLength,
            numRows,
            d_sliceStarts,
            d_sliceLengths,
            numSlices,
            d_rowIdxs,
            d_out,
            outlength,
        ),
        shared_mem=smReq,
    )

    return d_out


def multiplySlidesNormalised(
    d_x: cp.ndarray,  # This is the template/cutout (shorter array)
    d_y: cp.ndarray,  # This is the searched input (longer array)
    startIdx: int,  # First index of d_y to start searching
    idxlen: int,  # Number of searched indices i.e. [startIdx, startIdx+idxlen)
    THREADS_PER_BLOCK: int = 128,
    numSlidesPerBlk: int = None,
    coefficient: float = None,  # Extra constant coefficient to multiply, this should default to norm of d_x
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

    # Compute norm of x if required
    if coefficient is None:
        coefficient = cp.array(
            [cp.linalg.norm(d_x)], cp.float64
        )  # Create a single element 1D array
    # Make sure the type is okay
    cupyRequireDtype(cp.float64, coefficient)
    if coefficient.size != 1:
        raise ValueError("coefficient should be a single element 1D array.")

    # Calculate shared mem requirements
    if numSlidesPerBlk is None:
        # Calculate the maximum we can use
        numSlidesPerBlk = (48000 - 2 * d_x.nbytes - 8 * THREADS_PER_BLOCK) // 8
        if numSlidesPerBlk < 1:
            raise MemoryError("x is too large to use this kernel.")
        print("Using %d slides per block" % numSlidesPerBlk)
    smReq = (
        2 * d_x.nbytes + 8 * numSlidesPerBlk - 8 + 8 * THREADS_PER_BLOCK
    )  # Check kernel for details
    cupyCheckExceedsSharedMem(smReq)

    # Allocate output
    d_pdts = cp.empty((idxlen, d_x.size), dtype=cp.complex64)

    # Execute kernel
    NUM_BLKS = cupyGetEnoughBlocks(idxlen, numSlidesPerBlk)
    _slidingMultiplyKernel(
        (NUM_BLKS,),
        (THREADS_PER_BLOCK,),
        (
            d_x,
            d_x.size,
            d_y,
            d_y.size,
            startIdx,
            idxlen,
            d_pdts,
            numSlidesPerBlk,
            coefficient,
        ),
        shared_mem=smReq,
    )

    return d_pdts


def multiTemplateSlidingDotProduct(
    d_x: cp.ndarray,  # This is the searched input (longer array)
    d_templates: cp.ndarray,  # This is the matrix of templates (1 row = 1 template)
    startIdx: int,  # First index of d_x to start searching
    idxlen: int,  # Number of searched indices i.e. [startIdx, startIdx+idxlen)
    d_templateEnergies: cp.ndarray = None,
    numSlidesPerBlk: int = None,
    THREADS_PER_BLOCK: int = 128,
):
    # Check 32fc input arrays
    cupyRequireDtype(cp.complex64, d_x)
    cupyRequireDtype(cp.complex64, d_templates)

    # Ensure templates is 2D
    if d_templates.ndim != 2:
        raise ValueError("Templates should be 2D; each row is an individual template.")
    numTemplates, templateLength = d_templates.shape

    # Check that the slides do not exceed bounds
    if startIdx < 0:
        raise ValueError("startIdx should be >= 0.")
    endIdx = startIdx + idxlen - 1
    if endIdx + templateLength - 1 >= d_x.size:
        raise ValueError(
            "final slide index (%d) should be within the bounds of d_x (%d)."
            % (endIdx + templateLength - 1, d_x.size)
        )

    # Pre-compute the template energies if not provided
    if d_templateEnergies is None:
        # d_templateEnergies = cp.linalg.norm(d_templates, axis=1) # DO NOT USE THIS. nsys will complain for some reason (cupy library bug?)
        d_templateEnergies = cp.sum(cp.abs(d_templates) ** 2, axis=1)
    # Ensure types of template energies
    cupyRequireDtype(cp.float32, d_templateEnergies)
    if d_templateEnergies.size != numTemplates:
        raise ValueError(
            "d_templateEnergies size should be equal to the rows of templates."
        )

    # Calculate shared mem requirements
    smReq = (
        templateLength * 16 + THREADS_PER_BLOCK * 8
    )  # This is the minimal requirement
    if numSlidesPerBlk is None:
        # Calculate the maximum we can use
        numSlidesPerBlk = (48000 - smReq) // 16
        if numSlidesPerBlk < 1:
            raise MemoryError("x is too large to use this kernel.")
        print("Using %d slides per block" % numSlidesPerBlk)
    smReq += numSlidesPerBlk * 16
    print("smReq = %d" % (smReq))
    cupyCheckExceedsSharedMem(smReq)

    # Allocate output
    d_templateIdx = cp.empty(idxlen, dtype=cp.int32)
    d_qf2 = cp.empty(idxlen, dtype=cp.float32)

    # Execute kernel
    NUM_BLKS = cupyGetEnoughBlocks(idxlen, numSlidesPerBlk)
    print("NUM_BLKS = %d" % (NUM_BLKS))
    _multiTemplateSlidingDotKernel(
        (NUM_BLKS,),
        (THREADS_PER_BLOCK,),
        (
            d_templates,
            d_templateEnergies,
            d_templates.shape[0],
            templateLength,
            d_x[startIdx : startIdx + idxlen + templateLength - 1],
            idxlen + templateLength - 1,
            numSlidesPerBlk,
            d_templateIdx,
            d_qf2,
        ),
        shared_mem=smReq,
    )

    return d_templateIdx, d_qf2


# %% Peak finding kernels
peakfindingKernels, _ = cupyModuleToKernelsLoader("peakfinding.cu", ["findLocalMaxima"])
(_findLocalMaximaKernel,) = peakfindingKernels  # Unpack


def cupyFindLocalMaxima(
    x: cp.ndarray,
    minHeight: float,
    numOutputPerBlk: int = 32,
    THREADS_PER_BLK: int = 32,
    maxNumPeaks: int = 10000,
):
    # Check type
    cupyRequireDtype(cp.float32, x)

    # Make output
    numPeaksFound = cp.zeros(1, cp.int32)
    peakIndex = cp.zeros(maxNumPeaks, cp.int32)

    # Shared mem req
    smReq = (numOutputPerBlk + 2) * x.itemsize

    # Invoke
    NUM_BLKS = cupyGetEnoughBlocks(x.size, numOutputPerBlk)
    _findLocalMaximaKernel(
        (NUM_BLKS,),
        (THREADS_PER_BLK,),
        (
            x,
            x.size,
            np.float32(
                minHeight
            ),  # You MUST CAST it like this, else it interprets it wrongly in the kernel
            numOutputPerBlk,
            peakIndex,
            numPeaksFound,
        ),
        shared_mem=smReq,
    )

    return peakIndex, numPeaksFound


# %%
if __name__ == "__main__":
    from signalCreationRoutines import *
    from verifyRoutines import *
    from timingRoutines import *

    timer = Timer()

    # Make some small signal
    x = cp.zeros(50, cp.float32)
    x[5] = 1.0
    x[8] = 0.5
    x[9] = 0.6
    x[32 + 5] = 1.0
    x[32 + 8] = 0.5
    x[32 + 9] = 0.6
    x += cp.asarray(np.abs(np.random.randn(x.size) * 1e-3))
    print(x)

    # Run the kernel?
    peakIndex, numPeaksFound = cupyFindLocalMaxima(x, 0.4)
    pki = peakIndex.get()
    numPki = numPeaksFound.get()[0]
    print(np.sort(pki[:numPki]))

    # TODO: write peakfinding kernel unittests

    # # Create a short signal
    # x = randnoise(50, 1, 1, 1).astype(np.complex64)
    # # Create a long signal
    # y = randnoise(100000, 1, 1, 1).astype(np.complex64)

    # # Run the sliding multiply on cpu
    # startIdx = 0
    # idxlen = y.size - x.size + 1
    # out = np.zeros((idxlen, x.size), dtype=np.complex64)

    # # pre-compute x norm
    # xnorm = np.linalg.norm(x)

    # timer.start()
    # for i in range(startIdx, idxlen):
    #     outnormsq = np.linalg.norm(y[i:i+x.size])
    #     out[i,:] = y[i:i+x.size] * x / outnormsq / xnorm
    # timer.end("numpy")

    # # Run the sliding multiply on gpu
    # d_x = cp.asarray(x)
    # d_y = cp.asarray(y)

    # # Use the custom kernel
    # timer.start()
    # d_pdts = multiplySlidesNormalised(
    #     d_x,
    #     d_y,
    #     startIdx,
    #     idxlen,
    #     THREADS_PER_BLOCK=32
    # )
    # timer.end("kernel")

    # compareValues(
    #     d_pdts.get().flatten(),
    #     out.flatten()
    # )

    # # Testing the magn sq kernel
    # d_cp_yAbsSq = cp.abs(d_y)**2
    # print(d_cp_yAbsSq.dtype)
    # d_yAbsSq = cupyComplexMagnSq(d_y)
    # print(d_yAbsSq.dtype)

    # compareValues(
    #     d_yAbsSq.get().flatten(),
    #     d_cp_yAbsSq.get().flatten()
    # )
