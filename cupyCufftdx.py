from cupyHelpers import *

import cupy as cp
import numpy as np
import os



#%%
if __name__ == "__main__":
    from verifyRoutines import compareValues

    #################### Simple test using the test kernel
    FFT_PER_BLK = 1
    FFT_SIZE = 4096
    ELEMENTS_PER_THREAD = 4
    # Note that the pair of FFT_SIZE and ELEMENTS_PER_THREAD must satisfy the max threads per block condition,
    # which is usually 1024 threads per block.
    # Hence if the length is 4096 then elements_per_thread must be at least 4.

    (cf, cfiltconv), cfModule = cupyModuleToKernelsLoader(
        "cufftdxKernels.cu", 
        ["test_kernel", "filter_sm_convolution"], 
        options=(
            '-std=c++17', 
            '-I%s' % (os.environ['CUFFTDX_INCLUDE_DIR']),
            '-DFFT_SIZE=%d' % (FFT_SIZE), # on-demand compilation for a particular length
            '-DFFT_EPT=%d' % (ELEMENTS_PER_THREAD),
            '-DFFT_PER_BLK=%d' % (FFT_PER_BLK),
        )
    )
    # Extract kernel parameters from the compilation
    block_dim = cp.ndarray((3,), cp.uint32, cfModule.get_global('fft_block_dim'))
    print(block_dim)
    smReq = cp.ndarray((1,), cp.uint32, cfModule.get_global('fft_shared_memory'))
    print(smReq)
    requires_workspace = cp.ndarray((1,), cp.uint8, cfModule.get_global('fft_requires_workspace'))
    print(requires_workspace)

    numFFTs = 2
    data = cp.arange(2*FFT_SIZE).astype(cp.complex64)
    h_data = data.get()

    # Invoke kernel
    cf(
        (numFFTs//FFT_PER_BLK,),tuple(block_dim.get()),(data),shared_mem=smReq.get()[0]
    )

    print(data)
    h_out = np.fft.fft(h_data.reshape((numFFTs, FFT_SIZE)), axis=1)
    print(h_out)

    compareValues(data.get(), h_out.reshape(-1))


    ########################### Testing the filter conv kernel
    FFT_PER_BLK = 1
    FFT_SIZE = 64
    ELEMENTS_PER_THREAD = 8
    # Note that the pair of FFT_SIZE and ELEMENTS_PER_THREAD must satisfy the max threads per block condition,
    # which is usually 1024 threads per block.
    # Hence if the length is 4096 then elements_per_thread must be at least 4.

    (cf, cfiltconv), cfModule = cupyModuleToKernelsLoader(
        "cufftdxKernels.cu", 
        ["test_kernel", "filter_sm_convolution"], 
        options=(
            '-std=c++17', 
            '-I%s' % (os.environ['CUFFTDX_INCLUDE_DIR']),
            '-DFFT_SIZE=%d' % (FFT_SIZE), # on-demand compilation for a particular length
            '-DFFT_EPT=%d' % (ELEMENTS_PER_THREAD),
            '-DFFT_PER_BLK=%d' % (FFT_PER_BLK),
        )
    )
    # Extract kernel parameters from the compilation
    block_dim = cp.ndarray((3,), cp.uint32, cfModule.get_global('fft_block_dim'))
    print(block_dim)
    smReq = cp.ndarray((1,), cp.uint32, cfModule.get_global('fft_shared_memory'))
    print(smReq)
    requires_workspace = cp.ndarray((1,), cp.uint8, cfModule.get_global('fft_requires_workspace'))
    print(requires_workspace)
    if requires_workspace.get()[0] == 1:
        raise ValueError("This test requires a workspace")

    numFFTs = 1

    import scipy.signal as sps
    taps = sps.firwin(32, 1/2.0)
    tapsfft = np.fft.fft(taps).astype(np.complex64)
    d_tapsfft = cp.array(tapsfft)

    data = cp.arange(200).astype(cp.complex64)
    h_data = data.get()

    out = cp.zeros(data.size, dtype=cp.complex64)

    # Invoke kernel
    NUM_OUT_PER_BLK = FFT_SIZE - taps.size + 1
    NUM_BLKS = cupyGetEnoughBlocks(data.size, NUM_OUT_PER_BLK)
    print("NUM_BLKS: %d" % NUM_BLKS)
    cfiltconv(
        (NUM_BLKS,),
        tuple(block_dim.get()),
        (data, data.size, d_tapsfft, d_tapsfft.size, out),
        shared_mem=smReq.get()[0]
    )
