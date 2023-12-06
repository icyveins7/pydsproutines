from cupyHelpers import *

import cupy as cp
import numpy as np
import os



#%%
if __name__ == "__main__":
    from verifyRoutines import compareValues

    # Simple test using the test kernel
    FFT_PER_BLK = 1
    FFT_SIZE = 4096
    ELEMENTS_PER_THREAD = 4

    (cf,), cfModule = cupyModuleToKernelsLoader(
        "cufftdxKernels.cu", 
        ["test_kernel"], 
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