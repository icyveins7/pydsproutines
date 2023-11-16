from xcorrRoutines import *

from timingRoutines import Timer
from verifyRoutines import compareValues

import numpy as np

timer = Timer()

def benchmark(cutoutlen=1000, numShifts=100000, cupybatchsize=4096, THREADS_PER_BLOCK=32):
    # Generate some data
    datalen = cutoutlen + numShifts - 1
    x = np.random.randn(datalen) + 1j*np.random.randn(datalen)
    x = x.astype(np.complex64)

    # Cutout a certain length
    cutout = x[:cutoutlen]

    # Run the v1 cupy xcorr
    d_cutout = cp.asarray(cutout)
    d_x = cp.asarray(x)
    # # Need a warmup for gpu
    # print("warmup v1")
    # cpout = cp_fastXcorr(
    #     d_cutout, 
    #     d_x, 
    #     freqsearch=True, 
    #     shifts=np.arange(numShifts), 
    #     BATCH=cupybatchsize)
    
    # print("starting v1")
    # timer.start()
    # cpout = cp_fastXcorr(
    #     d_cutout, 
    #     d_x, 
    #     freqsearch=True, 
    #     shifts=np.arange(numShifts), 
    #     BATCH=cupybatchsize)
    # timer.end("cp_fastxcorr v1")

    # Run the v2 cupy xcorr (warmup)
    cpout2 = cp_fastXcorr_v2(
        d_cutout,
        d_x,
        0,
        numShifts,
        THREADS_PER_BLOCK
    )

    timer.start()
    cpout2 = cp_fastXcorr_v2(
        d_cutout,
        d_x,
        0,
        numShifts,
        THREADS_PER_BLOCK
    )
    timer.end("cp_fastxcorr v2")


if __name__ == "__main__":
    benchmark()