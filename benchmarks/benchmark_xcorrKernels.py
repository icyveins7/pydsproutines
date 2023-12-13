from xcorrRoutines import *

from timingRoutines import Timer
from verifyRoutines import compareValues

import numpy as np

timer = Timer()

"""
Notes on optimisation for the new optimised kernel.
It is important to increase THREADS_PER_BLOCK somewhat to around 128 threads,
but not too high (256) seems to make it slower!

Setting a non-maximum numSlidesPerBlk also increases speed substantially!

Some measurements from nsys profiling:
128 threads, 3872 slides (maxed), 6.3ms (only 26 blocks, which is bad occupancy for SMs)
128 threads, 1024 slides, 3.05ms (98 blocks, decent occupancy)
128 threads, 512 slides, 3.1ms (suggests that 98 blocks is good enough occupancy)
256 threads, 1024 slides, 4.7ms (maybe too much load on SMs at this point?)
"""

def benchmark(cutoutlen=1000, numShifts=100000, cupybatchsize=4096, 
              THREADS_PER_BLOCK=128, numSlidesPerBlk=1024):
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
        THREADS_PER_BLOCK,
        numSlidesPerBlk
    )

    timer.start()
    cpout2 = cp_fastXcorr_v2(
        d_cutout,
        d_x,
        0,
        numShifts,
        THREADS_PER_BLOCK,
        numSlidesPerBlk
    )
    timer.end("cp_fastxcorr v2")


if __name__ == "__main__":
    import argparse
    # Generate commandline args for the benchmark function
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoutlen', default=1000, type=int)
    parser.add_argument('--numShifts', default=100000, type=int)
    parser.add_argument('--cupyBatchSize', default=4096, type=int)
    parser.add_argument('--threadsPerBlk', default=128, type=int)
    parser.add_argument('--numSlidesPerBlk', default=1024, type=int)

    args = parser.parse_args()
    print(args)
    benchmark(
        args.cutoutlen, 
        args.numShifts,
        args.cupyBatchSize, 
        args.threadsPerBlk, 
        args.numSlidesPerBlk
    )