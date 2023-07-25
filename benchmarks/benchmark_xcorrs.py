from xcorrRoutines import *

from timingRoutines import *
from verifyRoutines import compareValues

import numpy as np

def benchmark(cutoutlen=1000, cupybatchsize=1, numShifts=128):
    # Generate some data
    datalen = 100000000
    x = np.random.randn(datalen) + 1j*np.random.randn(datalen)
    x = x.astype(np.complex64)

    # Cutout a certain length
    start = 10000
    cutout = x[start:start+cutoutlen]

    timer = Timer()
    loops = 1 # No need to repeat honestly..

    # Run CPU xcorr
    startIdx = 0
    endIdx = numShifts
    idxStep = 1
    shifts = np.arange(startIdx, endIdx, idxStep)
    timer.start()
    for i in range(loops):
        out = fastXcorr(cutout, x, freqsearch=True, shifts=shifts)
        timer.evt("cpu %d" % i)
    timer.end()
    # print(out)

    # Run the cythonised xcorr
    numThreads = 4
    cyxc = CyIppXcorrFFT(cutout, numThreads)
    timer.start()
    for i in range(loops):
        cyout = cyxc.xcorr(x, startIdx, endIdx, idxStep)
        timer.evt("cython %d, %d threads" % (i, numThreads))
    timer.end()
    # print(cyout)

    # Run cupy xcorr
    d_cutout = cp.asarray(cutout)
    d_x = cp.asarray(x)
    # Need a warmup for gpu
    cpout = cp_fastXcorr(d_cutout, d_x, freqsearch=True, shifts=shifts, BATCH=cupybatchsize)
    timer.start()
    for i in range(loops):
        cpout = cp_fastXcorr(d_cutout, d_x, freqsearch=True, shifts=shifts, BATCH=cupybatchsize)
        timer.evt("cupy %d" % i)
    timer.end()

    # Compare results
    compareValues(out[0], cyout[0])
    compareValues(out[1], cyout[1])

    compareValues(out[0], cpout[0])
    compareValues(out[1], cpout[1])

#%%
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        benchmark(int(sys.argv[1]))
    elif len(sys.argv) == 3:
        benchmark(int(sys.argv[1]), int(sys.argv[2]))
    elif len(sys.argv) == 4:
        benchmark(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    else:
        benchmark()