from xcorrRoutines import *

from timingRoutines import *
from verifyRoutines import compareValues

import numpy as np

# Generate some data
datalen = 10000000
x = np.random.randn(datalen) + 1j*np.random.randn(datalen)
x = x.astype(np.complex64)

# Cutout a certain length
cutoutlen = 1000
start = 10000
cutout = x[start:start+cutoutlen]

timer = Timer()
loops = 5

# Run CPU xcorr
startIdx = 0
endIdx = 20000
idxStep = 1
shifts = np.arange(startIdx, endIdx, idxStep)
timer.start()
for i in range(loops):
    out = fastXcorr(cutout, x, freqsearch=True, shifts=shifts)
    timer.evt("cpu %d" % i)
timer.end()
print(out)

# Run the cythonised xcorr
cyxc = CyIppXcorrFFT(cutout, 4)
timer.start()
for i in range(loops):
    cyout = cyxc.xcorr(x, startIdx, endIdx, idxStep)
    timer.evt("cython %d" % i)
timer.end()
print(cyout)

# Compare results
compareValues(out[0], cyout[0])
compareValues(out[1], cyout[1])