from matrixProfileRoutines import CupyMatrixProfile
from signalCreationRoutines import randPSKsyms
from timingRoutines import Timer

import numpy as np
import cupy as cp

windowLength = 10

mp = CupyMatrixProfile(windowLength)

# Make some data
x, bits = randPSKsyms(100000, 4, dtype=np.complex64)
d_x = cp.asarray(x)

# Compute the matrix profile
timer = Timer()

"""
Some stats for 100e3 length, 10 windowLength:
Internal python timer: 18s.
nsys profile: 22s.
Note that this completely pummels the VRAM, so it gets throttled
when it starts to run out of memory and virtualises some of it to the CPU.
"""
# timer.start()
# mpl = mp.compute(d_x)
# timer.end()

"""
Benchmarking like this prevents the storage of the outputs, but still
performs the compute.
Some stats for 100e3 length, 10 windowLength:
Internal python timer: 9s.
"""
timer.start()
mp._normsSq = mp._computeNormsSq(d_x)
for i in range(1, x.size - mp._windowLength + 1):
    diag = mp._computeDiagonal(d_x, i)
timer.end()

print("Complete.")
