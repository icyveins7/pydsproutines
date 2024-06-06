from matrixProfileRoutines import CupyMatrixProfile
from signalCreationRoutines import randPSKsyms, addManySigToNoise
from timingRoutines import Timer

import numpy as np
import cupy as cp

windowLength = 10
length = 100000

mp = CupyMatrixProfile(windowLength)

# Make some data
syms, bits = randPSKsyms(100, 4, dtype=np.complex64)

# Repeat it many times into a long array
sigStartIdxList = np.arange(1, length//200, 2) * syms.size
signalList = [syms for i in sigStartIdxList]
snrList = [100.0 for i in sigStartIdxList]
noise, x = addManySigToNoise(
    length,
    sigStartIdxList,
    signalList,
    1.0, 1.0, snrList
)

d_x = cp.asarray(x).astype(cp.complex64)

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
# timer.start()
# mp._normsSq = mp._computeNormsSq(d_x)
# for i in range(1, x.size - mp._windowLength + 1):
#     diag = mp._computeDiagonal(d_x, i)
# timer.end("cupy raw diagonals (compute-only)")

"""
Finally we benchmark with the chains instead
"""
mpc = CupyMatrixProfile(
    windowLength, True, 0.9, 0
)
timer.start()
chains = mpc.compute(d_x)
timer.end()

print("Complete.")
