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

timer.start()
mpl = mp.compute(d_x)
timer.end()

print("Complete.")

"""
Some stats for 100e3 length, 10 windowLength:
Internal python timer: 18s.
nsys profile: 22s.
"""
