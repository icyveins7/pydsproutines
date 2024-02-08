"""
Compare against the kernel using filter taps like
filter_smtaps_sminput.

Simple benchmark for 10M elements:
filter_smtaps_sminput_real 128 TPB, 256 OPB,  1.259 ms
movingAverageKernel          33 NPT, 32 TPB,   472 us (this seems to be about 1/4 of copy performance, not too bad)
"""

from filterRoutines import *
from verifyRoutines import compareValues
from timingRoutines import *
from signalCreationRoutines import *

import numpy as np
import cupy as cp
import cupyx.scipy.signal as cpxsps
import scipy.signal as sps

timer = Timer()

# Generate a long signal
length = 10000000
x = np.random.randn(length).astype(np.float32) + 2.0

# Generate taps for averaging window
avgLength = 100
avgTaps = np.ones(avgLength).astype(np.float32) / avgLength

# Move to GPU
d_x = cp.asarray(x, cp.float32)
d_avgTaps = cp.asarray(avgTaps, cp.float32)

# Use old filter kernel
cpkf = CupyKernelFilter()
d_old = cpkf.filter_smtaps_sminput(
    d_x, d_avgTaps,
    THREADS_PER_BLOCK=128,
    OUTPUT_PER_BLK=256
)

# Use new filter kernel specific to moving average
d_new = cupyMovingAverage(
    d_x, avgLength,
    NUM_PER_THREAD=33,
    THREADS_PER_BLK=32
)

compareValues(d_old.get(), d_new.get())




