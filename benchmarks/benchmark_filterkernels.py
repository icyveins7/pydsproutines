"""
Early results:

Note that the order of kernels called affects the timer.start/end() timing.
As such we report nvprof kernel durations here.

96.256us filter_smtaps
64.512us filter_smtaps_sminput
78.656us filter_smtaps with 4x downsample 
    (this is not a 4x improvement! probably a lot of global mem reads,
     but this would allow us to skip an expensive copy using cp.ascontiguousarray(),
     which is necessary after manually downsampling the output)
"""

from filterRoutines import *
from verifyRoutines import compareValues
from timingRoutines import *
from signalCreationRoutines import *

import numpy as np
import cupy as cp
import scipy.signal as sps

timer = Timer()

# Generate some noise
length = 100000
noise = randnoise(length, 1.0, 1.0, 1.0).astype(np.complex64)
d_noise = cp.asarray(noise, cp.complex64)

# And some taps
taps = sps.firwin(128, 0.5).astype(np.float32)
d_taps = cp.asarray(taps, cp.float32)

# Filter the noise (CPU)
timer.start()
cpu_filt = sps.lfilter(taps, 1.0, noise)
timer.end("cpu")

# Instantiate the CupyKernelFilter class
cpkf = CupyKernelFilter()

# Filter the noise (GPU)
timer.start()
d_filt_smtaps = cpkf.filter_smtaps(d_noise, d_taps)
timer.end("gpu smtaps")

# # Filter the noise (GPU, sm input)
# timer.start()
# d_filt_smtaps_sminput = cpkf.filter_smtaps_sminput(d_noise, d_taps)
# timer.end("gpu smtaps sminput")

# # Check results
# compareValues(cpu_filt, d_filt_smtaps.get())
# compareValues(cpu_filt, d_filt_smtaps_sminput.get())

# # Try to filter with downsampling
# dsr = 4
# dsphase = 1
# timer.start()
# d_filt_smtaps_ds = cpkf.filter_smtaps(d_noise, d_taps, dsr=dsr, dsPhase=dsphase)
# timer.end("gpu smtaps downsample")

# Check results
# compareValues(cpu_filt[dsphase::dsr], d_filt_smtaps_ds.get())






