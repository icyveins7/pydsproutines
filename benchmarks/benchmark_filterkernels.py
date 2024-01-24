"""
Results from nsys profile:
For length 1000000
670us cupyx.scipy.signal.convolve; time from first regular_fft to end of true_divide
317us filter_smtaps
230us filter_smtaps_sminput, 128 per blk
240us filter_smtaps_sminput, 1024 per blk

Generally, filter_smtaps_sminput kernel uses less blocks, and hence the occupancy for the SMs may be low.
This causes it to be slower when the input length is small, but it gains performance
and crosses over the filter_smtaps kernel at about 1M samples.

Also, it's better for the filter_smtaps_sminput kernel to use less per block; probably due to better SMs work balancing since
there will be more blocks.

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

# Generate some noise
length = 1000000
noise = randnoise(length, 1.0, 1.0, 1.0).astype(np.complex64)
d_noise = cp.asarray(noise, cp.complex64)

# And some taps
taps = sps.firwin(128, 0.5).astype(np.float32)
d_taps = cp.asarray(taps, cp.float32)

# Filter the noise (CPU)
timer.start()
cpu_filt = sps.lfilter(taps, 1.0, noise)
timer.end("cpu")


# Filter using cupy's convolve
d_cupy_conv = cpxsps.convolve(d_taps, d_noise) # Run once to make it warm
timer.start()
d_cupy_conv = cpxsps.convolve(d_taps, d_noise) # note that .lfilter isn't available until cupy > 13.0, so use convolve manually
# Note that convolve, similar to numpy, will produce longer output
# we need the front part only but we won't time this: d_cupy_conv[:d_noise.size]
timer.end("cupy convolve")

# Instantiate the CupyKernelFilter class
cpkf = CupyKernelFilter()

# Filter the noise (GPU)
timer.start()
d_filt_smtaps = cpkf.filter_smtaps(d_noise, d_taps)
timer.end("gpu smtaps")

# Filter the noise (GPU, sm input)
timer.start()
d_filt_smtaps_sminput = cpkf.filter_smtaps_sminput(d_noise, d_taps)
timer.end("gpu smtaps sminput 128 per blk")

# Do again, but use more per block
timer.start()
d_filt_smtaps_sminput = cpkf.filter_smtaps_sminput(d_noise, d_taps, OUTPUT_PER_BLK=1024)
timer.end("gpu smtaps sminput 1028 per blk")

# Check results
compareValues(cpu_filt, d_cupy_conv.get()[:cpu_filt.size])
compareValues(cpu_filt, d_filt_smtaps.get())
compareValues(cpu_filt, d_filt_smtaps_sminput.get())

# # Try to filter with downsampling
# dsr = 4
# dsphase = 1
# timer.start()
# d_filt_smtaps_ds = cpkf.filter_smtaps(d_noise, d_taps, dsr=dsr, dsPhase=dsphase)
# timer.end("gpu smtaps downsample")

# Check results
# compareValues(cpu_filt[dsphase::dsr], d_filt_smtaps_ds.get())






