import numpy as np
import scipy as sp
import scipy.signal as sps

from signalCreationRoutines import *
from timingRoutines import *

def manualRationalResample(x, taps, up, down):
    # Upsample first
    us = np.zeros(x.size*up, dtype=x.dtype)
    us[::up] = x[:]

    # Filter
    f = sps.lfilter(taps, 1, us)
    
    # Downsample
    ds = f[::down]

    return ds, f

#%%
def benchmark(
    fs=15000,
    up=5000,
    down=3,
    numTaps=10000
):
    # Create signal
    data = np.arange(fs).astype(np.complex64)

    # Create filter taps
    taps = sps.firwin(numTaps, 1/up)

    print("Expected length of resample is %d" % (data.size*up//down))

    # Perform via scipy's resample
    x = sps.upfirdn(taps, data, up, down)
    # Note that upfirdn does a standard convolution,
    # which filters up to beyond the size of data
    print(x)
    print(x.size)

    # Note that resample_poly does some weird preprocessing?
    # Currently not understood how to replicate this without diving into source
    xrs = sps.resample_poly(data, up, down, window=taps)
    print(xrs)
    print(xrs.size)

    # Check with manual known rational resampler
    xmrs, xmf = manualRationalResample(data, taps, up, down)
    print(xmf)
    print(xmrs)
    print(xmrs.size)

    # assert(np.all(x[:xrs.size] == xrs))



#%%
if __name__ == "__main__":
    import os
    benchmark(fs=15, up=5, numTaps=10)