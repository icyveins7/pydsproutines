from filterRoutines import *
from verifyRoutines import *
from signalCreationRoutines import *
from plotRoutines import *

closeAllFigs()

import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt

# Test iterations
testIterations = 10
for test in range(testIterations):

    # Generate random data
    numRows = 100
    length = 1000
    h_x = randnoise(numRows * length, 1, 1, 1).reshape((numRows, length)).astype(np.complex64)
    d_x = cp.asarray(h_x)
    
    # Generate taps
    h_taps = sps.firwin(128, 0.2).astype(np.float32)
    d_taps = cp.asarray(h_taps)
    
    # Instantiate kernel filter
    cpkf = CupyKernelFilter()
    up = 5
    down = 3
    
    # Run the upfirdn over each row manually
    d_manualout = cp.zeros(
        (numRows, cpkf.getUpfirdnSize(d_x.shape[1], d_taps.size, up, down)),
        dtype=cp.complex64)
    
    for i in range(numRows):
        cpkf.upfirdn_naive(
            d_x[i,:],
            d_taps,
            up, down,
            d_out=d_manualout[i,:]
        )
    # print(d_manualout.get())
    
    # Run the shared mem kernel for the entire matrix
    d_out = cp.zeros_like(d_manualout)
    cpkf.upfirdn_sm(d_x, d_taps, up, down, d_out=d_out)
    # print(d_out.get())
    
    # Check with the cpu version
    h_out = []
    for i in range(numRows):
        h_out.append(sps.upfirdn(h_taps, h_x[i,:], up, down))
    h_out = np.vstack(h_out)
    
    # Verify
    print("==== Comparing CPU to GPU (single row) ====")
    compareValues(h_out.reshape(-1), d_manualout.reshape(-1).get())
    print("==== Comparing GPU (single row) to GPU (all rows) ====")
    rawChg, fracChg = compareValues(d_manualout.reshape(-1).get(), d_out.reshape(-1).get())
    print("\nGreatest changes:\n%g raw\n%g frac" % (rawChg, fracChg))
    assert(fracChg < 1e-4)
