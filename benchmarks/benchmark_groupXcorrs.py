from xcorrRoutines import *

from timingRoutines import *
from verifyRoutines import compareValues
from signalCreationRoutines import *

import numpy as np

import sys
if sys.platform.startswith('win32'):
    import os
    os.add_dll_directory(os.path.join(os.environ['IPPROOT'], "redist", "intel64"))

from pbIppGroupXcorrCZT import pbIppGroupXcorrCZT


def benchmark(*args, **kwargs):
    timer = Timer()

    # Create some data
    x, _ = randPSKsyms(1000000, 4, dtype=np.complex64)

    # Define freq search space
    f1 = -100.0
    f2 = 100.0
    fstep = 1.0
    fs = 10000

    # Define a few groups
    firstGroupStart = 100
    groupLength = 5000
    groupStarts = np.arange(firstGroupStart, x.size, groupLength * 2, dtype=np.int32)
    print("Total %d groups" % (groupStarts.size))

    # Create a pythonic group xcorr czt object
    timer.start()
    gxc = GroupXcorrCZT(x, groupStarts,
                        np.zeros(groupStarts.size, dtype=np.int32) + groupLength,
                        f1, f2, fstep, fs)
    timer.evt("Preparing py group xcorr czt object")
    shiftStart = 80
    shiftStep = 1
    numShifts = 41
    results, cztfreq = gxc.xcorr(x, np.arange(shiftStart, shiftStart+numShifts, shiftStep))
    timer.end("Xcorring using py group xcorr czt object")
    print(results.shape)

    # Create the pybind equivalent
    timer.start()
    pbgxc = pbIppGroupXcorrCZT(groupLength, f1, f2, fstep, fs)
    num_threads = pbgxc.getNumThreads()
    timer.evt("Preparing pybind group xcorr czt object (%d threads)" % (num_threads))
    for gs in groupStarts:
        pbgxc.addGroup(gs-firstGroupStart,
                       x[gs:gs+groupLength])
    timer.evt("Adding groups")
    pbresults = pbgxc.xcorr(x, shiftStart, shiftStep, numShifts)
    timer.end("Xcorring using pybind group xcorr czt object (%d threads)" % (num_threads))

    # Create the pybind equivalent with more threads
    timer.start()
    pbgxc = pbIppGroupXcorrCZT(groupLength, f1, f2, fstep, fs, 4)
    num_threads = pbgxc.getNumThreads()
    timer.evt("Preparing pybind group xcorr czt object (%d threads)" % (num_threads))
    for gs in groupStarts:
        pbgxc.addGroup(gs-firstGroupStart,
                       x[gs:gs+groupLength])
    timer.evt("Adding groups")
    pbresults = pbgxc.xcorr(x, shiftStart, shiftStep, numShifts)
    timer.end("Xcorring using pybind group xcorr czt object (%d threads)" % (num_threads))

    compareValues(results.flatten(), pbresults.flatten())



#%%
if __name__ == "__main__":
    benchmark()