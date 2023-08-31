from xcorrRoutines import *

from timingRoutines import *
from verifyRoutines import compareValues
from signalCreationRoutines import *

import numpy as np

import sys
if sys.platform.startswith('win32'):
    import os
    os.add_dll_directory(os.path.join(os.environ['IPPROOT'], "redist", "intel64"))


def benchmark(*args, **kwargs):
    timer = Timer()

    # Create some data
    x, _ = randPSKsyms(100000, 4, dtype=np.complex64)

    # Define freq search space
    f1 = -100.0
    f2 = 100.0
    fstep = 1.0
    fs = 10000

    # Define a few groups
    groupLength = 5000
    groupStarts = np.arange(100, x.size, groupLength * 2)

    # Create a pythonic group xcorr czt object
    timer.start()
    gxc = GroupXcorrCZT(x, groupStarts,
                        np.zeros(groupStarts.size) + groupLength,
                        f1, f2, fstep, fs)
    timer.evt("Preparing py group xcorr czt object")
    results, cztfreq = gxc.xcorr(x, np.arange(90, 111))
    timer.end("Xcorring using py group xcorr czt object")

    




#%%
if __name__ == "__main__":
    benchmark()