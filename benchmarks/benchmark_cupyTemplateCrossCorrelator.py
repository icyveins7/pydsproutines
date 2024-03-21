from xcorrRoutines import TemplateCrossCorrelator
from signalCreationRoutines import randPSKsyms
from plotRoutines import *

import cupy as cp
import numpy as np
import argparse

closeAllFigs()


def benchmark(
    length,
    cutoutlen,
    cutoutstart,
    cutoutjump,
    numCutouts,
    plotOn
):
    # Create signal and cutout
    x, _ = randPSKsyms(length, 4, dtype=np.complex64)
    cutouts = np.zeros((numCutouts, cutoutlen), dtype=x.dtype)
    for i in range(numCutouts):
        cutouts[i] = x[cutoutstart+cutoutjump *
                       i:cutoutstart+cutoutjump*i+cutoutlen]

    # Move to GPU
    dx = cp.asarray(x)
    dcutouts = cp.asarray(cutouts)

    # Create class
    correlator = TemplateCrossCorrelator(dcutouts, dx.size)
    out, ti = correlator.correlate(dx, returnMax=True)

    # Run a few more times
    for i in range(3):
        out, ti = correlator.correlate(dx, returnMax=True)

    if plotOn:
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(out.get())
        ax[1].plot(ti.get())

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", default=1000000)
    parser.add_argument("--cutoutlen", default=200)
    parser.add_argument("--cutoutstart", default=1000)
    parser.add_argument("--cutoutjump", default=1000)
    parser.add_argument("--numcutouts", default=5)
    parser.add_argument("--plot", default=False)

    args = parser.parse_args()

    benchmark(args.length, args.cutoutlen, args.cutoutstart,
              args.cutoutjump, args.numcutouts, args.plot)
