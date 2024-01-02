from cupyExtensions import *

from xcorrRoutines import *
from signalCreationRoutines import *
from plotRoutines import *

from timingRoutines import Timer
from verifyRoutines import compareValues

import numpy as np

timer = Timer()

"""
Initial tests suggest using as small THREADS_PER_BLOCK as possible,
so default to 32 (especially try to be less than templateLength).

Seems like for 20 templates at length 100, with 10M slides,
this takes about 260ms.

TODO: more scenarios for comparison.
"""

def benchmark(
    idxlen: int=100000,
    numTemplates: int=10,
    templateLength: int=100,
    THREADS_PER_BLOCK: int=32,
    numSlidesPerBlk: int=100
):
    # Create signal
    templates = np.zeros((numTemplates, templateLength), dtype=np.complex64)
    for i in range(numTemplates):
        syms, bits = randPSKsyms(
            templateLength, 4, dtype=cp.complex64
        )
        templates[i,:] = syms

    # Add to long input
    sigStartIdxList = np.arange(numTemplates)*2*templateLength
    print(sigStartIdxList)
    _, rx = addManySigToNoise(
        idxlen+templateLength-1,
        sigStartIdxList,
        templates,
        1, 1,
        np.zeros(numTemplates) + 10
    )
    # print(rx.size)

    # Move to GPU
    d_x = cp.asarray(rx, dtype=cp.complex64)
    d_templates = cp.asarray(templates, dtype=cp.complex64)

    # Run the kernel
    d_templateIdx, d_qf2 = multiTemplateSlidingDotProduct(
        d_x,
        d_templates.conj(),
        0, idxlen,
        numSlidesPerBlk=numSlidesPerBlk,
        THREADS_PER_BLOCK=THREADS_PER_BLOCK
    )

    return d_x, d_templates, d_templateIdx, d_qf2, sigStartIdxList



if __name__ == "__main__":

    import argparse
    # Generate commandline args for the benchmark function
    parser = argparse.ArgumentParser()
    parser.add_argument('--templateLength', default=100, type=int)
    parser.add_argument('--numTemplates', default=10, type=int)
    parser.add_argument('--numShifts', default=100000, type=int)
    parser.add_argument('--threadsPerBlk', default=128, type=int)
    parser.add_argument('--numSlidesPerBlk', default=100, type=int)
    parser.add_argument('--plot', default=True, type=bool)

    args = parser.parse_args()
    print(args)

    d_x, d_templates, d_templateIdx, d_qf2, sigStartIdxList = benchmark(
        args.numShifts,
        args.numTemplates,
        args.templateLength, 
        args.threadsPerBlk, 
        args.numSlidesPerBlk
    )
    
    if args.plot:
        rx = d_x.get()
        win, ax = pgPlotAmpTime(
            [rx/np.max(np.abs(rx)), d_qf2.get()],
            [1, 1]
        )
        tax = win.addPlot(row=1, col=0)
        tax.setXLink(ax)
        tax.plot(d_templateIdx.get())
        tax.plot(sigStartIdxList, d_templateIdx.get()[sigStartIdxList], pen=None, symbol='x', symbolPen='r')
