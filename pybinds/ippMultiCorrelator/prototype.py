import numpy as np
import scipy.signal as sps

from signalCreationRoutines import *
from plotRoutines import *

closeAllFigs()

# Create some small preambles
numPreambles = 2
numPreambleSyms = 10
preamble_syms = np.empty(
    (numPreambles, numPreambleSyms),
    dtype=np.complex64
)
for i in range(numPreambles):
    syms, bits = randPSKsyms(
        numPreambleSyms, 4, dtype=np.complex64
    )
    preamble_syms[i] = syms

# Resample at some OSR
OSR = 4
preamble_rx = [
    sps.resample_poly(s, OSR, 1).astype(np.complex64)
    for s in preamble_syms
]

N = 1000
noise, x = addManySigToNoise(
    N,
    [10, 10+numPreambleSyms*OSR*2],
    preamble_rx,
    1, OSR,
    [100]*numPreambles
)

plotAmpTime([x], [1])