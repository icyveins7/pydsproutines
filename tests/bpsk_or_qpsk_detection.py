from signalCreationRoutines import *
from demodulationRoutines import *

import numpy as np

numIter = 1000

# Randomise m = 2 or 4
m_arr = (np.random.randint(0, 2, numIter) + 1) * 2

# Create a bunch of signals
length = 10000
sigs = np.zeros((numIter, length), dtype=np.complex128)

for i in range(numIter):
    syms, bits = randPSKsyms(length, m_arr[i])
    noise, sig = addSigToNoise(length, 0, syms, snr_inband_linear=10.0)
    sigs[i, :] = sig

# Test the signals
m_detect = SimpleDemodulatorPSK.detect_B_or_Q(sigs)
print(m_arr)
print(m_detect)
