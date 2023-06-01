from signalCreationRoutines import *


import numpy as np

numIter = 2
sigPwr = 100.0

# Randomise m = 2 or 4
# m_arr = (np.random.randint(0, 2, numIter) + 1) * 2
m_arr = np.array([2, 4])

# Create a bunch of signals
length = 10000
sigs = np.zeros((numIter, length), dtype=np.complex128)

for i in range(numIter):
    syms, bits = randPSKsyms(length, m_arr[i])
    noise, sig = addSigToNoise(length, 0, syms * sigPwr**0.5, snr_inband_linear=10.0, sigPwr=sigPwr)
    sigs[i, :] = sig

# Test the signals
# from demodulationRoutines import *
# m_detect = SimpleDemodulatorPSK.detect_B_or_Q(sigs)
# print(m_arr)
# print(m_detect)

# Try using the cm20/cm40 methods to estimate
print(sigs.shape)
from plotRoutines import *
for i in range(numIter):
    print(m_arr[i])
    cm20 = np.abs(np.fft.fft(sigs[i]**2))
    cm40 = np.abs(np.fft.fft(sigs[i]**4))
    cm20mi = np.argmax(cm20)
    print(cm20mi)
    cm20max = cm20[cm20mi]
    cm40mi = np.argmax(cm40)
    print(cm40mi)
    cm40max = cm40[cm40mi]
    # Value of max is essentially A^2 * N^2, A^4 * N^2
    print(cm20max)
    print(cm40max)
    print((cm20max / length)**2 * length)
    print(" ")

    plotSpectra([sigs[i]**2, sigs[i]**4], [1, 1])

plt.show()