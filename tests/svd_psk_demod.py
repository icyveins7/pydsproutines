from signalCreationRoutines import *
from plotRoutines import *

import numpy as np

closeAllFigs()

# Generate signal with arbitrary
# phase, freq
psk, _ = randPSKsyms(1000, 4)
pskf = psk * np.exp(1j*2*np.pi*np.random.rand()*0.002 * np.arange(psk.size))

_, rx = addSigToNoise(psk*np.exp(1j*np.random.rand()
                      * np.pi/4), snr_inband_linear=10)
_, rxf = addSigToNoise(pskf*np.exp(1j*np.random.rand()
                       * np.pi/4), snr_inband_linear=10)

ax = plotConstellation(rx)
axf = plotConstellation(rxf)

# Map to 3d with new variable
rxr = np.pad(rx.view(np.float64).reshape((-1, 2)),
             [(0, 0), (0, 1)])
rxfr = np.pad(rxf.view(np.float64).reshape((-1, 2)),
              [(0, 0), (0, 1)])


# Define some common mapping
def map3rd(a):
    # a[:, 2] = a[:, 1]  # Just use y?
    a[:, 2] = a[:, 1] - a[:, 0]  # y-x?


map3rd(rxr)
map3rd(rxfr)

# Plot in 3d to see
fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')
ax1.scatter(rxr[:, 0], rxr[:, 1], rxr[:, 2])
