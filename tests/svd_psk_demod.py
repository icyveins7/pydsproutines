"""
Doesn't seem like i can find any good 3rd dimension mapping..
Gotta find some other way?
Gonna leave this here as a reminder.
"""

from signalCreationRoutines import randPSKsyms, addSigToNoise
from plotRoutines import plotConstellation, closeAllFigs

import numpy as np
import matplotlib.pyplot as plt

closeAllFigs()

# Generate signal with arbitrary
# phase, freq
psk, _ = randPSKsyms(1000, 4)
pskf = psk * np.exp(1j*2*np.pi*np.random.rand()*0.0001 * np.arange(psk.size))

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
    # a[:, 2] = a[:, 1] - a[:, 0]  # y-x?
    a[:, 2] = a[:, 0] * a[:, 1]  # x * y?


map3rd(rxr)
map3rd(rxfr)

# Plot in 3d to see
fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')
ax1.scatter(rxr[:, 0], rxr[:, 1], rxr[:, 2])
ax1.set_aspect('equal')

u, s, v = np.linalg.svd(rxr.T @ rxr)
for i in range(3):
    ax1.plot([0, u[0, i]],
             [0, u[1, i]],
             [0, u[2, i]])


fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
ax2.scatter(rxfr[:, 0], rxfr[:, 1], rxfr[:, 2])
ax2.set_aspect('equal')

uf, sf, vf = np.linalg.svd(rxfr.T @ rxfr)
for i in range(3):
    ax2.plot([0, uf[0, i]],
             [0, uf[1, i]],
             [0, uf[2, i]])

# Check for multiple freq shifts


def metric(x):
    xr = np.pad(x.view(np.float64).reshape((-1, 2)), [(0, 0), (0, 1)])
    map3rd(xr)
    _, s, _ = np.linalg.svd(xr.T @ xr)
    return s[2] / (s[0] + s[1])


fshifts = np.linspace(-0.0001, 0.0001, 1000)
metric1 = [metric(rx * np.exp(1j*2*np.pi*f*np.arange(rx.size)))
           for f in fshifts]
metric2 = [metric(rxf * np.exp(1j*2*np.pi*f*np.arange(rxf.size)))
           for f in fshifts]

plt.figure()
plt.plot(fshifts, metric1, label='rx')
# plt.plot(fshifts, metric2, label='rxf')


plt.show()
