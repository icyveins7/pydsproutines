from signalCreationRoutines import *
from xcorrRoutines import *

import numpy as np
import matplotlib.pyplot as plt

#%% Generate a base signal
syms, bits = randPSKsyms(1000, 4)

#%% Create two separate received signals
snr1 = 30.0
snr2 = 10.0
noise1, sig1 = addSigToNoise(syms.size, 0, syms, snr_inband_linear=snr1)
noise2, sig2 = addSigToNoise(syms.size, 0, syms, snr_inband_linear=snr2)

# Compare with theoretical value
theoreticalEffSNR = expectedEffSNR(snr1, snr2)

#%% Shift by RFOA
alist = np.logspace(-14,-6)
effSNRs = np.zeros(len(alist))
for i, a in enumerate(alist):
    rfoa = np.exp(1j*2*np.pi*0.5*a*np.arange(syms.size)**2)
    rx1 = sig1*rfoa
    rx2 = sig2

    #%% Calculate the QF2
    qf2 = calcQF2(rx1, rx2)
    # Convert to effective SNR
    effSNRs[i] = convertQF2toEffSNR(qf2)

    print("Measured effective SNR: %f" % (effSNRs[i]))
    print("Theoretical effective SNR: %f" % (theoreticalEffSNR))

#%% Plot results
xp = alist # This shows that the effSNR is not constant with the length of the signal
plt.plot(xp, effSNRs)
plt.hlines([theoreticalEffSNR], 0, xp[-1], linestyles='dashed', label='Theoretical')
plt.xlabel("Normalised RFOA constant")
plt.ylabel("Effective SNR")
plt.title("Length %d" % (syms.size))
plt.legend()
plt.show()
